"""Audited finite-memory comparison on unchanged E1c validation trajectories."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .config import config_hash
from .run_logistics_e1b import ROOT, _json, _sha
from .run_temporal_e1d import aggregate, EVENT_METRICS, POINT_METRICS
from .finite_memory import calibrate_sma, score_sma, coverage_metrics, coverage_events

GROUPS = ["scenario", "method", "temporal", "view", "regime", "window", "support"]
EXTRA = ["native_score_coverage", "common_row_fraction", "all_timeline_recall",
         "all_timeline_f1", "unavailable_positive_fraction"]
METRICS = POINT_METRICS + EXTRA
EVENTS = EVENT_METRICS + ["scoreable_fraction", "fully_unavailable"]
BASELINES = {"point": "a1", "ewma": "a0.2"}


def audited_source(path: Path) -> dict[str, Any]:
    """Verify every source output and require synthetic reserved-test metadata."""
    meta = json.loads((path / "metadata.json").read_text())
    if meta["data_source"] != "synthetic" or meta["test_evaluated"]:
        raise ValueError("Need synthetic source with test reserved.")
    for name, expected in meta["output_sha256"].items():
        artifact = (path / name).resolve()
        if not artifact.is_relative_to(path.resolve()) or _sha(artifact) != expected:
            raise ValueError(f"Source artifact integrity failure: {name}")
    return meta


def validate_config(config: dict, source_config: dict, baseline_config: dict) -> None:
    """Keep methods, data, nominal budget and post-event definition comparable."""
    if config.get("data_source") != "synthetic":
        raise ValueError("Synthetic-only protocol.")
    for key in ("seeds", "methods", "scenarios"):
        if not config[key] or len(config[key]) != len(set(config[key])):
            raise ValueError("Unique nonempty lists required.")
        if not set(config[key]) <= set(baseline_config[key]):
            raise ValueError("Baseline inputs do not cover requested configuration.")
    if set(config["methods"]) != {"regime_marginal", "regime_conditional"}:
        raise ValueError("Both residual methods required.")
    if type(config["window_size"]) is not int or config["window_size"] < 1:
        raise ValueError("Positive integer window required.")
    if not {1., .2} <= set(baseline_config["alphas"]):
        raise ValueError("Missing point/EWMA baselines.")
    if (config["threshold_quantile"] != baseline_config["threshold_quantile"]
            or config["threshold_quantile"] != source_config["threshold_quantile"]):
        raise ValueError("Keep source nominal quantile budget.")
    if (config["post_event_steps"] != baseline_config["post_event_steps"]
            or type(config["post_event_steps"]) is not int or config["post_event_steps"] < 1):
        raise ValueError("Keep source tail definition.")


def paired_differences(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    """Return seed-paired SMA minus each fixed comparator, with exact key matching."""
    all_rows = []
    a = frame[frame.temporal == "sma"].set_index(keys)[metrics].sort_index()
    for baseline in BASELINES:
        b = frame[frame.temporal == baseline].set_index(keys)[metrics].sort_index()
        if not a.index.equals(b.index):
            raise ValueError("Unmatched comparison rows.")
        delta = (a-b).reset_index()
        delta["comparison"] = f"sma_minus_{baseline}"
        all_rows.append(delta)
    return pd.concat(all_rows, ignore_index=True)


def plot_comparison(summary: pd.DataFrame, out: Path) -> None:
    """Plot detection/tail tradeoffs and actual coverage without hiding warmup."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    base = summary[(summary.regime == "all") & (summary.window == "all") & (summary.support == "common")
                   & (summary.view == "observed")]
    candidates = [("steady_m05", .5), ("steady_m1", 1), ("steady_m2_r10_k1", 2), ("steady_m4", 4)]
    for (method, temporal), group in base.groupby(["method", "temporal"]):
        selected = [(name, x) for name, x in candidates if name in set(group.scenario)]
        if not selected:
            continue
        names, x = zip(*selected)
        points = group.set_index("scenario").loc[list(names)]
        for ax, metric, label in zip(axes, ["all_timeline_recall", "post_event_fpr", "native_score_coverage"],
                                     ["Recall (all time steps)", "Post-event FPR (common support)", "Native score coverage"]):
            ax.errorbar(x, points[f"{metric}_mean"], yerr=points[f"{metric}_std"], marker="o",
                        capsize=2, label=f"{method.removeprefix('regime_')} / {temporal}")
            ax.set(xlabel="Bias / within-regime std", ylabel=label)
            ax.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    axes[2].set_ylim(0, 1.05)
    fig.suptitle("Synthetic validation; five-seed SD; same nominal q99, not matched realized FPR")
    fig.savefig(out/"finite_memory.png", dpi=180)
    fig.savefig(out/"finite_memory.pdf")
    plt.close(fig)


def run_finite_memory(config_path: str | Path, *, output_root: str | Path | None = None,
                      run_id: str | None = None) -> Path:
    """Reuse frozen baselines; calibrate SMA and save coverage-aware paired evidence."""
    config = yaml.safe_load(Path(config_path).read_text())
    source, baseline = (ROOT/config[k] for k in ("source_run", "baseline_run"))
    source_meta, baseline_meta = audited_source(source), audited_source(baseline)
    source_config, baseline_config = (yaml.safe_load((p/"config.yaml").read_text()) for p in (source, baseline))
    validate_config(config, source_config, baseline_config)
    if ((ROOT/baseline_meta["source_run"]).resolve() != source.resolve()
            or baseline_meta["source_metadata_sha256"] != _sha(source/"metadata.json")
            or (ROOT/baseline_config["source_run"]).resolve() != source.resolve()):
        raise ValueError("Baseline does not reference the same source data.")
    identifier = run_id or datetime.now(timezone.utc).strftime("finite_%Y%m%dT%H%M%S%fZ")
    if Path(identifier).name != identifier or identifier in {".", ".."}:
        raise ValueError("run_id must be a single directory name.")
    out = (Path(output_root) if output_root is not None else ROOT/config["output_root"])/identifier
    out.mkdir(parents=True, exist_ok=False)
    (out/"config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    _json(out/"input_manifest.json", {name: {"run": config[key], "metadata_sha256": _sha(path/"metadata.json"),
          "verified_outputs": meta["output_sha256"]} for name, key, path, meta in
          [("data", "source_run", source, source_meta), ("baselines", "baseline_run", baseline, baseline_meta)]})
    rows, all_events, audit = [], [], []
    for seed in config["seeds"]:
        src, old, dest = source/f"seed_{seed}", baseline/f"seed_{seed}", out/f"seed_{seed}"
        dest.mkdir()
        references = json.loads((src/"references.json").read_text())
        with np.load(src/"base_data.npz", allow_pickle=False) as data:
            cal = data["splits"] == "train_cal"
            values = data["clean"][cal]
            records = np.repeat(data["regimes"][cal, None], values.shape[1], axis=1)
            cal_ids = data["sequence_ids"][cal].tolist()
        models = {}
        for method in config["methods"]:
            model = calibrate_sma(values, records, references[method], window_size=config["window_size"],
                                  quantile=config["threshold_quantile"])
            models[method] = model
            scores = score_sma(values, records, model)
            for regime in model["thresholds"]:
                for j in range(values.shape[-1]):
                    valid = (records == regime) & scores["available"][:, :, j]
                    audit.append({"seed": seed, "method": method, "regime": regime, "channel": j,
                                  "threshold": model["thresholds"][regime][j],
                                  "n_calibration_cells": int(valid.sum()),
                                  "calibration_fpr": float(scores["alarm"][:, :, j][valid].mean())})
        _json(dest/"models.json", {"calibration_ids": cal_ids, "models": models})
        for scenario in config["scenarios"]:
            with np.load(src/scenario/"validation_data.npz", allow_pickle=False) as loaded:
                data = {k: loaded[k] for k in loaded.files}
            events = json.loads((src/scenario/"events.json").read_text())
            with np.load(old/f"{scenario}_scores.npz", allow_pickle=False) as loaded:
                baseline_scores = {k: loaded[k] for k in loaded.files}
            new_scores = {}
            for view in ("clean", "observed"):
                all_scores = {}
                for method in config["methods"]:
                    scored = score_sma(data[view], data["recorded_regimes"], models[method])
                    all_scores[method, "sma"] = scored
                    for field, value in scored.items():
                        new_scores[f"{method}__sma__{view}__{field}"] = value
                    for temporal, suffix in BASELINES.items():
                        all_scores[method, temporal] = {field: baseline_scores[f"{method}__{suffix}__{view}__{field}"]
                              for field in ("deviation_ratio", "reliability", "available", "alarm")}
                common = np.logical_and.reduce([s["available"].all(axis=-1) for s in all_scores.values()])
                actual = data["mask"]
                truth = actual if view == "observed" else np.zeros_like(actual)
                for (method, temporal), scored in all_scores.items():
                    event_rows, tail = coverage_events(scored, actual, events, post_steps=config["post_event_steps"])
                    for event in event_rows:
                        all_events.append({"seed": seed, "scenario": scenario, "method": method,
                                           "temporal": temporal, "view": view, **event})
                    for regime in ["all", *source_config["regimes"]]:
                        state = np.ones(truth.shape[:2], bool) if regime == "all" else data["true_regimes"] == regime
                        for window, selected in (("all", state), ("transition", state & data["transition"]),
                                                  ("stable", state & ~data["transition"])):
                            if not selected.any():
                                continue
                            for support in ("own", "common"):
                                result = coverage_metrics(truth, scored, selected, common, tail, support=support)
                                rows.append({"seed": seed, "scenario": scenario, "method": method, "temporal": temporal,
                                             "view": view, "regime": regime, "window": window, "support": support, **result})
            np.savez_compressed(dest/f"{scenario}_sma_scores.npz", **new_scores)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out/"metrics_by_seed.csv", index=False)
    summary = aggregate(metrics, GROUPS, METRICS)
    summary.to_csv(out/"summary.csv", index=False)
    pd.DataFrame(audit).to_csv(out/"calibration_audit.csv", index=False)
    events = pd.DataFrame(all_events)
    events.to_csv(out/"events.csv", index=False)
    event_keys = ["seed", "scenario", "method", "temporal", "view"]
    event_seed = events.groupby(event_keys)[EVENTS].mean().reset_index()
    event_seed = event_seed.merge(events.groupby(event_keys).size().rename("event_channel_count").reset_index(),
                                  on=event_keys, validate="one_to_one")
    event_seed.to_csv(out/"events_by_seed.csv", index=False)
    aggregate(event_seed, event_keys[1:], EVENTS).to_csv(out/"event_summary.csv", index=False)
    for frame, keys, fields, prefix in [
        (metrics, ["seed", *[k for k in GROUPS if k != "temporal"]], METRICS, "paired"),
        (event_seed, ["seed", "scenario", "method", "view"], EVENTS, "event_paired")]:
        difference = paired_differences(frame, keys, fields)
        difference.to_csv(out/f"{prefix}_by_seed.csv", index=False)
        aggregate(difference, [*keys[1:], "comparison"], fields).to_csv(out/f"{prefix}_summary.csv", index=False)
    plot_comparison(summary, out)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "src", "tests", "configs"],
                                    cwd=ROOT, text=True).strip()
    _json(out/"metadata.json", {"run_id": identifier, "data_source": "synthetic", "evaluation_split": "val",
        "test_evaluated": False, "quality_evaluated": False, "code_commit": commit,
        "implementation_worktree_dirty": bool(dirty), "seeds": config["seeds"], "config_sha256": config_hash(config),
        "baseline_scores": "read_without_modification_from_audited_e1d",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {str(p.relative_to(ROOT)): _sha(p) for p in sorted((ROOT/"src").glob("*.py"))},
        "environment": {"python": platform.python_version(), **{p: importlib.metadata.version(p)
                        for p in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "PyYAML")}},
        "output_sha256": {str(p.relative_to(out)): _sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
        "limitations": ["synthetic bias only", "full-window startup unavailability",
                        "nominal not realized matched FPR", "classical SMA not a novelty claim",
                        "validation-guided method development; final test reserved", "quality task not defined"]})
    return out


def main() -> None:
    """Run the frozen finite-memory protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(run_finite_memory(args.config, output_root=args.output_root, run_id=args.run_id))


if __name__ == "__main__":
    main()
