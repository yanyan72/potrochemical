"""Paired record-change reset ablation on audited E1f development validation."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .config import config_hash
from .run_logistics_e1b import ROOT, _json, _sha
from .run_finite_memory_e1e import audited_source, METRICS
from .run_controlled_e1f import GROUPS, EVENTS, differences
from .run_temporal_e1d import aggregate
from .temporal_reliability import score_temporal
from .finite_memory import coverage_metrics, coverage_events

FIELDS = ("deviation_ratio", "reliability", "available", "alarm")


def record_windows(records: np.ndarray, steps: int) -> np.ndarray:
    """Mark steps after each visible record change, excluding sequence starts."""
    if records.ndim != 2 or type(steps) is not int or steps < 1:
        raise ValueError("Require record matrix and positive window length.")
    result = np.zeros(records.shape, bool)
    for i, t in np.argwhere(records[:, 1:] != records[:, :-1]):
        result[i, t+1:t+1+steps] = True
    return result


def identical(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    """Require exact score replay, including unavailability and NaNs."""
    for field in FIELDS:
        if not np.array_equal(a[field], b[field], equal_nan=True):
            raise ValueError(f"Score replay differs: {field}")


def validate_config(config: dict[str, Any], source: dict[str, Any]) -> None:
    """Reject retuning, unpaired methods and unsupported source subsets."""
    if config["data_source"] != "synthetic":
        raise ValueError("Synthetic-only protocol.")
    for name, allowed in (("seeds", source["seeds"]), ("methods", source["methods"]),
                          ("scenarios", [s["name"] for s in source["scenarios"]])):
        values = config[name]
        if not values or len(values) != len(set(values)) or not set(values) <= set(allowed):
            raise ValueError("Require unique source subsets.")
    if set(config["methods"]) != {"regime_marginal", "regime_conditional"}:
        raise ValueError("Require both residual methods.")
    if (config["alpha"] != .2 or config["threshold_quantile"] != source["threshold_quantile"]
            or config["post_event_steps"] != source["post_event_steps"]):
        raise ValueError("Do not change source parameters or budget.")
    if type(config["record_window_steps"]) is not int or config["record_window_steps"] < 1:
        raise ValueError("Positive record window required.")


def plot_results(summary: pd.DataFrame, out: Path) -> None:
    """Plot paired phase detection and delayed-record recovery cost with seed SD."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    names = ["switch_before6", "switch_at", "switch_after6"]
    for method in ("regime_marginal", "regime_conditional"):
        for temporal in ("ewma", "carry"):
            part = summary[(summary.method == method) & (summary.temporal == temporal) & (summary.regime == "all")]
            phase = part[(part.view == "observed") & (part.window == "all")].set_index("scenario")
            label = f"{method.removeprefix('regime_')}/{temporal}"
            if set(names) <= set(phase.index):
                phase = phase.loc[names]
                axes[0].errorbar([-6, 0, 6], phase.all_timeline_recall_mean,
                    yerr=phase.all_timeline_recall_std, marker="o", capsize=3, label=label)
            recovery = part[(part.view == "clean") & (part.scenario == "switch_delay12")].set_index("window")
            windows = ["transition", "record_transition", "all"]
            if set(windows) <= set(recovery.index):
                recovery = recovery.loc[windows]
                axes[1].errorbar(range(3), recovery.clean_channel_fpr_mean,
                    yerr=recovery.clean_channel_fpr_std, marker="o", capsize=3, label=label)
    axes[0].set(xlabel="Event start relative to true switch / steps", ylabel="Cell recall")
    axes[1].set(xlabel="Delayed-record clean evaluation window", ylabel="Clean-cell FPR",
                xticks=range(3), xticklabels=["True switch", "Record update", "All"])
    for ax in axes:
        ax.grid(alpha=.25)
        if ax.lines:
            ax.legend(fontsize=7)
    fig.suptitle("Synthetic paired reset ablation; fixed thresholds; error bars: seed SD")
    fig.savefig(out/"transition_reset.png", dpi=180)
    fig.savefig(out/"transition_reset.pdf")
    plt.close(fig)


def run_transition(config_path: str | Path, *, output_root: str | Path | None = None,
                   run_id: str | None = None) -> Path:
    """Evaluate the reset toggle without changing sources, labels or thresholds."""
    config = yaml.safe_load(Path(config_path).read_text())
    source, training = (ROOT/config[k] for k in ("source_run", "training_run"))
    source_meta, train_meta = audited_source(source), audited_source(training)
    source_config = yaml.safe_load((source/"config.yaml").read_text())
    train_config = yaml.safe_load((training/"config.yaml").read_text())
    validate_config(config, source_config)
    manifest = json.loads((source/"input_manifest.json").read_text())["training_source"]
    if ((ROOT/manifest["path"]).resolve() != training.resolve()
            or manifest["metadata_sha256"] != _sha(training/"metadata.json")):
        raise ValueError("Source training provenance mismatch.")
    identifier = run_id or datetime.now(timezone.utc).strftime("transition_%Y%m%dT%H%M%S%fZ")
    if Path(identifier).name != identifier or identifier in {".", ".."}:
        raise ValueError("Unsafe run ID.")
    out = (Path(output_root) if output_root is not None else ROOT/config["output_root"])/identifier
    out.mkdir(parents=True, exist_ok=False)
    (out/"config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    _json(out/"input_manifest.json", {key: {"path": config[key], "metadata_sha256": _sha(p/"metadata.json"),
         "verified_outputs": m["output_sha256"]} for key, p, m in
         (("source_run", source, source_meta), ("training_run", training, train_meta))})
    rows, event_rows, audit = [], [], []
    replay_count, steady_count = 0, 0
    for seed in config["seeds"]:
        src, dest = source/f"seed_{seed}", out/f"seed_{seed}"
        dest.mkdir()
        saved = json.loads((src/"models.json").read_text())
        with np.load(training/f"seed_{seed}"/"base_data.npz", allow_pickle=False) as base:
            selected = base["splits"] == "train_cal"
            cal = base["clean"][selected]
            records = np.repeat(base["regimes"][selected, None], cal.shape[1], axis=1)
            ids = base["sequence_ids"][selected].tolist()
        if ids != saved["calibration_ids"] or set(ids) & set(saved["validation_ids"]):
            raise ValueError("Calibration/validation membership mismatch.")
        carry = {}
        for method in config["methods"]:
            old = saved["models"][f"{method}__ewma"]
            if (old["alpha"] != config["alpha"] or old["quantile"] != config["threshold_quantile"]
                    or old["reset_on_record_change"] is not True):
                raise ValueError("Frozen EWMA model mismatch.")
            new = deepcopy(old)
            new["reset_on_record_change"] = False
            carry[method] = new
            calibrated = score_temporal(cal, records, new)
            identical(calibrated, score_temporal(cal, records, old))
            for regime, thresholds in new["thresholds"].items():
                for j, threshold in enumerate(thresholds):
                    mask = records == regime
                    audit.append({"seed": seed, "method": method, "regime": regime, "channel": j,
                        "threshold": threshold, "calibration_fpr": float(calibrated["alarm"][:, :, j][mask].mean()),
                        "n_calibration_cells": int(mask.sum()), "reset_carry_exact": True})
        _json(dest/"models.json", {"models": carry, "calibration_ids": ids,
                                   "validation_ids": saved["validation_ids"], "thresholds_refit": False})
        for scenario in config["scenarios"]:
            with np.load(src/scenario/"validation_data.npz", allow_pickle=False) as loaded:
                data = {k: loaded[k] for k in loaded.files}
            if data["sequence_ids"].tolist() != saved["validation_ids"]:
                raise ValueError("Validation identity mismatch.")
            events = json.loads((src/scenario/"events.json").read_text())
            with np.load(src/scenario/"scores.npz", allow_pickle=False) as loaded:
                baseline = {k: loaded[k] for k in loaded.files}
            record_window = record_windows(data["recorded_regimes"], config["record_window_steps"])
            new_scores = {}
            for view in ("clean", "observed"):
                scored_all = {}
                for method in config["methods"]:
                    for temporal in ("point", "ewma", "cusum"):
                        scored_all[method, temporal] = {k: baseline[f"{method}__{temporal}__{view}__{k}"] for k in FIELDS}
                    identical(scored_all[method, "ewma"], score_temporal(data[view], data["recorded_regimes"],
                              saved["models"][f"{method}__ewma"]))
                    replay_count += 1
                    scored = score_temporal(data[view], data["recorded_regimes"], carry[method])
                    if not record_window.any():
                        identical(scored, scored_all[method, "ewma"])
                        steady_count += 1
                    scored_all[method, "carry"] = scored
                    for k, values in scored.items():
                        new_scores[f"{method}__carry__{view}__{k}"] = values
                common = np.logical_and.reduce([s["available"].all(axis=-1) for s in scored_all.values()])
                truth = data["mask"] if view == "observed" else np.zeros_like(data["mask"])
                for (method, temporal), scored in scored_all.items():
                    per_event, tail = coverage_events(scored, data["mask"], events, post_steps=config["post_event_steps"])
                    for event in per_event:
                        event_rows.append({"seed": seed, "scenario": scenario, "method": method,
                            "temporal": temporal, "view": view, **event,
                            "delay_fraction_capped": event["fresh_delay_capped"]/event["duration"]})
                    for regime in ["all", *train_config["regimes"]]:
                        state = np.ones(truth.shape[:2], bool) if regime == "all" else data["true_regimes"] == regime
                        for window, mask in (("all", np.ones_like(state)), ("transition", data["transition"]),
                            ("stable", ~data["transition"]), ("record_transition", record_window),
                            ("record_stable", ~record_window)):
                            selected = state & mask
                            if selected.any():
                                rows.append({"seed": seed, "scenario": scenario, "method": method, "temporal": temporal,
                                    "view": view, "regime": regime, "window": window,
                                    **coverage_metrics(truth, scored, selected, common, tail, support="common")})
            np.savez_compressed(dest/f"{scenario}_carry_scores.npz", **new_scores)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out/"metrics_by_seed.csv", index=False)
    summary = aggregate(metrics, GROUPS, METRICS)
    summary.to_csv(out/"summary.csv", index=False)
    pd.DataFrame(audit).to_csv(out/"calibration_audit.csv", index=False)
    events = pd.DataFrame(event_rows)
    events.to_csv(out/"events.csv", index=False)
    event_keys = ["seed", "scenario", "method", "temporal", "view"]
    event_seed = events.groupby(event_keys)[EVENTS].mean().reset_index()
    event_seed = event_seed.merge(events.groupby(event_keys).size().rename("event_channel_count").reset_index(),
                                 on=event_keys, validate="one_to_one")
    event_seed.to_csv(out/"events_by_seed.csv", index=False)
    aggregate(event_seed, event_keys[1:], EVENTS).to_csv(out/"event_summary.csv", index=False)
    for frame, keys, fields, prefix in (
        (metrics, ["seed", *[k for k in GROUPS if k != "temporal"]], METRICS, "paired"),
        (event_seed, ["seed", "scenario", "method", "view"], EVENTS, "event_paired")):
        delta = differences(frame, keys, fields, column="temporal", a_value="carry", b_value="ewma")
        delta.to_csv(out/f"{prefix}_by_seed.csv", index=False)
        aggregate(delta, [*keys[1:], "comparison"], fields).to_csv(out/f"{prefix}_summary.csv", index=False)
    plot_results(summary, out)
    _json(out/"metadata.json", {"run_id": identifier, "data_source": "synthetic",
        "evaluation_split": "reused_e1f_development_validation", "test_evaluated": False, "quality_evaluated": False,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "implementation_worktree_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--", "src", "tests", "configs"], cwd=ROOT, text=True).strip()),
        "seeds": config["seeds"], "config_sha256": config_hash(config), "thresholds_refit": False,
        "ewma_replay_count": replay_count, "steady_invariance_count": steady_count,
        "calibration_reset_carry_exact": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {str(p.relative_to(ROOT)): _sha(p) for p in sorted((ROOT/"src").glob("*.py"))},
        "environment": {"python": platform.python_version(), **{p: importlib.metadata.version(p) for p in
                         ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "PyYAML")}},
        "output_sha256": {str(p.relative_to(out)): _sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
        "limitations": ["reused synthetic development validation, not blind test", "clean oracle training",
            "cross-regime residual comparability is an assumption", "nominal not realized matched FPR",
            "carry can retain wrong-record history", "no quality or real-data evidence"]})
    return out


def main() -> None:
    """Run the frozen record-reset ablation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(run_transition(args.config, output_root=args.output_root, run_id=args.run_id))


if __name__ == "__main__":
    main()
