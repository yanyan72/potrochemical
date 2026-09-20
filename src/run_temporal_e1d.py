"""Replay audited E1c data with a prespecified signed EWMA ablation."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .config import config_hash
from .run_logistics_e1b import ROOT, _json, _sha
from .run_logistics_robustness import METRICS, stress_metrics
from .temporal_reliability import calibrate_temporal, event_evaluation, score_temporal

GROUPS = ["scenario", "method", "alpha", "view", "regime", "window"]
EVENT_METRICS = ["any_active_hit", "fresh_onset_hit", "pre_active", "fresh_delay_detected", "fresh_delay_capped"]
POINT_METRICS = METRICS + ["post_event_fpr", "false_onsets_per_1000_clean_cells"]


def aggregate(frame: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    """Aggregate independent seeds; count reports undefined/missing metric support."""
    result = frame.groupby(groups)[metrics].agg(["mean", "std", "count"])
    result.columns = [f"{name}_{stat}" for name, stat in result.columns]
    return result.reset_index()


def validate_config(config: dict, source_config: dict) -> None:
    """Reject undeclared methods/seeds/scenarios and alpha settings before execution."""
    if config.get("data_source") != "synthetic":
        raise ValueError("Synthetic-only protocol.")
    for key in ("seeds", "methods", "alphas", "scenarios"):
        if not config[key] or len(set(config[key])) != len(config[key]):
            raise ValueError("Require unique nonempty configuration lists.")
    if set(config["seeds"]) - set(source_config["seeds"]):
        raise ValueError("Unknown source seeds.")
    if set(config["methods"]) - {"regime_marginal", "regime_conditional"}:
        raise ValueError("Unsupported reference methods.")
    if set(config["scenarios"]) - {s["name"] for s in source_config["stress"]["scenarios"]}:
        raise ValueError("Unknown source scenarios.")
    if 1. not in config["alphas"] or any(not 0 < a <= 1 for a in config["alphas"]):
        raise ValueError("Need alpha=1 ablation and valid alpha values.")
    if not 0 < config["threshold_quantile"] < 1 or config["threshold_quantile"] != source_config["threshold_quantile"]:
        raise ValueError("Keep the source nominal quantile budget.")
    if type(config["post_event_steps"]) is not int or config["post_event_steps"] < 1:
        raise ValueError("Invalid post-event window.")


def _plot(summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    selected = summary[(summary.regime == "all") & (summary.window == "all")]
    for ax, view, metric, label in zip(axes, ["observed", "clean"], ["recall", "clean_channel_fpr"],
                                      ["Numeric corruption recall", "Clean-channel false-positive rate"]):
        for (method, alpha), group in selected[selected.view == view].groupby(["method", "alpha"]):
            candidates = [("steady_m05", .5), ("steady_m1", 1), ("steady_m2_r10_k1", 2), ("steady_m4", 4)]
            present = [(name, value) for name, value in candidates if name in set(group.scenario)]
            if not present:
                continue
            names, x = zip(*present)
            points = group.set_index("scenario").loc[list(names)]
            ax.errorbar(x, points[f"{metric}_mean"], yerr=points[f"{metric}_std"], capsize=3,
                        marker="o", label=f"{method}, alpha={alpha:g}")
        ax.set(xlabel="Bias magnitude / within-regime std", ylabel=label)
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.suptitle("Synthetic validation; same nominal q99 budget; error bars = seed SD")
    fig.savefig(out / "temporal_comparison.png", dpi=180)
    fig.savefig(out / "temporal_comparison.pdf")
    plt.close(fig)


def run_temporal(config_path: str | Path, *, output_root: str | Path | None = None,
                 run_id: str | None = None) -> Path:
    """Replay frozen source data, audit alpha=1 equivalence and save complete evidence."""
    config = yaml.safe_load(Path(config_path).read_text())
    source = ROOT / config["source_run"]
    source_metadata = json.loads((source / "metadata.json").read_text())
    source_config = yaml.safe_load((source / "config.yaml").read_text())
    validate_config(config, source_config)
    if source_metadata["data_source"] != "synthetic" or source_metadata["test_evaluated"]:
        raise ValueError("Source must be the reserved-test synthetic protocol.")
    # Audits happen before output creation; no pickle loading or credential access.
    for name, expected in source_metadata["output_sha256"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()) or _sha(path) != expected:
            raise ValueError(f"Source artifact integrity failure: {name}")
    identifier = run_id or datetime.now(timezone.utc).strftime("temporal_%Y%m%dT%H%M%S%fZ")
    if Path(identifier).name != identifier or identifier in {".", ".."}:
        raise ValueError("run_id must be a single directory name.")
    out = (Path(output_root) if output_root is not None else ROOT / config["output_root"]) / identifier
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    _json(out / "input_manifest.json", {"source_run": config["source_run"],
                                       "source_metadata_sha256": _sha(source / "metadata.json"),
                                       "verified_source_outputs": source_metadata["output_sha256"]})
    rows, events_all, calibration_rows = [], [], []
    for seed in config["seeds"]:
        src = source / f"seed_{seed}"
        directory = out / f"seed_{seed}"
        directory.mkdir()
        references = json.loads((src / "references.json").read_text())
        with np.load(src / "base_data.npz", allow_pickle=False) as base:
            cal = base["splits"] == "train_cal"
            cal_values = base["clean"][cal]
            cal_records = np.repeat(base["regimes"][cal, None], cal_values.shape[1], axis=1)
            cal_ids = base["sequence_ids"][cal].tolist()
        models = {}
        for method in config["methods"]:
            for alpha in config["alphas"]:
                key = f"{method}__a{alpha:g}"
                model = calibrate_temporal(cal_values, cal_records, references[method], alpha=alpha,
                                           quantile=config["threshold_quantile"])
                models[key] = model
                calibrated = score_temporal(cal_values, cal_records, model)
                for regime in model["thresholds"]:
                    for j in range(cal_values.shape[-1]):
                        subset = cal_records == regime
                        calibration_rows.append({"seed": seed, "method": method, "alpha": alpha,
                                                 "regime": regime, "channel": j,
                                                 "threshold": model["thresholds"][regime][j],
                                                 "n_calibration_cells": int(subset.sum()),
                                                 "calibration_fpr": float(calibrated["alarm"][subset, j].mean())})
        _json(directory / "models.json", {"calibration_ids": cal_ids, "models": models})
        for scenario in config["scenarios"]:
            with np.load(src / scenario / "validation_data.npz", allow_pickle=False) as file:
                data = {key: file[key] for key in file.files}
            event_list = json.loads((src / scenario / "events.json").read_text())
            actual_mask = data["mask"]
            scores = {}
            for key, model in models.items():
                method, alpha = model["reference"]["method"], model["alpha"]
                for view in ("clean", "observed"):
                    scored = score_temporal(data[view], data["recorded_regimes"], model)
                    if alpha == 1:
                        with np.load(src / scenario / "scores.npz", allow_pickle=False) as old:
                            for name, values in scored.items():
                                expected = old[f"{method}__{view}__{name}"].reshape(values.shape)
                                if values.dtype == bool:
                                    np.testing.assert_array_equal(values, expected)
                                else:
                                    np.testing.assert_allclose(values, expected, rtol=1e-12, atol=1e-14, equal_nan=True)
                    for name, values in scored.items():
                        scores[f"{key}__{view}__{name}"] = values
                    event_rows, tail = event_evaluation(scored["alarm"], actual_mask, event_list,
                                                       post_steps=config["post_event_steps"])
                    for event in event_rows:
                        i, start = event["sequence_index"], event["start"]
                        events_all.append({"seed": seed, "scenario": scenario, "method": method, "alpha": alpha,
                                           "view": view, "regime_at_onset": data["true_regimes"][i, start], **event})
                    truth = actual_mask if view == "observed" else np.zeros_like(actual_mask)
                    previous = np.concatenate((np.zeros_like(truth[:, :1]), scored["alarm"][:, :-1]), axis=1)
                    false_onsets = scored["alarm"] & ~previous & ~truth
                    for regime in ["all", *source_config["regimes"]]:
                        state = np.ones(truth.shape[:2], bool) if regime == "all" else data["true_regimes"] == regime
                        for window, selected in (("all", state), ("transition", state & data["transition"]),
                                                  ("stable", state & ~data["transition"])):
                            if not selected.any():
                                continue
                            result = stress_metrics(truth[selected], {k: v[selected] for k, v in scored.items()})
                            tail_selected = tail & selected[:, :, None] & scored["available"]
                            negatives = (~truth & scored["available"])[selected].sum()
                            result.update(post_event_cells=int(tail_selected.sum()),
                                post_event_fpr=float(scored["alarm"][tail_selected].mean()) if tail_selected.any() else float("nan"),
                                false_onsets_per_1000_clean_cells=float(1000 * false_onsets[selected].sum() / negatives) if negatives else float("nan"))
                            rows.append({"seed": seed, "scenario": scenario, "method": method, "alpha": alpha,
                                         "view": view, "regime": regime, "window": window, **result})
            np.savez_compressed(directory / f"{scenario}_scores.npz", **scores)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics_by_seed.csv", index=False)
    summary = aggregate(metrics, GROUPS, POINT_METRICS)
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(out / "calibration_audit.csv", index=False)
    events = pd.DataFrame(events_all)
    events.to_csv(out / "events.csv", index=False)
    event_groups = ["seed", "scenario", "method", "alpha", "view"]
    event_seed = events.groupby(event_groups)[EVENT_METRICS].mean().reset_index()
    counts = events.groupby(event_groups).size().rename("event_channel_count").reset_index()
    event_seed = event_seed.merge(counts, on=event_groups, validate="one_to_one")
    event_seed.to_csv(out / "events_by_seed.csv", index=False)
    aggregate(event_seed, event_groups[1:], EVENT_METRICS).to_csv(out / "event_summary.csv", index=False)
    paired = []
    indices = ["seed", "scenario", "method", "view", "regime", "window"]
    for alpha in config["alphas"]:
        if alpha == 1:
            continue
        a = metrics[metrics.alpha == alpha].set_index(indices)[POINT_METRICS]
        b = metrics[metrics.alpha == 1].set_index(indices)[POINT_METRICS]
        delta = (a - b).reset_index()
        delta["alpha"] = alpha
        paired.append(delta)
    if paired:
        difference = pd.concat(paired, ignore_index=True)
        difference.to_csv(out / "paired_by_seed.csv", index=False)
        aggregate(difference, GROUPS, POINT_METRICS).to_csv(out / "paired_summary.csv", index=False)
    _plot(summary, out)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "src", "tests", "configs"], cwd=ROOT, text=True).strip()
    _json(out / "metadata.json", {"run_id": identifier, "data_source": "synthetic", "evaluation_split": "val",
        "test_evaluated": False, "code_commit": commit, "implementation_worktree_dirty": bool(dirty),
        "source_run": config["source_run"], "source_metadata_sha256": _sha(source / "metadata.json"),
        "alpha_one_replay_verified": True, "config_sha256": config_hash(config), "seeds": config["seeds"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {str(path.relative_to(ROOT)): _sha(path) for path in sorted((ROOT / "src").glob("*.py"))},
        "environment": {"python": platform.python_version(), **{p: importlib.metadata.version(p)
                         for p in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "PyYAML")}},
        "output_sha256": {str(p.relative_to(out)): _sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
        "limitations": ["synthetic bias only", "nominal rather than realized matched FPR", "alpha fixed not tuned",
                        "classical EWMA not novel by itself", "no quality prediction or industrial evidence"]})
    return out


def main() -> None:
    """Execute E1d using a YAML config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(run_temporal(args.config, output_root=args.output_root, run_id=args.run_id))


if __name__ == "__main__":
    main()
