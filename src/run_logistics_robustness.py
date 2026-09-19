"""Frozen five-seed paired robustness evaluation, restricted to validation."""

from __future__ import annotations

import argparse
import importlib.metadata
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
from .logistics_simulator import generate_logistics
from .logistics_robustness import make_stress_scenario, training_regime_scales, validate_stress_config
from .run_logistics_e1b import ROOT, _json, _sha, detection_metrics
from .sensor_reliability import fit_sensor_reference, score_sensors

METRICS = ["f1", "recall", "precision", "average_precision", "clean_channel_fpr",
           "bystander_fpr", "top1_localization", "top1_random_baseline",
           "all_faults_row_recall", "exact_fault_set_rate", "score_coverage"]
GROUPS = ["scenario", "method", "view", "regime", "window"]


def stress_metrics(truth: np.ndarray, result: dict[str, np.ndarray]) -> dict[str, Any]:
    """Add multilabel row metrics and the k/p random Top-1 reference rate."""

    complete = result["available"].all(axis=1)
    metrics = detection_metrics(truth, result, complete)
    faults = complete & truth.any(axis=1)
    if faults.any():
        actual, alarm = truth[faults], result["alarm"][faults]
        metrics.update(
            top1_random_baseline=float(actual.mean(axis=1).mean()),
            all_faults_row_recall=float((alarm | ~actual).all(axis=1).mean()),
            exact_fault_set_rate=float((actual == alarm).all(axis=1).mean()),
        )
    else:
        metrics.update(dict.fromkeys(("top1_random_baseline", "all_faults_row_recall",
                                      "exact_fault_set_rate"), float("nan")))
    return metrics


def _figure(summary: pd.DataFrame, destination: Path) -> None:
    selected = summary[(summary.regime == "all") & (summary.window == "all")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    panels = [(["steady_m05", "steady_m1", "steady_m2_r10_k1", "steady_m4"],
               [.5, 1, 2, 4], "observed", "f1", "Bias magnitude / within-regime std", "Sensor detection F1"),
              (["switch_d0", "switch_d4", "switch_d12"],
               [0, 4, 12], "clean", "clean_channel_fpr", "Record delay / sampling steps", "Clean-channel false-positive rate")]
    for ax, (names, x, view, metric, xlabel, ylabel) in zip(axes, panels):
        for method in ("pooled_marginal", "regime_marginal", "regime_conditional"):
            rows = selected[(selected.method == method) & (selected.view == view)].set_index("scenario").loc[names]
            ax.errorbar(x, rows[f"{metric}_mean"], yerr=rows[f"{metric}_std"],
                        marker="o", capsize=3, label=method)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.suptitle("Synthetic validation: five seeds; error bars = sample standard deviation")
    fig.savefig(destination / "robustness.png", dpi=180)
    fig.savefig(destination / "robustness.pdf")
    plt.close(fig)


def run_robustness(config_path: str | Path, *, output_root: str | Path | None = None,
                   run_id: str | None = None) -> Path:
    """Save paired data, models, scores, stratified metrics and provenance."""

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    validate_stress_config(config)
    identifier = run_id or datetime.now(timezone.utc).strftime("stress_%Y%m%dT%H%M%S%fZ")
    if Path(identifier).name != identifier or identifier in {".", ".."}:
        raise ValueError("run_id must be a single directory name.")
    root = Path(output_root) if output_root is not None else ROOT / config["output_root"]
    out = root / identifier
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    rows = []
    for seed in config["seeds"]:
        base = generate_logistics(config, seed)
        directory = out / f"seed_{seed}"
        directory.mkdir()
        np.savez_compressed(directory / "base_data.npz", clean=base.clean,
                            sequence_ids=base.sequence_ids, regimes=base.regimes, splits=base.splits,
                            features=config["features"], units=config["units"])
        p = base.clean.shape[-1]
        fit, cal = (base.splits == k for k in ("train_fit", "train_cal"))
        models = {method: fit_sensor_reference(
            base.clean[fit].reshape(-1, p), np.repeat(base.regimes[fit], config["sequence_length"]),
            base.clean[cal].reshape(-1, p), np.repeat(base.regimes[cal], config["sequence_length"]),
            method=method, single_regime=config["single_reference_regime"],
            quantile=config["threshold_quantile"],
        ) for method in config["methods"]}
        _json(directory / "references.json", models)
        _json(directory / "scales.json", {k: v.tolist() for k, v in training_regime_scales(base).items()})
        _json(directory / "membership.json", {k: base.sequence_ids[base.splits == k].tolist()
                                              for k in ("train_fit", "train_cal", "val", "test")})
        for scenario in config["stress"]["scenarios"]:
            data = make_stress_scenario(base, config, scenario, seed)
            dest = directory / scenario["name"]
            dest.mkdir()
            np.savez_compressed(dest / "validation_data.npz", clean=data.clean, observed=data.observed,
                                mask=data.mask, true_regimes=data.true_regimes,
                                recorded_regimes=data.recorded_regimes, transition=data.transition,
                                sequence_ids=data.sequence_ids)
            _json(dest / "events.json", data.events)
            true_state, records = data.true_regimes.ravel(), data.recorded_regimes.ravel()
            transition = data.transition.ravel()
            scores = {}
            for method, model in models.items():
                for view, values, truth in (
                    ("clean", data.clean, np.zeros_like(data.mask)),
                    ("observed", data.observed, data.mask),
                ):
                    result = score_sensors(values.reshape(-1, p), records, model)
                    for key, value in result.items():
                        scores[f"{method}__{view}__{key}"] = value
                    for regime in ["all", *config["regimes"]]:
                        in_state = np.ones(len(records), bool) if regime == "all" else true_state == regime
                        for window, in_window in (("all", np.ones(len(records), bool)),
                                                  ("transition", transition), ("stable", ~transition)):
                            selected = in_state & in_window
                            if not selected.any():
                                continue
                            metric = stress_metrics(truth.reshape(-1, p)[selected],
                                                    {k: v[selected] for k, v in result.items()})
                            rows.append({"seed": seed, "scenario": scenario["name"], "method": method,
                                         "view": view, "regime": regime, "window": window,
                                         "n_rows": int(selected.sum()),
                                         "state_mismatch_rate": float((true_state[selected] != records[selected]).mean()),
                                         **metric})
            np.savez_compressed(dest / "scores.npz", **scores)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics_by_seed.csv", index=False)
    summary = metrics.groupby(GROUPS)[METRICS].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    paired = []
    full = metrics[(metrics.regime == "all") & (metrics.window == "all")]
    keys = ["seed", "method", "view"]
    for scenario in config["stress"]["scenarios"]:
        if scenario["compare_to"] is None:
            continue
        a = full[full.scenario == scenario["name"]].set_index(keys)[METRICS]
        b = full[full.scenario == scenario["compare_to"]].set_index(keys)[METRICS]
        delta = (a - b).reset_index()
        delta["scenario"], delta["compare_to"] = scenario["name"], scenario["compare_to"]
        paired.append(delta)
    differences = pd.concat(paired, ignore_index=True)
    differences.to_csv(out / "paired_differences.csv", index=False)
    delta_summary = differences.groupby(["scenario", "compare_to", "method", "view"])[METRICS].agg(["mean", "std", "count"])
    delta_summary.columns = [f"{metric}_{stat}" for metric, stat in delta_summary.columns]
    delta_summary.reset_index().to_csv(out / "paired_summary.csv", index=False)
    _figure(summary, out)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "configs", "tests"],
                           cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    _json(out / "metadata.json", {
        "run_id": identifier, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "synthetic", "evaluation_split": "val", "test_evaluated": False,
        "quality_target": "not_defined_not_evaluated", "seeds": config["seeds"],
        "config_sha256": config_hash(config), "code_commit": commit,
        "implementation_worktree_dirty": bool(dirty),
        "source_sha256": {str(path.relative_to(ROOT)): _sha(path) for path in sorted((ROOT / "src").glob("*.py"))},
        "environment": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                        for name in ("numpy", "pandas", "scikit-learn", "scipy", "PyYAML", "matplotlib")}},
        "output_sha256": {str(path.relative_to(out)): _sha(path) for path in sorted(out.rglob("*")) if path.is_file()},
        "limitations": ["illustrative abrupt switching", "ideal clean training", "segment bias only",
                        "known or deliberately delayed records; not estimated regimes",
                        "independent channel signs; not all coordinated faults", "no real quality or industrial validation"],
    })
    return out


def main() -> None:
    """Run the frozen E1c robustness protocol from YAML."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(run_robustness(args.config, output_root=args.output_root, run_id=args.run_id))


if __name__ == "__main__":
    main()
