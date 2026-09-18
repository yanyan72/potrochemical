"""Reproducible multi-seed storage/transport sensor reliability comparisons."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

from .config import config_hash
from .logistics_simulator import generate_logistics, validate_logistics_config
from .sensor_reliability import fit_sensor_reference, score_sensors

ROOT = Path(__file__).resolve().parent.parent


def detection_metrics(truth: np.ndarray, scored: dict[str, np.ndarray],
                      complete_rows: np.ndarray) -> dict[str, float | int]:
    """Evaluate numeric corruption on common complete-row support, not missingness."""

    support = np.broadcast_to(complete_rows[:, None], truth.shape)
    support = support & scored["available"]
    actual, alarm = truth[support], scored["alarm"][support]
    tp = int(np.sum(actual & alarm))
    fp = int(np.sum(~actual & alarm))
    fn = int(np.sum(actual & ~alarm))
    tn = int(np.sum(~actual & ~alarm))
    def divide(a: int, b: int) -> float:
        return float(a / b) if b else float("nan")
    localized = complete_rows & truth.any(axis=1) & scored["available"].all(axis=1)
    if localized.any():
        top = np.argmax(scored["deviation_ratio"][localized], axis=1)
        hits = int(truth[localized][np.arange(localized.sum()), top].sum())
    else:
        hits = 0
    bystander = support & ~truth & truth.any(axis=1)[:, None]
    clean_rows = complete_rows & ~truth.any(axis=1) & scored["available"].all(axis=1)
    return {
        "n_cells": int(support.sum()), "positive_cells": int(actual.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": divide(tp, tp + fp), "recall": divide(tp, tp + fn),
        "f1": divide(2 * tp, 2 * tp + fp + fn),
        "clean_channel_fpr": divide(fp, fp + tn),
        "average_precision": (float(average_precision_score(actual, scored["deviation_ratio"][support]))
                              if actual.any() and (~actual).any() else float("nan")),
        "top1_localization": divide(hits, int(localized.sum())),
        "localization_rows": int(localized.sum()),
        "bystander_fpr": divide(int(scored["alarm"][bystander].sum()), int(bystander.sum())),
        "bystander_cells": int(bystander.sum()),
        "clean_row_alarm_rate": divide(int(scored["alarm"][clean_rows].any(axis=1).sum()), int(clean_rows.sum())),
        "score_coverage": float(scored["available"].mean()),
        "common_complete_row_fraction": float(complete_rows.mean()),
    }


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_experiment(config_path: str | Path, *, output_root: str | Path | None = None,
                   run_id: str | None = None) -> Path:
    """Run a frozen protocol; save test data but never evaluate test performance."""

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    validate_logistics_config(config)
    identifier = run_id or datetime.now(timezone.utc).strftime("logistics_%Y%m%dT%H%M%S%fZ")
    if Path(identifier).name != identifier or identifier in {".", ".."}:
        raise ValueError("run_id must be a single directory name.")
    root = Path(output_root) if output_root is not None else ROOT / config["output_root"]
    destination = root / identifier
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    rows, type_rows, missing_rows = [], [], []
    for seed in config["seeds"]:
        data = generate_logistics(config, seed)
        directory = destination / f"seed_{seed}"
        directory.mkdir()
        np.savez_compressed(directory / "data.npz", clean=data.clean, observed=data.observed,
                            mask=data.mask, corruption_type=data.corruption_type,
                            sequence_ids=data.sequence_ids, splits=data.splits, regimes=data.regimes,
                            features=np.asarray(config["features"]), units=np.asarray(config["units"]))
        _json(directory / "events.json", data.events)
        length, p = data.clean.shape[1:]
        fit, cal, val = (data.splits == split for split in ("train_fit", "train_cal", "val"))
        val_regimes = np.repeat(data.regimes[val], length)
        observed = data.observed[val].reshape(-1, p)
        clean = data.clean[val].reshape(-1, p)
        truth = data.mask[val].reshape(-1, p)
        types = data.corruption_type[val].reshape(-1)
        complete = np.isfinite(observed).all(axis=1)
        model_parameters, scores = {}, {}
        for method in config["methods"]:
            model = fit_sensor_reference(
                data.clean[fit].reshape(-1, p), np.repeat(data.regimes[fit], length),
                data.clean[cal].reshape(-1, p), np.repeat(data.regimes[cal], length),
                method=method, single_regime=config["single_reference_regime"],
                quantile=config["threshold_quantile"],
            )
            model_parameters[method] = model
            scored = score_sensors(observed, val_regimes, model)
            for key, value in scored.items():
                scores[f"{method}__{key}"] = value
            for scenario, values, actual, supported in (
                ("observed", observed, truth, complete),
                ("clean", clean, np.zeros_like(truth), np.ones(len(clean), bool)),
            ):
                result = scored if scenario == "observed" else score_sensors(values, val_regimes, model)
                for regime in ["all", *config["regimes"]]:
                    selected = np.ones(len(values), bool) if regime == "all" else val_regimes == regime
                    metric = detection_metrics(actual[selected], {k: v[selected] for k, v in result.items()}, supported[selected])
                    rows.append({"seed": seed, "method": method, "scenario": scenario,
                                 "regime": regime, **metric})
            for kind in config["corruption"]["types"]:
                if kind == "missing":
                    continue
                selected = types == kind
                if selected.any():
                    type_rows.append({"seed": seed, "method": method, "corruption_type": kind,
                                      **detection_metrics(truth[selected], {k: v[selected] for k, v in scored.items()}, complete[selected])})
        np.savez_compressed(directory / "validation_scores.npz", **scores)
        _json(directory / "references.json", model_parameters)
        _json(directory / "reference_sequences.json", {
            "fit": data.sequence_ids[fit].tolist(), "calibration": data.sequence_ids[cal].tolist(),
            "validation": data.sequence_ids[val].tolist(),
        })
        missing_rows.append({"seed": seed, "validation_cells": int(observed.size),
                             "visible_missing_cells": int(np.isnan(observed).sum()),
                             "numeric_corrupted_cells": int((truth & np.isfinite(observed)).sum()),
                             "complete_rows": int(complete.sum()), "validation_rows": len(observed)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(destination / "metrics_by_seed.csv", index=False)
    pd.DataFrame(type_rows).to_csv(destination / "metrics_by_type.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(destination / "missingness.csv", index=False)
    numeric = [c for c in metrics if c not in {"seed", "method", "scenario", "regime"}]
    summary = metrics.groupby(["method", "scenario", "regime"])[numeric].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(destination / "summary.csv", index=False)
    code = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal", "--", "src", "configs", "tests"],
                           cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    sources = sorted((ROOT / "src").glob("*.py"))
    metadata = {
        "run_id": identifier, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "synthetic", "scenario": "storage_transport_environment_monitoring",
        "quality_target": "not_defined_not_evaluated", "regime_information": "known_simulator_labels",
        "reference_source": "ideal_reliable_clean_training", "fit_split": "train_fit",
        "calibration_split": "train_cal", "evaluation_split": "val", "test_evaluated": False,
        "time_unit": config["time_unit"], "seeds": config["seeds"], "config_sha256": config_hash(config),
        "code_commit": code, "implementation_worktree_dirty": bool(dirty),
        "source_sha256": {str(path.relative_to(ROOT)): _sha(path) for path in sources},
        "environment": {"python": platform.python_version(), **{package: importlib.metadata.version(package)
                        for package in ("numpy", "pandas", "scikit-learn", "scipy", "PyYAML")}},
        "output_sha256": {str(path.relative_to(destination)): _sha(path)
                          for path in sorted(destination.rglob("*")) if path.is_file()},
        "limitations": ["illustrative parameter choices", "fixed regime per sequence", "one corrupted channel per affected row",
                        "no real faults", "no forecasting result", "no unseen-regime or transition performance claim"],
    }
    _json(destination / "metadata.json", metadata)
    return destination


def main() -> None:
    """Run the storage/transport E1b comparison from a YAML configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(run_experiment(args.config, output_root=args.output_root, run_id=args.run_id))


if __name__ == "__main__":
    main()
