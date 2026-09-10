"""Command-line pipeline for E1 trust calibration and first detection evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import config_hash
from .evaluate_trust import (
    build_condition_false_positive_table,
    build_corruption_type_recall_table,
    build_detection_metrics_table,
    plot_distance_trust_timeseries,
)
from .trust_score import (
    assign_trust_groups,
    exponential_trust_score,
    fit_trust_calibration,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_version(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return completed.stdout.strip()
    return "unavailable_no_git_repository"


def load_e1_config(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the E1 trust-evaluation configuration."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    for section in (
        "project",
        "input",
        "calibration",
        "trust_mapping",
        "evaluation",
        "plot",
        "output",
    ):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing mapping section: '{section}'.")

    input_config = config["input"]
    input_fields = (
        "simulation_data_root",
        "simulation_run_id",
        "simulation_csv_filename",
        "simulation_csv_sha256",
        "trust_scoring_data_root",
        "trust_scoring_run_id",
        "trust_scoring_arrays_filename",
        "trust_scoring_arrays_sha256",
        "trust_scoring_metadata_filename",
        "trust_scoring_metadata_sha256",
    )
    for key in input_fields:
        if not isinstance(input_config.get(key), str) or not input_config[key]:
            raise ValueError(f"'input.{key}' must be a non-empty string.")
    for key in [name for name in input_fields if name.endswith("sha256")]:
        value = input_config[key]
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"'input.{key}' must be a lowercase SHA-256.")

    calibration = config["calibration"]
    if calibration.get("source") != "normal_train_clean_reference_distances":
        raise ValueError("E1 calibration must use normal train clean reference distances.")
    if calibration.get("quantile_method") != "linear":
        raise ValueError("E1 quantile_method must be 'linear'.")
    high_quantile = calibration.get("high_quantile")
    low_quantile = calibration.get("low_quantile")
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in (high_quantile, low_quantile)
    ) or not 0.0 < high_quantile < low_quantile < 1.0:
        raise ValueError("Calibration quantiles must satisfy 0 < high < low < 1.")

    mapping = config["trust_mapping"]
    if mapping.get("method") != "exponential":
        raise ValueError("E1 trust_mapping.method must be 'exponential'.")
    if mapping.get("formula") != "exp(-d2/(2*tau))":
        raise ValueError("E1 trust mapping formula is not the documented baseline.")
    if mapping.get("tau_source") != "q90":
        raise ValueError("E1 trust_mapping.tau_source must be 'q90'.")

    evaluation = config["evaluation"]
    if evaluation.get("evaluation_splits") != ["train", "val"]:
        raise ValueError("E1 metrics must use exactly train and val.")
    if evaluation.get("holdout_split") != "test":
        raise ValueError("E1 holdout_split must be 'test'.")
    if evaluation.get("threshold_names") != ["q90", "q99"]:
        raise ValueError("E1 must report both q90 and q99 rules.")
    if evaluation.get("positive_label") != "is_corrupted":
        raise ValueError("E1 positive label must be 'is_corrupted'.")
    if evaluation.get("anomaly_score") != "squared_mahalanobis_distance":
        raise ValueError("E1 anomaly score must be squared Mahalanobis distance.")
    if config["plot"].get("split") != "val":
        raise ValueError("The first E1 diagnostic plot must use validation data.")
    if config["plot"].get("sequence_selection") != "most_corrupted_then_sequence_id":
        raise ValueError("Unsupported plot sequence selection rule.")
    for key in ("data_root", "figures_root", "tables_root"):
        if not isinstance(config["output"].get(key), str):
            raise ValueError(f"'output.{key}' must be a string path.")
    return config


def _make_run_id(configuration_hash: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"e1trust_{timestamp}_{configuration_hash[:8]}"


def _resolve_root(path_value: str, project_root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def _ordered_label_frame(
    simulation_csv: Path,
    sequence_ids: np.ndarray,
    time_indices: np.ndarray,
    split_labels: np.ndarray,
) -> pd.DataFrame:
    frame = pd.read_csv(simulation_csv)
    required = {
        "sequence_id",
        "time_index",
        "split",
        "condition",
        "is_corrupted",
        "corruption_type",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Simulation CSV is missing columns: {sorted(missing)}")
    if frame.duplicated(["sequence_id", "time_index"]).any():
        raise ValueError("Simulation CSV contains duplicate sequence/time keys.")
    if frame[list(required)].isna().any().any():
        raise ValueError("Required simulation labels must not contain missing values.")
    frame["sequence_id"] = frame["sequence_id"].astype(str)
    frame["corruption_type"] = frame["corruption_type"].astype(str)

    ordered: list[pd.DataFrame] = []
    if time_indices.shape[0] != len(sequence_ids):
        raise ValueError("Distance sequence IDs and time indices are inconsistent.")
    for sequence_index, sequence_id in enumerate(sequence_ids):
        sequence_frame = frame.loc[frame["sequence_id"] == str(sequence_id)].sort_values(
            "time_index"
        )
        if sequence_frame.empty:
            raise ValueError(f"Simulation CSV has no sequence '{sequence_id}'.")
        expected_times = time_indices[sequence_index]
        if not np.array_equal(
            sequence_frame["time_index"].to_numpy(dtype=int),
            expected_times,
        ):
            raise ValueError(f"Time indices differ for sequence '{sequence_id}'.")
        sequence_splits = sequence_frame["split"].unique()
        if len(sequence_splits) != 1 or sequence_splits[0] != split_labels[sequence_index]:
            raise ValueError(f"Split identity differs for sequence '{sequence_id}'.")
        ordered.append(sequence_frame)
    ordered_frame = pd.concat(ordered, ignore_index=True)
    if len(ordered_frame) != time_indices.size:
        raise ValueError("Simulation labels and distance rows have different sizes.")
    return ordered_frame


def run_e1_evaluation(
    config_path: str | Path,
    *,
    run_id: str | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Calibrate trust from train references and evaluate only train/validation."""

    project_root_path = Path(project_root).resolve()
    config = load_e1_config(Path(config_path).resolve())
    configuration_hash = config_hash(config)
    actual_run_id = run_id or _make_run_id(configuration_hash)
    input_config = config["input"]

    simulation_dir = _resolve_root(
        input_config["simulation_data_root"], project_root_path
    ) / input_config["simulation_run_id"]
    trust_scoring_dir = _resolve_root(
        input_config["trust_scoring_data_root"], project_root_path
    ) / input_config["trust_scoring_run_id"]
    paths = {
        "simulation_csv": simulation_dir / input_config["simulation_csv_filename"],
        "trust_scoring_arrays": trust_scoring_dir
        / input_config["trust_scoring_arrays_filename"],
        "trust_scoring_metadata": trust_scoring_dir
        / input_config["trust_scoring_metadata_filename"],
    }
    expected_hashes = {
        "simulation_csv": input_config["simulation_csv_sha256"],
        "trust_scoring_arrays": input_config["trust_scoring_arrays_sha256"],
        "trust_scoring_metadata": input_config["trust_scoring_metadata_sha256"],
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required input file not found: {path}")
        if _sha256_file(path) != expected_hashes[name]:
            raise ValueError(f"Input SHA-256 mismatch for {name}.")

    with paths["trust_scoring_metadata"].open(encoding="utf-8") as handle:
        scoring_metadata = json.load(handle)
    if scoring_metadata["trust_scoring_run_id"] != input_config["trust_scoring_run_id"]:
        raise ValueError("Trust-scoring metadata run ID does not match the config.")
    if scoring_metadata["distances_npz_sha256"] != expected_hashes[
        "trust_scoring_arrays"
    ]:
        raise ValueError("Trust-scoring metadata arrays hash is inconsistent.")
    if scoring_metadata["data_source"] != "synthetic":
        raise ValueError("E1 currently accepts only declared synthetic inputs.")
    if scoring_metadata["thresholds_computed"] is not False:
        raise ValueError("Input distance run must not contain fitted thresholds.")

    with np.load(paths["trust_scoring_arrays"]) as payload:
        observation_squared = payload["squared_mahalanobis_observations"].astype(float)
        reference_squared = payload["squared_mahalanobis_reference"].astype(float)
        sequence_ids = payload["sequence_ids"].astype(str)
        time_indices = payload["time_indices"].astype(int)
        split_labels = payload["split_labels"].astype(str)
    if observation_squared.ndim != 2:
        raise ValueError("Observation distances must have shape [sequence, time].")
    if time_indices.shape != observation_squared.shape:
        raise ValueError("Time indices do not match observation distances.")
    if sequence_ids.shape != (observation_squared.shape[0],):
        raise ValueError("Sequence IDs do not match observation distances.")
    if split_labels.shape != (observation_squared.shape[0],):
        raise ValueError("Split labels do not match observation distances.")
    if set(split_labels) != {"train", "val", "test"}:
        raise ValueError("Distance inputs must preserve train, val, and test splits.")

    labels = _ordered_label_frame(
        paths["simulation_csv"],
        sequence_ids,
        time_indices,
        split_labels,
    )
    calibration_config = config["calibration"]
    calibration = fit_trust_calibration(
        reference_squared,
        high_quantile=float(calibration_config["high_quantile"]),
        low_quantile=float(calibration_config["low_quantile"]),
        quantile_method=calibration_config["quantile_method"],
        tau_source=config["trust_mapping"]["tau_source"],
        fit_source=calibration_config["source"],
    )
    trust_scores = exponential_trust_score(observation_squared, calibration)
    trust_groups = assign_trust_groups(observation_squared, calibration)

    pointwise = labels[
        ["sequence_id", "time_index", "split", "condition", "corruption_type"]
    ].copy()
    pointwise.insert(
        4,
        "corruption_label",
        labels["is_corrupted"].astype(bool).astype(int),
    )
    pointwise["mahalanobis_distance_squared"] = observation_squared.reshape(-1)
    pointwise["mahalanobis_distance"] = np.sqrt(observation_squared.reshape(-1))
    pointwise["trust_score"] = trust_scores.reshape(-1)
    pointwise["trust_group"] = trust_groups.reshape(-1)
    if not np.isfinite(
        pointwise[
            ["mahalanobis_distance_squared", "mahalanobis_distance", "trust_score"]
        ].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("Pointwise E1 numeric outputs must all be finite.")

    thresholds = {
        "q90": calibration.high_threshold_squared,
        "q99": calibration.low_threshold_squared,
    }
    evaluation_splits = tuple(config["evaluation"]["evaluation_splits"])
    metrics = build_detection_metrics_table(
        pointwise,
        evaluation_splits=evaluation_splits,
        thresholds=thresholds,
    )
    type_recall = build_corruption_type_recall_table(
        pointwise,
        evaluation_splits=evaluation_splits,
        thresholds=thresholds,
    )
    condition_false_positives = build_condition_false_positive_table(
        pointwise,
        evaluation_splits=evaluation_splits,
        thresholds=thresholds,
    )
    if config["evaluation"]["holdout_split"] in set(metrics["split"]):
        raise RuntimeError("The holdout test split must not appear in E1 metric tables.")

    output = config["output"]
    data_run_dir = _resolve_root(output["data_root"], project_root_path) / actual_run_id
    figures_run_dir = _resolve_root(
        output["figures_root"], project_root_path
    ) / actual_run_id
    tables_run_dir = _resolve_root(output["tables_root"], project_root_path) / actual_run_id
    for run_dir in (data_run_dir, figures_run_dir, tables_run_dir):
        if run_dir.exists():
            raise FileExistsError(
                f"Run ID already exists and will not be overwritten: {run_dir}"
            )
    data_run_dir.mkdir(parents=True, exist_ok=False)
    figures_run_dir.mkdir(parents=True, exist_ok=False)
    tables_run_dir.mkdir(parents=True, exist_ok=False)

    pointwise_path = data_run_dir / "pointwise_trust_scores.csv"
    calibration_path = data_run_dir / "trust_calibration.json"
    metadata_path = data_run_dir / "metadata.json"
    config_snapshot_path = data_run_dir / "config_snapshot.yaml"
    metrics_path = tables_run_dir / "detection_metrics.csv"
    type_recall_path = tables_run_dir / "corruption_type_recall.csv"
    condition_fpr_path = tables_run_dir / "condition_false_positive_rates.csv"
    figure_path = figures_run_dir / "distance_trust_timeseries.png"

    pointwise.to_csv(pointwise_path, index=False, float_format="%.17g")
    metrics.to_csv(metrics_path, index=False, float_format="%.17g")
    type_recall.to_csv(type_recall_path, index=False, float_format="%.17g")
    condition_false_positives.to_csv(
        condition_fpr_path,
        index=False,
        float_format="%.17g",
    )
    with calibration_path.open("w", encoding="utf-8") as handle:
        json.dump(calibration.to_dict(), handle, ensure_ascii=False, indent=2)
    with config_snapshot_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    _, selected_sequence_id = plot_distance_trust_timeseries(
        pointwise,
        figure_path,
        plot_split=config["plot"]["split"],
        high_threshold_squared=calibration.high_threshold_squared,
        low_threshold_squared=calibration.low_threshold_squared,
    )

    metadata: dict[str, Any] = {
        "e1_run_id": actual_run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "synthetic",
        "input_simulation_run_id": input_config["simulation_run_id"],
        "input_trust_scoring_run_id": input_config["trust_scoring_run_id"],
        "config_sha256": configuration_hash,
        "code_version": _code_version(project_root_path),
        "calibration": calibration.to_dict(),
        "evaluation_splits": list(evaluation_splits),
        "holdout_split": config["evaluation"]["holdout_split"],
        "holdout_metrics_computed": False,
        "num_pointwise_rows": int(len(pointwise)),
        "trust_score_minimum": float(pointwise["trust_score"].min()),
        "trust_score_maximum": float(pointwise["trust_score"].max()),
        "trust_group_counts": {
            key: int(value)
            for key, value in pointwise["trust_group"].value_counts().sort_index().items()
        },
        "plot_split": config["plot"]["split"],
        "plot_sequence_id": selected_sequence_id,
        "input_sha256": expected_hashes,
        "output_sha256": {
            "pointwise_csv": _sha256_file(pointwise_path),
            "calibration_json": _sha256_file(calibration_path),
            "detection_metrics_csv": _sha256_file(metrics_path),
            "corruption_type_recall_csv": _sha256_file(type_recall_path),
            "condition_false_positive_rates_csv": _sha256_file(condition_fpr_path),
            "distance_trust_figure_png": _sha256_file(figure_path),
        },
        "files": {
            "simulation_csv": str(paths["simulation_csv"].resolve()),
            "trust_scoring_arrays": str(paths["trust_scoring_arrays"].resolve()),
            "pointwise_csv": str(pointwise_path.resolve()),
            "calibration_json": str(calibration_path.resolve()),
            "detection_metrics_csv": str(metrics_path.resolve()),
            "corruption_type_recall_csv": str(type_recall_path.resolve()),
            "condition_false_positive_rates_csv": str(condition_fpr_path.resolve()),
            "distance_trust_figure_png": str(figure_path.resolve()),
            "config_snapshot": str(config_snapshot_path.resolve()),
            "metadata_json": str(metadata_path.resolve()),
        },
        "scientific_scope_note": (
            "Synthetic E1 baseline. Thresholds and tau use only normal-train-clean "
            "reference distances. Metrics use train/validation; test metrics are held "
            "out. Results do not establish industrial validity or forecasting benefit."
        ),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E1 train-reference trust calibration and validation evaluation."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--run-id", help="Optional unique run ID; existing runs fail.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = _parse_args()
    metadata = run_e1_evaluation(args.config, run_id=args.run_id)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
