"""Command-line pipeline for leakage-safe Milestone 1 preprocessing."""

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

from .config import SUPPORTED_FEATURES, config_hash
from .preprocess import fit_preprocessor, impute_missing, transform


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


def load_preprocessing_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the Milestone 1 preprocessing configuration."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    for section in ("project", "input", "data", "preprocessing", "output"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing mapping section: '{section}'.")

    input_config = config["input"]
    for key in ("data_root", "run_id", "csv_filename", "expected_csv_sha256"):
        if not isinstance(input_config.get(key), str) or not input_config[key]:
            raise ValueError(f"'input.{key}' must be a non-empty string.")
    expected_hash = input_config["expected_csv_sha256"]
    if len(expected_hash) != 64 or any(char not in "0123456789abcdef" for char in expected_hash):
        raise ValueError("'input.expected_csv_sha256' must be a lowercase SHA-256.")

    features = tuple(config["data"].get("features", ()))
    if features != SUPPORTED_FEATURES:
        raise ValueError(
            "Milestone 1 requires features in this order: "
            f"{list(SUPPORTED_FEATURES)}."
        )
    preprocessing = config["preprocessing"]
    if preprocessing.get("imputation_strategy") != "feature_median":
        raise ValueError("Only feature_median imputation is supported.")
    if preprocessing.get("fit_source") != "clean_train":
        raise ValueError("'preprocessing.fit_source' must be 'clean_train'.")
    if preprocessing.get("fit_split") != "train":
        raise ValueError("'preprocessing.fit_split' must be 'train'.")
    std_ddof = preprocessing.get("std_ddof")
    if not isinstance(std_ddof, int) or isinstance(std_ddof, bool) or std_ddof < 0:
        raise ValueError("'preprocessing.std_ddof' must be a non-negative integer.")
    if not isinstance(config["output"].get("data_root"), str):
        raise ValueError("'output.data_root' must be a string path.")
    return config


def _load_dense_sequences(
    csv_path: Path,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(csv_path)
    required = {
        "sequence_id",
        "time_index",
        "split",
        *[f"{name}_clean" for name in feature_names],
        *[f"{name}_obs" for name in feature_names],
    }
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing_columns)}")
    if frame.duplicated(["sequence_id", "time_index"]).any():
        raise ValueError("Input CSV contains duplicate sequence/time keys.")

    sequence_ids = pd.unique(frame["sequence_id"]).astype(str)
    clean_sequences: list[np.ndarray] = []
    observed_sequences: list[np.ndarray] = []
    time_sequences: list[np.ndarray] = []
    split_labels: list[str] = []
    expected_length: int | None = None
    for sequence_id in sequence_ids:
        sequence_frame = frame.loc[frame["sequence_id"].astype(str) == sequence_id].sort_values(
            "time_index"
        )
        split_values = sequence_frame["split"].unique()
        if len(split_values) != 1:
            raise ValueError(f"Sequence {sequence_id} belongs to multiple splits.")
        times = sequence_frame["time_index"].to_numpy(dtype=int)
        if not np.array_equal(times, np.arange(len(times))):
            raise ValueError(f"Sequence {sequence_id} has incomplete time indices.")
        if expected_length is None:
            expected_length = len(times)
        elif len(times) != expected_length:
            raise ValueError("All sequences must have the same length.")
        clean_sequences.append(
            sequence_frame[[f"{name}_clean" for name in feature_names]].to_numpy(
                dtype=float
            )
        )
        observed_sequences.append(
            sequence_frame[[f"{name}_obs" for name in feature_names]].to_numpy(
                dtype=float
            )
        )
        time_sequences.append(times)
        split_labels.append(str(split_values[0]))

    clean = np.stack(clean_sequences)
    observed = np.stack(observed_sequences)
    time_indices = np.stack(time_sequences)
    splits = np.asarray(split_labels)
    if not np.isfinite(clean).all():
        raise ValueError("Clean input features must all be finite.")
    if np.isinf(observed).any():
        raise ValueError("Observed input features must not contain infinity.")
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("Input data must contain train, val, and test sequences.")
    return clean, observed, sequence_ids, time_indices, splits


def _make_run_id(configuration_hash: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"preprocess_{timestamp}_{configuration_hash[:8]}"


def run_preprocessing(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
    run_id: str | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Fit on clean train and transform all observed splits without leakage."""

    project_root_path = Path(project_root).resolve()
    config_path_obj = Path(config_path).resolve()
    config = load_preprocessing_config(config_path_obj)
    configuration_hash = config_hash(config)
    actual_run_id = run_id or _make_run_id(configuration_hash)

    input_root = Path(config["input"]["data_root"])
    if not input_root.is_absolute():
        input_root = project_root_path / input_root
    input_dir = input_root / config["input"]["run_id"]
    csv_path = input_dir / config["input"]["csv_filename"]
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    input_hash = _sha256_file(csv_path)
    if input_hash != config["input"]["expected_csv_sha256"]:
        raise ValueError("Input CSV SHA-256 does not match the configured dataset.")

    configured_output_root = Path(config["output"]["data_root"])
    output_root_path = Path(output_root) if output_root is not None else configured_output_root
    if not output_root_path.is_absolute():
        output_root_path = project_root_path / output_root_path
    run_dir = output_root_path / actual_run_id
    if run_dir.exists():
        raise FileExistsError(
            f"Run ID already exists and will not be overwritten: {actual_run_id}"
        )
    run_dir.mkdir(parents=True, exist_ok=False)

    feature_names = tuple(config["data"]["features"])
    clean, observed, sequence_ids, time_indices, splits = _load_dense_sequences(
        csv_path,
        feature_names,
    )
    train_mask = splits == config["preprocessing"]["fit_split"]
    params = fit_preprocessor(
        clean[train_mask],
        feature_names,
        imputation_strategy=config["preprocessing"]["imputation_strategy"],
        std_ddof=int(config["preprocessing"]["std_ddof"]),
    )
    missing_mask = np.isnan(observed)
    imputed = impute_missing(observed, params)
    standardized = transform(observed, params)

    arrays_path = run_dir / "standardized_observations.npz"
    params_path = run_dir / "preprocessor_params.json"
    metadata_path = run_dir / "metadata.json"
    config_snapshot_path = run_dir / "config_snapshot.yaml"
    np.savez_compressed(
        arrays_path,
        standardized_observations=standardized,
        imputed_observations=imputed,
        original_missing_mask=missing_mask,
        sequence_ids=sequence_ids,
        time_indices=time_indices,
        split_labels=splits,
        feature_names=np.asarray(feature_names),
    )
    with params_path.open("w", encoding="utf-8") as handle:
        json.dump(params.to_dict(), handle, ensure_ascii=False, indent=2)
    with config_snapshot_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)

    metadata: dict[str, Any] = {
        "preprocessing_run_id": actual_run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "synthetic",
        "input_dataset_id": config["input"]["run_id"],
        "input_csv_sha256": input_hash,
        "config_sha256": configuration_hash,
        "arrays_npz_sha256": _sha256_file(arrays_path),
        "code_version": _code_version(project_root_path),
        "feature_names": list(feature_names),
        "fit_source": params.fit_source,
        "fit_split": params.fit_split,
        "imputation_strategy": params.imputation_strategy,
        "standardization": "train_mean_and_sample_std",
        "std_ddof": params.std_ddof,
        "num_sequences": int(clean.shape[0]),
        "sequence_length": int(clean.shape[1]),
        "num_features": int(clean.shape[2]),
        "split_sequence_counts": {
            split: int(np.sum(splits == split)) for split in ("train", "val", "test")
        },
        "missing_values_before": int(missing_mask.sum()),
        "missing_values_after_imputation": int(np.isnan(imputed).sum()),
        "non_finite_values_after_standardization": int(
            standardized.size - np.isfinite(standardized).sum()
        ),
        "files": {
            "input_csv": str(csv_path.resolve()),
            "arrays_npz": str(arrays_path.resolve()),
            "preprocessor_params_json": str(params_path.resolve()),
            "config_snapshot": str(config_snapshot_path.resolve()),
            "metadata_json": str(metadata_path.resolve()),
        },
        "scientific_scope_note": (
            "Synthetic preprocessing baseline only; clean-train fit does not "
            "represent real deployment access to hidden ground truth. No trust "
            "score, anomaly detection, forecasting, physics, blockchain, or UI."
        ),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit clean-train imputation/scaling and transform synthetic observations."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--output-root", help="Optional output root override.")
    parser.add_argument("--run-id", help="Optional unique run ID; existing runs fail.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = _parse_args()
    metadata = run_preprocessing(
        args.config,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
