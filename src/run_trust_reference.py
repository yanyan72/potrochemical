"""Command-line pipeline for Milestone 2 normal-reference estimation."""

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
import sklearn
import yaml

from .config import SUPPORTED_FEATURES, config_hash
from .preprocess import PreprocessorParams, transform
from .trust_reference import fit_trust_reference


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


def load_trust_reference_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the Milestone 2 configuration."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    for section in ("project", "input", "data", "reference", "output"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing mapping section: '{section}'.")

    input_config = config["input"]
    string_fields = (
        "simulation_data_root",
        "simulation_run_id",
        "simulation_csv_filename",
        "simulation_csv_sha256",
        "preprocessing_data_root",
        "preprocessing_run_id",
        "preprocessing_arrays_filename",
        "preprocessing_arrays_sha256",
        "preprocessing_params_filename",
        "preprocessing_params_sha256",
        "preprocessing_metadata_filename",
        "preprocessing_metadata_sha256",
    )
    for key in string_fields:
        if not isinstance(input_config.get(key), str) or not input_config[key]:
            raise ValueError(f"'input.{key}' must be a non-empty string.")
    for key in [name for name in string_fields if name.endswith("sha256")]:
        value = input_config[key]
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"'input.{key}' must be a lowercase SHA-256.")

    features = tuple(config["data"].get("features", ()))
    if features != SUPPORTED_FEATURES:
        raise ValueError(
            "Milestone 2 requires features in this order: "
            f"{list(SUPPORTED_FEATURES)}."
        )
    reference = config["reference"]
    expected_values = {
        "fit_split": "train",
        "fit_condition": "normal",
        "fit_source": "clean",
        "standardization_params_source": "milestone1_clean_train",
        "covariance_estimator": "ledoit_wolf",
    }
    for key, expected in expected_values.items():
        if reference.get(key) != expected:
            raise ValueError(f"'reference.{key}' must be '{expected}'.")
    if reference.get("assume_centered") is not False:
        raise ValueError("'reference.assume_centered' must be false.")
    if not isinstance(config["output"].get("data_root"), str):
        raise ValueError("'output.data_root' must be a string path.")
    return config


def _load_preprocessor_params(path: Path) -> PreprocessorParams:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return PreprocessorParams(
        feature_names=tuple(raw["feature_names"]),
        imputation_strategy=str(raw["imputation_strategy"]),
        imputation_values=np.asarray(raw["imputation_values"], dtype=float),
        means=np.asarray(raw["means"], dtype=float),
        scales=np.asarray(raw["scales"], dtype=float),
        std_ddof=int(raw["std_ddof"]),
        fit_source=str(raw["fit_source"]),
        fit_split=str(raw["fit_split"]),
    )


def _make_run_id(configuration_hash: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"trustref_{timestamp}_{configuration_hash[:8]}"


def run_trust_reference(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
    run_id: str | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Build a clean-train-normal reference and fit Ledoit-Wolf covariance."""

    project_root_path = Path(project_root).resolve()
    config = load_trust_reference_config(Path(config_path).resolve())
    configuration_hash = config_hash(config)
    actual_run_id = run_id or _make_run_id(configuration_hash)
    input_config = config["input"]

    simulation_root = Path(input_config["simulation_data_root"])
    preprocessing_root = Path(input_config["preprocessing_data_root"])
    if not simulation_root.is_absolute():
        simulation_root = project_root_path / simulation_root
    if not preprocessing_root.is_absolute():
        preprocessing_root = project_root_path / preprocessing_root
    simulation_dir = simulation_root / input_config["simulation_run_id"]
    preprocessing_dir = preprocessing_root / input_config["preprocessing_run_id"]
    paths = {
        "simulation_csv": simulation_dir / input_config["simulation_csv_filename"],
        "preprocessing_arrays": preprocessing_dir
        / input_config["preprocessing_arrays_filename"],
        "preprocessing_params": preprocessing_dir
        / input_config["preprocessing_params_filename"],
        "preprocessing_metadata": preprocessing_dir
        / input_config["preprocessing_metadata_filename"],
    }
    expected_hashes = {
        "simulation_csv": input_config["simulation_csv_sha256"],
        "preprocessing_arrays": input_config["preprocessing_arrays_sha256"],
        "preprocessing_params": input_config["preprocessing_params_sha256"],
        "preprocessing_metadata": input_config["preprocessing_metadata_sha256"],
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required input file not found: {path}")
        if _sha256_file(path) != expected_hashes[name]:
            raise ValueError(f"Input SHA-256 mismatch for {name}.")

    with paths["preprocessing_metadata"].open(encoding="utf-8") as handle:
        preprocessing_metadata = json.load(handle)
    if preprocessing_metadata["input_dataset_id"] != input_config["simulation_run_id"]:
        raise ValueError("Preprocessing metadata references a different simulation run.")
    if preprocessing_metadata["preprocessing_run_id"] != input_config["preprocessing_run_id"]:
        raise ValueError("Preprocessing metadata run ID does not match the config.")
    if preprocessing_metadata["fit_source"] != "clean_train":
        raise ValueError("Preprocessing parameters were not fitted from clean train.")

    with np.load(paths["preprocessing_arrays"]) as payload:
        stored_sequence_ids = payload["sequence_ids"].astype(str)
        stored_time_indices = payload["time_indices"].astype(int)
        stored_split_labels = payload["split_labels"].astype(str)
        stored_feature_names = payload["feature_names"].astype(str)
    feature_names = tuple(config["data"]["features"])
    if tuple(stored_feature_names) != feature_names:
        raise ValueError("Preprocessing feature order does not match the config.")

    frame = pd.read_csv(paths["simulation_csv"])
    required = {
        "sequence_id",
        "time_index",
        "split",
        "condition",
        *[f"{name}_clean" for name in feature_names],
    }
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Simulation CSV is missing columns: {sorted(missing_columns)}")
    if frame.duplicated(["sequence_id", "time_index"]).any():
        raise ValueError("Simulation CSV contains duplicate sequence/time keys.")

    sequence_table = frame[["sequence_id", "split", "condition"]].drop_duplicates()
    if sequence_table.groupby("sequence_id").size().max() != 1:
        raise ValueError("Each sequence must have exactly one split and condition.")
    csv_sequence_ids = pd.unique(frame["sequence_id"]).astype(str)
    if not np.array_equal(csv_sequence_ids, stored_sequence_ids):
        raise ValueError("Simulation and preprocessing sequence order differ.")
    csv_split_labels = (
        sequence_table.set_index(sequence_table["sequence_id"].astype(str))
        .reindex(stored_sequence_ids)["split"]
        .to_numpy(dtype=str)
    )
    if not np.array_equal(csv_split_labels, stored_split_labels):
        raise ValueError("Simulation and preprocessing split labels differ.")

    reference_config = config["reference"]
    reference_sequence_table = sequence_table.loc[
        (sequence_table["split"] == reference_config["fit_split"])
        & (sequence_table["condition"] == reference_config["fit_condition"])
    ]
    reference_sequence_ids = reference_sequence_table["sequence_id"].astype(str).to_numpy(
        dtype=str
    )
    if reference_sequence_ids.size == 0:
        raise ValueError("No train-normal reference sequences were found.")
    reference_frame = frame.loc[
        frame["sequence_id"].astype(str).isin(reference_sequence_ids)
    ].copy()
    reference_frame["sequence_id"] = pd.Categorical(
        reference_frame["sequence_id"].astype(str),
        categories=reference_sequence_ids,
        ordered=True,
    )
    reference_frame = reference_frame.sort_values(["sequence_id", "time_index"])
    sequence_lengths = reference_frame.groupby("sequence_id", observed=False).size()
    if sequence_lengths.nunique() != 1:
        raise ValueError("Reference sequences must have equal lengths.")
    sequence_length = int(sequence_lengths.iloc[0])
    expected_times = np.tile(np.arange(sequence_length), len(reference_sequence_ids))
    if not np.array_equal(reference_frame["time_index"].to_numpy(dtype=int), expected_times):
        raise ValueError("Reference sequences have incomplete time indices.")

    params = _load_preprocessor_params(paths["preprocessing_params"])
    if params.feature_names != feature_names or params.fit_split != "train":
        raise ValueError("Preprocessor parameters are incompatible with this run.")
    clean_reference = reference_frame[
        [f"{name}_clean" for name in feature_names]
    ].to_numpy(dtype=float)
    standardized_reference = transform(clean_reference, params)
    reference_params = fit_trust_reference(
        standardized_reference,
        feature_names,
        assume_centered=bool(reference_config["assume_centered"]),
        fit_split=reference_config["fit_split"],
        fit_condition=reference_config["fit_condition"],
    )

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

    arrays_path = run_dir / "reference_set.npz"
    params_path = run_dir / "trust_reference_params.json"
    metadata_path = run_dir / "metadata.json"
    config_snapshot_path = run_dir / "config_snapshot.yaml"
    np.savez_compressed(
        arrays_path,
        standardized_clean_reference=standardized_reference,
        reference_sequence_ids=np.asarray(reference_sequence_ids, dtype=str),
        time_indices=np.tile(np.arange(sequence_length), (len(reference_sequence_ids), 1)),
        feature_names=np.asarray(feature_names),
        location=reference_params.location,
        covariance=reference_params.covariance,
        precision=reference_params.precision,
    )
    with params_path.open("w", encoding="utf-8") as handle:
        json.dump(reference_params.to_dict(), handle, ensure_ascii=False, indent=2)
    with config_snapshot_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)

    metadata: dict[str, Any] = {
        "trust_reference_run_id": actual_run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "synthetic",
        "input_simulation_run_id": input_config["simulation_run_id"],
        "input_preprocessing_run_id": input_config["preprocessing_run_id"],
        "config_sha256": configuration_hash,
        "reference_npz_sha256": _sha256_file(arrays_path),
        "params_json_sha256": _sha256_file(params_path),
        "code_version": _code_version(project_root_path),
        "sklearn_version": sklearn.__version__,
        "feature_names": list(feature_names),
        "fit_split": reference_params.fit_split,
        "fit_condition": reference_params.fit_condition,
        "fit_source": reference_params.fit_source,
        "covariance_estimator": reference_params.estimator,
        "assume_centered": reference_params.assume_centered,
        "num_reference_sequences": int(len(reference_sequence_ids)),
        "sequence_length": sequence_length,
        "num_reference_rows": reference_params.num_reference_rows,
        "shrinkage": reference_params.shrinkage,
        "diagnostics": reference_params.diagnostics(),
        "input_sha256": expected_hashes,
        "files": {
            "simulation_csv": str(paths["simulation_csv"].resolve()),
            "preprocessing_arrays": str(paths["preprocessing_arrays"].resolve()),
            "preprocessing_params": str(paths["preprocessing_params"].resolve()),
            "reference_npz": str(arrays_path.resolve()),
            "trust_reference_params_json": str(params_path.resolve()),
            "config_snapshot": str(config_snapshot_path.resolve()),
            "metadata_json": str(metadata_path.resolve()),
        },
        "scientific_scope_note": (
            "Synthetic clean-train-normal oracle reference only. This run estimates "
            "Ledoit-Wolf covariance but does not compute Mahalanobis distances, "
            "thresholds, trust scores, anomaly metrics, forecasts, or industrial validity."
        ),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a clean-train-normal Ledoit-Wolf trust reference."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--output-root", help="Optional output root override.")
    parser.add_argument("--run-id", help="Optional unique run ID; existing runs fail.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = _parse_args()
    metadata = run_trust_reference(
        args.config,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
