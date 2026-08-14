"""Command-line pipeline for Milestone 3 squared Mahalanobis distances."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import SUPPORTED_FEATURES, config_hash
from .trust_score import squared_mahalanobis_distance


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


def load_trust_scoring_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the Milestone 3 configuration."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    for section in ("project", "input", "data", "distance", "output"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing mapping section: '{section}'.")

    input_config = config["input"]
    string_fields = (
        "preprocessing_data_root",
        "preprocessing_run_id",
        "preprocessing_arrays_filename",
        "preprocessing_arrays_sha256",
        "preprocessing_metadata_filename",
        "preprocessing_metadata_sha256",
        "reference_data_root",
        "reference_run_id",
        "reference_arrays_filename",
        "reference_arrays_sha256",
        "reference_params_filename",
        "reference_params_sha256",
        "reference_metadata_filename",
        "reference_metadata_sha256",
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
            "Milestone 3 requires features in this order: "
            f"{list(SUPPORTED_FEATURES)}."
        )
    distance = config["distance"]
    if distance.get("method") != "squared_mahalanobis":
        raise ValueError("'distance.method' must be 'squared_mahalanobis'.")
    if (
        distance.get("parameter_source")
        != "milestone2_clean_train_normal_ledoit_wolf"
    ):
        raise ValueError("Milestone 3 must use the frozen Milestone 2 parameters.")
    tolerance = distance.get("negative_tolerance")
    if (
        not isinstance(tolerance, (int, float))
        or isinstance(tolerance, bool)
        or not np.isfinite(tolerance)
        or tolerance < 0.0
    ):
        raise ValueError("'distance.negative_tolerance' must be non-negative.")
    if not isinstance(config["output"].get("data_root"), str):
        raise ValueError("'output.data_root' must be a string path.")
    return config


def _make_run_id(configuration_hash: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"trustscore_{timestamp}_{configuration_hash[:8]}"


def _summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(values.size),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "non_finite_count": int(values.size - np.isfinite(values).sum()),
        "negative_count": int(np.sum(values < 0.0)),
    }


def run_trust_scoring(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
    run_id: str | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Apply frozen M2 parameters to M1 observations without fitting thresholds."""

    project_root_path = Path(project_root).resolve()
    config = load_trust_scoring_config(Path(config_path).resolve())
    configuration_hash = config_hash(config)
    actual_run_id = run_id or _make_run_id(configuration_hash)
    input_config = config["input"]

    preprocessing_root = Path(input_config["preprocessing_data_root"])
    reference_root = Path(input_config["reference_data_root"])
    if not preprocessing_root.is_absolute():
        preprocessing_root = project_root_path / preprocessing_root
    if not reference_root.is_absolute():
        reference_root = project_root_path / reference_root
    preprocessing_dir = preprocessing_root / input_config["preprocessing_run_id"]
    reference_dir = reference_root / input_config["reference_run_id"]
    paths = {
        "preprocessing_arrays": preprocessing_dir
        / input_config["preprocessing_arrays_filename"],
        "preprocessing_metadata": preprocessing_dir
        / input_config["preprocessing_metadata_filename"],
        "reference_arrays": reference_dir / input_config["reference_arrays_filename"],
        "reference_params": reference_dir / input_config["reference_params_filename"],
        "reference_metadata": reference_dir
        / input_config["reference_metadata_filename"],
    }
    expected_hashes = {
        name: input_config[f"{name}_sha256"] for name in paths
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required input file not found: {path}")
        if _sha256_file(path) != expected_hashes[name]:
            raise ValueError(f"Input SHA-256 mismatch for {name}.")

    with paths["preprocessing_metadata"].open(encoding="utf-8") as handle:
        preprocessing_metadata = json.load(handle)
    with paths["reference_metadata"].open(encoding="utf-8") as handle:
        reference_metadata = json.load(handle)
    with paths["reference_params"].open(encoding="utf-8") as handle:
        reference_params = json.load(handle)

    if (
        preprocessing_metadata["preprocessing_run_id"]
        != input_config["preprocessing_run_id"]
    ):
        raise ValueError("Preprocessing metadata run ID does not match the config.")
    if preprocessing_metadata["arrays_npz_sha256"] != expected_hashes[
        "preprocessing_arrays"
    ]:
        raise ValueError("Preprocessing metadata arrays hash is inconsistent.")
    if reference_metadata["trust_reference_run_id"] != input_config[
        "reference_run_id"
    ]:
        raise ValueError("Reference metadata run ID does not match the config.")
    if reference_metadata["input_preprocessing_run_id"] != input_config[
        "preprocessing_run_id"
    ]:
        raise ValueError("Reference and observation preprocessing runs differ.")
    if reference_metadata["reference_npz_sha256"] != expected_hashes[
        "reference_arrays"
    ]:
        raise ValueError("Reference metadata arrays hash is inconsistent.")
    if reference_metadata["params_json_sha256"] != expected_hashes[
        "reference_params"
    ]:
        raise ValueError("Reference metadata parameters hash is inconsistent.")
    if reference_metadata["fit_split"] != "train":
        raise ValueError("Reference parameters were not fitted on train.")
    if reference_metadata["fit_condition"] != "normal":
        raise ValueError("Reference parameters were not fitted on normal sequences.")
    if reference_params["fit_split"] != "train" or reference_params[
        "fit_condition"
    ] != "normal":
        raise ValueError("Reference parameter provenance is not train-normal.")
    if reference_params["estimator"] != "ledoit_wolf":
        raise ValueError("Reference parameters must use Ledoit-Wolf covariance.")
    if preprocessing_metadata["data_source"] != "synthetic" or reference_metadata[
        "data_source"
    ] != "synthetic":
        raise ValueError("Milestone 3 currently accepts only declared synthetic inputs.")

    with np.load(paths["preprocessing_arrays"]) as payload:
        observations = payload["standardized_observations"].astype(float)
        sequence_ids = payload["sequence_ids"].astype(str)
        time_indices = payload["time_indices"].astype(int)
        split_labels = payload["split_labels"].astype(str)
        observation_feature_names = payload["feature_names"].astype(str)
    with np.load(paths["reference_arrays"]) as payload:
        reference_values = payload["standardized_clean_reference"].astype(float)
        reference_sequence_ids = payload["reference_sequence_ids"].astype(str)
        reference_time_indices = payload["time_indices"].astype(int)
        reference_feature_names = payload["feature_names"].astype(str)
        stored_location = payload["location"].astype(float)
        stored_precision = payload["precision"].astype(float)

    feature_names = tuple(config["data"]["features"])
    if tuple(observation_feature_names) != feature_names:
        raise ValueError("Preprocessing feature order does not match the config.")
    if tuple(reference_feature_names) != feature_names:
        raise ValueError("Reference feature order does not match the config.")
    if tuple(reference_params["feature_names"]) != feature_names:
        raise ValueError("Reference parameter feature order does not match the config.")
    if observations.ndim != 3 or observations.shape[-1] != len(feature_names):
        raise ValueError("Standardized observations must have [sequence, time, feature].")
    if time_indices.shape != observations.shape[:2]:
        raise ValueError("Observation time indices do not match the observation shape.")
    if sequence_ids.shape != (observations.shape[0],):
        raise ValueError("Sequence IDs do not match the observation shape.")
    if split_labels.shape != (observations.shape[0],):
        raise ValueError("Split labels do not match the observation shape.")
    if set(split_labels) != {"train", "val", "test"}:
        raise ValueError("Observation inputs must preserve train, val, and test splits.")
    if reference_values.ndim != 2 or reference_values.shape[1] != len(feature_names):
        raise ValueError("Reference values must have shape [row, feature].")
    if reference_time_indices.size != reference_values.shape[0]:
        raise ValueError("Reference time indices do not match the reference rows.")
    if not np.isfinite(observations).all() or not np.isfinite(reference_values).all():
        raise ValueError("All standardized values must be finite before scoring.")

    location = np.asarray(reference_params["location"], dtype=float)
    precision = np.asarray(reference_params["precision"], dtype=float)
    if not np.array_equal(stored_location, location):
        raise ValueError("Reference location differs between NPZ and parameter JSON.")
    if not np.array_equal(stored_precision, precision):
        raise ValueError("Reference precision differs between NPZ and parameter JSON.")
    tolerance = float(config["distance"]["negative_tolerance"])
    observation_distances = squared_mahalanobis_distance(
        observations,
        location,
        precision,
        negative_tolerance=tolerance,
    )
    reference_distances = squared_mahalanobis_distance(
        reference_values,
        location,
        precision,
        negative_tolerance=tolerance,
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

    arrays_path = run_dir / "mahalanobis_distances.npz"
    metadata_path = run_dir / "metadata.json"
    config_snapshot_path = run_dir / "config_snapshot.yaml"
    np.savez_compressed(
        arrays_path,
        squared_mahalanobis_observations=observation_distances,
        squared_mahalanobis_reference=reference_distances,
        sequence_ids=np.asarray(sequence_ids, dtype=str),
        time_indices=time_indices,
        split_labels=np.asarray(split_labels, dtype=str),
        reference_sequence_ids=np.asarray(reference_sequence_ids, dtype=str),
        reference_time_indices=reference_time_indices,
        feature_names=np.asarray(feature_names, dtype=str),
    )
    with config_snapshot_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)

    metadata: dict[str, Any] = {
        "trust_scoring_run_id": actual_run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "synthetic",
        "input_preprocessing_run_id": input_config["preprocessing_run_id"],
        "input_reference_run_id": input_config["reference_run_id"],
        "config_sha256": configuration_hash,
        "distances_npz_sha256": _sha256_file(arrays_path),
        "code_version": _code_version(project_root_path),
        "feature_names": list(feature_names),
        "distance_method": "squared_mahalanobis",
        "parameter_source": config["distance"]["parameter_source"],
        "transform_only": True,
        "thresholds_computed": False,
        "trust_scores_computed": False,
        "anomaly_metrics_computed": False,
        "observation_shape": list(observations.shape),
        "observation_distance_shape": list(observation_distances.shape),
        "reference_shape": list(reference_values.shape),
        "reference_distance_shape": list(reference_distances.shape),
        "split_sequence_counts": {
            split: int(np.sum(split_labels == split))
            for split in ("train", "val", "test")
        },
        "observation_distance_summary": _summary(observation_distances),
        "reference_distance_summary": _summary(reference_distances),
        "input_sha256": expected_hashes,
        "files": {
            **{name: str(path.resolve()) for name, path in paths.items()},
            "distances_npz": str(arrays_path.resolve()),
            "config_snapshot": str(config_snapshot_path.resolve()),
            "metadata_json": str(metadata_path.resolve()),
        },
        "scientific_scope_note": (
            "Synthetic squared Mahalanobis distances using frozen clean-train-normal "
            "oracle reference parameters. No thresholds, trust groups, trust scores, "
            "anomaly metrics, forecasts, or industrial validity are computed."
        ),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute squared Mahalanobis distances from frozen M2 parameters."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--output-root", help="Optional output root override.")
    parser.add_argument("--run-id", help="Optional unique run ID; existing runs fail.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = _parse_args()
    metadata = run_trust_scoring(
        args.config,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
