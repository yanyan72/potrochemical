"""End-to-end tests for Milestone 3 squared-distance artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.generate_phase1 import run_generation
from src.run_preprocessing import run_preprocessing
from src.run_trust_reference import run_trust_reference
from src.run_trust_scoring import run_trust_scoring


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_trust_scoring_pipeline_is_identity_checked_and_reproducible(
    small_config: dict,
    tmp_path: Path,
) -> None:
    generation_config_path = tmp_path / "generation.yaml"
    with generation_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(small_config, handle, sort_keys=False)
    generated = run_generation(
        generation_config_path,
        data_root=tmp_path / "generated",
        figures_root=tmp_path / "figures",
        run_id="source_run",
        project_root=tmp_path,
    )
    simulation_csv = Path(generated["files"]["data_csv"])

    preprocessing_config = {
        "project": {"name": "test", "experiment_name": "preprocess"},
        "input": {
            "data_root": str(simulation_csv.parent.parent),
            "run_id": "source_run",
            "csv_filename": simulation_csv.name,
            "expected_csv_sha256": _sha256(simulation_csv),
        },
        "data": {"features": list(small_config["data"]["features"])},
        "preprocessing": {
            "imputation_strategy": "feature_median",
            "fit_source": "clean_train",
            "fit_split": "train",
            "std_ddof": 1,
        },
        "output": {"data_root": str(tmp_path / "preprocessed")},
    }
    preprocessing_config_path = tmp_path / "preprocessing.yaml"
    with preprocessing_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(preprocessing_config, handle, sort_keys=False)
    preprocessed = run_preprocessing(
        preprocessing_config_path,
        run_id="preprocess_run",
        project_root=tmp_path,
    )
    preprocessing_dir = Path(preprocessed["files"]["arrays_npz"]).parent

    reference_config = {
        "project": {"name": "test", "experiment_name": "trust_reference"},
        "input": {
            "simulation_data_root": str(simulation_csv.parent.parent),
            "simulation_run_id": "source_run",
            "simulation_csv_filename": simulation_csv.name,
            "simulation_csv_sha256": _sha256(simulation_csv),
            "preprocessing_data_root": str(preprocessing_dir.parent),
            "preprocessing_run_id": "preprocess_run",
            "preprocessing_arrays_filename": "standardized_observations.npz",
            "preprocessing_arrays_sha256": _sha256(
                preprocessing_dir / "standardized_observations.npz"
            ),
            "preprocessing_params_filename": "preprocessor_params.json",
            "preprocessing_params_sha256": _sha256(
                preprocessing_dir / "preprocessor_params.json"
            ),
            "preprocessing_metadata_filename": "metadata.json",
            "preprocessing_metadata_sha256": _sha256(
                preprocessing_dir / "metadata.json"
            ),
        },
        "data": {"features": list(small_config["data"]["features"])},
        "reference": {
            "fit_split": "train",
            "fit_condition": "normal",
            "fit_source": "clean",
            "standardization_params_source": "milestone1_clean_train",
            "covariance_estimator": "ledoit_wolf",
            "assume_centered": False,
        },
        "output": {"data_root": str(tmp_path / "reference")},
    }
    reference_config_path = tmp_path / "reference.yaml"
    with reference_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(reference_config, handle, sort_keys=False)
    reference = run_trust_reference(
        reference_config_path,
        run_id="reference_run",
        project_root=tmp_path,
    )
    reference_dir = Path(reference["files"]["reference_npz"]).parent

    scoring_config = {
        "project": {"name": "test", "experiment_name": "trust_scoring"},
        "input": {
            "preprocessing_data_root": str(preprocessing_dir.parent),
            "preprocessing_run_id": "preprocess_run",
            "preprocessing_arrays_filename": "standardized_observations.npz",
            "preprocessing_arrays_sha256": _sha256(
                preprocessing_dir / "standardized_observations.npz"
            ),
            "preprocessing_metadata_filename": "metadata.json",
            "preprocessing_metadata_sha256": _sha256(
                preprocessing_dir / "metadata.json"
            ),
            "reference_data_root": str(reference_dir.parent),
            "reference_run_id": "reference_run",
            "reference_arrays_filename": "reference_set.npz",
            "reference_arrays_sha256": _sha256(reference_dir / "reference_set.npz"),
            "reference_params_filename": "trust_reference_params.json",
            "reference_params_sha256": _sha256(
                reference_dir / "trust_reference_params.json"
            ),
            "reference_metadata_filename": "metadata.json",
            "reference_metadata_sha256": _sha256(reference_dir / "metadata.json"),
        },
        "data": {"features": list(small_config["data"]["features"])},
        "distance": {
            "method": "squared_mahalanobis",
            "parameter_source": "milestone2_clean_train_normal_ledoit_wolf",
            "negative_tolerance": 1e-10,
        },
        "output": {"data_root": str(tmp_path / "scoring")},
    }
    scoring_config_path = tmp_path / "scoring.yaml"
    with scoring_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(scoring_config, handle, sort_keys=False)

    metadata = run_trust_scoring(
        scoring_config_path,
        run_id="scoring_run",
        project_root=tmp_path,
    )
    run_dir = tmp_path / "scoring" / "scoring_run"
    expected_files = [
        run_dir / "mahalanobis_distances.npz",
        run_dir / "metadata.json",
        run_dir / "config_snapshot.yaml",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected_files)

    with np.load(preprocessing_dir / "standardized_observations.npz") as payload:
        observations = payload["standardized_observations"]
        expected_sequence_ids = payload["sequence_ids"].astype(str)
        expected_splits = payload["split_labels"].astype(str)
    with np.load(reference_dir / "reference_set.npz") as payload:
        reference_values = payload["standardized_clean_reference"]
        location = payload["location"]
        precision = payload["precision"]
    with np.load(run_dir / "mahalanobis_distances.npz") as payload:
        observation_distances = payload["squared_mahalanobis_observations"]
        reference_distances = payload["squared_mahalanobis_reference"]
        np.testing.assert_array_equal(payload["sequence_ids"].astype(str), expected_sequence_ids)
        np.testing.assert_array_equal(payload["split_labels"].astype(str), expected_splits)

    centered_observations = observations - location
    expected_observation_distances = np.einsum(
        "...i,ij,...j->...",
        centered_observations,
        precision,
        centered_observations,
    )
    centered_reference = reference_values - location
    expected_reference_distances = np.einsum(
        "...i,ij,...j->...",
        centered_reference,
        precision,
        centered_reference,
    )
    np.testing.assert_allclose(observation_distances, expected_observation_distances)
    np.testing.assert_allclose(reference_distances, expected_reference_distances)
    assert observation_distances.shape == observations.shape[:2]
    assert reference_distances.shape == (reference_values.shape[0],)
    assert np.isfinite(observation_distances).all()
    assert np.isfinite(reference_distances).all()
    assert (observation_distances >= 0.0).all()
    assert (reference_distances >= 0.0).all()
    assert metadata["transform_only"] is True
    assert metadata["thresholds_computed"] is False
    assert metadata["trust_scores_computed"] is False
    assert metadata["anomaly_metrics_computed"] is False

    with pytest.raises(FileExistsError):
        run_trust_scoring(
            scoring_config_path,
            run_id="scoring_run",
            project_root=tmp_path,
        )

    reference_metadata_path = reference_dir / "metadata.json"
    tampered_metadata = json.loads(reference_metadata_path.read_text(encoding="utf-8"))
    tampered_metadata["input_preprocessing_run_id"] = "different_preprocess_run"
    reference_metadata_path.write_text(
        json.dumps(tampered_metadata),
        encoding="utf-8",
    )
    scoring_config["input"]["reference_metadata_sha256"] = _sha256(
        reference_metadata_path
    )
    with scoring_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(scoring_config, handle, sort_keys=False)
    with pytest.raises(ValueError, match="preprocessing runs differ"):
        run_trust_scoring(
            scoring_config_path,
            run_id="identity_failure",
            project_root=tmp_path,
        )
