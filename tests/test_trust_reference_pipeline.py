"""End-to-end tests for Milestone 2 trust-reference artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.covariance import LedoitWolf

from src.generate_phase1 import run_generation
from src.run_preprocessing import run_preprocessing
from src.run_trust_reference import run_trust_reference


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_trust_reference_pipeline_selects_only_clean_train_normal(
    small_config: dict,
    tmp_path: Path,
) -> None:
    generation_config = tmp_path / "generation.yaml"
    with generation_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(small_config, handle, sort_keys=False)
    generated = run_generation(
        generation_config,
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
    preprocessed_dir = Path(preprocessed["files"]["arrays_npz"]).parent

    trust_config = {
        "project": {"name": "test", "experiment_name": "trust_reference"},
        "input": {
            "simulation_data_root": str(simulation_csv.parent.parent),
            "simulation_run_id": "source_run",
            "simulation_csv_filename": simulation_csv.name,
            "simulation_csv_sha256": _sha256(simulation_csv),
            "preprocessing_data_root": str(preprocessed_dir.parent),
            "preprocessing_run_id": "preprocess_run",
            "preprocessing_arrays_filename": "standardized_observations.npz",
            "preprocessing_arrays_sha256": _sha256(
                preprocessed_dir / "standardized_observations.npz"
            ),
            "preprocessing_params_filename": "preprocessor_params.json",
            "preprocessing_params_sha256": _sha256(
                preprocessed_dir / "preprocessor_params.json"
            ),
            "preprocessing_metadata_filename": "metadata.json",
            "preprocessing_metadata_sha256": _sha256(
                preprocessed_dir / "metadata.json"
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
        "output": {"data_root": str(tmp_path / "trust")},
    }
    trust_config_path = tmp_path / "trust.yaml"
    with trust_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(trust_config, handle, sort_keys=False)

    metadata = run_trust_reference(
        trust_config_path,
        run_id="trust_run",
        project_root=tmp_path,
    )
    run_dir = tmp_path / "trust" / "trust_run"
    expected_files = [
        run_dir / "reference_set.npz",
        run_dir / "trust_reference_params.json",
        run_dir / "metadata.json",
        run_dir / "config_snapshot.yaml",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected_files)

    frame = pd.read_csv(simulation_csv)
    sequence_info = frame[["sequence_id", "split", "condition"]].drop_duplicates()
    expected_ids = sequence_info.loc[
        (sequence_info["split"] == "train")
        & (sequence_info["condition"] == "normal"),
        "sequence_id",
    ].astype(str).to_numpy()
    assert len(expected_ids) > 0
    with np.load(run_dir / "reference_set.npz") as payload:
        reference = payload["standardized_clean_reference"]
        assert payload["reference_sequence_ids"].dtype.kind == "U"
        saved_ids = payload["reference_sequence_ids"].astype(str)
        covariance = payload["covariance"]
        precision = payload["precision"]
    np.testing.assert_array_equal(saved_ids, expected_ids)
    assert reference.shape == (len(expected_ids) * 64, 4)
    expected_estimator = LedoitWolf().fit(reference)
    np.testing.assert_allclose(covariance, expected_estimator.covariance_)
    np.testing.assert_allclose(precision, expected_estimator.precision_)
    assert metadata["num_reference_sequences"] == len(expected_ids)
    assert metadata["num_reference_rows"] == reference.shape[0]
    assert metadata["fit_split"] == "train"
    assert metadata["fit_condition"] == "normal"
    assert metadata["fit_source"] == "standardized_clean_train_normal"
    assert metadata["diagnostics"]["covariance_min_eigenvalue"] > 0.0

    with pytest.raises(FileExistsError):
        run_trust_reference(
            trust_config_path,
            run_id="trust_run",
            project_root=tmp_path,
        )
