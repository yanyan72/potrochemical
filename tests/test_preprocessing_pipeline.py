"""End-to-end tests for Milestone 1 preprocessing artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.generate_phase1 import run_generation
from src.run_preprocessing import run_preprocessing


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preprocessing_pipeline_is_leakage_safe_and_non_overwriting(
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
    input_csv = Path(generated["files"]["data_csv"])
    preprocessing_config = {
        "project": {"name": "test_project", "experiment_name": "preprocess_test"},
        "input": {
            "data_root": str(input_csv.parent.parent),
            "run_id": "source_run",
            "csv_filename": input_csv.name,
            "expected_csv_sha256": _sha256(input_csv),
        },
        "data": {
            "features": [
                "temperature",
                "pressure",
                "vibration",
                "concentration",
            ]
        },
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

    metadata = run_preprocessing(
        preprocessing_config_path,
        run_id="preprocess_test",
        project_root=tmp_path,
    )
    run_dir = tmp_path / "preprocessed" / "preprocess_test"
    expected_files = [
        run_dir / "standardized_observations.npz",
        run_dir / "preprocessor_params.json",
        run_dir / "metadata.json",
        run_dir / "config_snapshot.yaml",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected_files)

    frame = pd.read_csv(input_csv)
    feature_names = preprocessing_config["data"]["features"]
    train_clean = frame.loc[
        frame["split"] == "train",
        [f"{name}_clean" for name in feature_names],
    ].to_numpy(dtype=float)
    with (run_dir / "preprocessor_params.json").open(encoding="utf-8") as handle:
        params = json.load(handle)
    np.testing.assert_allclose(params["imputation_values"], np.median(train_clean, axis=0))
    np.testing.assert_allclose(params["means"], np.mean(train_clean, axis=0))
    np.testing.assert_allclose(params["scales"], np.std(train_clean, axis=0, ddof=1))

    with np.load(run_dir / "standardized_observations.npz") as payload:
        standardized = payload["standardized_observations"]
        imputed = payload["imputed_observations"]
        missing_mask = payload["original_missing_mask"]
        sequence_ids = payload["sequence_ids"]
        time_indices = payload["time_indices"]
        splits = payload["split_labels"]
    assert standardized.shape == (12, 64, 4)
    assert imputed.shape == standardized.shape
    assert missing_mask.shape == standardized.shape
    assert np.isfinite(standardized).all()
    assert np.isfinite(imputed).all()
    assert len(set(sequence_ids.tolist())) == 12
    assert np.array_equal(time_indices, np.tile(np.arange(64), (12, 1)))
    assert {split: int(np.sum(splits == split)) for split in set(splits)} == {
        "train": 6,
        "val": 3,
        "test": 3,
    }
    assert int(missing_mask.sum()) == metadata["missing_values_before"]
    assert metadata["missing_values_after_imputation"] == 0
    assert metadata["non_finite_values_after_standardization"] == 0
    assert metadata["fit_source"] == "clean_train"
    assert metadata["fit_split"] == "train"
    assert metadata["data_source"] == "synthetic"

    with pytest.raises(FileExistsError):
        run_preprocessing(
            preprocessing_config_path,
            run_id="preprocess_test",
            project_root=tmp_path,
        )
