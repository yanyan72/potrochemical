"""End-to-end test for E1 calibration, pointwise output, tables, and plot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.run_e1_evaluation import run_e1_evaluation


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_e1_pipeline_outputs_requested_fields_and_holds_out_test(tmp_path: Path) -> None:
    simulation_dir = tmp_path / "simulation" / "source_run"
    scoring_dir = tmp_path / "scoring" / "scoring_run"
    simulation_dir.mkdir(parents=True)
    scoring_dir.mkdir(parents=True)

    sequence_ids = np.asarray(["train_a", "val_a", "test_a"])
    split_labels = np.asarray(["train", "val", "test"])
    time_indices = np.tile(np.arange(4), (3, 1))
    rows: list[dict[str, object]] = []
    for sequence_id, split, condition in zip(
        sequence_ids,
        split_labels,
        ("normal", "hot", "normal"),
        strict=True,
    ):
        for time_index, is_corrupted, corruption_type in zip(
            range(4),
            (False, True, False, True),
            ("none", "spike", "none", "drift"),
            strict=True,
        ):
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "time_index": time_index,
                    "split": split,
                    "condition": condition,
                    "is_corrupted": is_corrupted,
                    "corruption_type": corruption_type,
                }
            )
    simulation_csv = simulation_dir / "simulated_timeseries.csv"
    pd.DataFrame(rows).to_csv(simulation_csv, index=False)

    observation_squared = np.asarray(
        [
            [0.1, 8.0, 12.0, 20.0],
            [0.2, 9.5, 15.0, 22.0],
            [0.3, 10.0, 18.0, 25.0],
        ]
    )
    reference_squared = np.arange(1.0, 11.0)
    scoring_arrays = scoring_dir / "mahalanobis_distances.npz"
    np.savez_compressed(
        scoring_arrays,
        squared_mahalanobis_observations=observation_squared,
        squared_mahalanobis_reference=reference_squared,
        sequence_ids=sequence_ids,
        time_indices=time_indices,
        split_labels=split_labels,
    )
    scoring_metadata_path = scoring_dir / "metadata.json"
    scoring_metadata = {
        "trust_scoring_run_id": "scoring_run",
        "distances_npz_sha256": _sha256(scoring_arrays),
        "data_source": "synthetic",
        "thresholds_computed": False,
    }
    scoring_metadata_path.write_text(
        json.dumps(scoring_metadata),
        encoding="utf-8",
    )

    config = {
        "project": {"name": "test", "experiment_name": "e1"},
        "input": {
            "simulation_data_root": str(simulation_dir.parent),
            "simulation_run_id": "source_run",
            "simulation_csv_filename": simulation_csv.name,
            "simulation_csv_sha256": _sha256(simulation_csv),
            "trust_scoring_data_root": str(scoring_dir.parent),
            "trust_scoring_run_id": "scoring_run",
            "trust_scoring_arrays_filename": scoring_arrays.name,
            "trust_scoring_arrays_sha256": _sha256(scoring_arrays),
            "trust_scoring_metadata_filename": scoring_metadata_path.name,
            "trust_scoring_metadata_sha256": _sha256(scoring_metadata_path),
        },
        "calibration": {
            "source": "normal_train_clean_reference_distances",
            "high_quantile": 0.90,
            "low_quantile": 0.99,
            "quantile_method": "linear",
        },
        "trust_mapping": {
            "method": "exponential",
            "formula": "exp(-d2/(2*tau))",
            "tau_source": "q90",
        },
        "evaluation": {
            "evaluation_splits": ["train", "val"],
            "holdout_split": "test",
            "threshold_names": ["q90", "q99"],
            "positive_label": "is_corrupted",
            "anomaly_score": "squared_mahalanobis_distance",
        },
        "plot": {
            "split": "val",
            "sequence_selection": "most_corrupted_then_sequence_id",
        },
        "output": {
            "data_root": str(tmp_path / "output_data"),
            "figures_root": str(tmp_path / "output_figures"),
            "tables_root": str(tmp_path / "output_tables"),
        },
    }
    config_path = tmp_path / "e1.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    metadata = run_e1_evaluation(config_path, run_id="e1_run", project_root=tmp_path)
    data_dir = tmp_path / "output_data" / "e1_run"
    figure_dir = tmp_path / "output_figures" / "e1_run"
    table_dir = tmp_path / "output_tables" / "e1_run"
    expected_files = [
        data_dir / "pointwise_trust_scores.csv",
        data_dir / "trust_calibration.json",
        data_dir / "metadata.json",
        data_dir / "config_snapshot.yaml",
        figure_dir / "distance_trust_timeseries.png",
        table_dir / "detection_metrics.csv",
        table_dir / "corruption_type_recall.csv",
        table_dir / "condition_false_positive_rates.csv",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected_files)

    pointwise = pd.read_csv(data_dir / "pointwise_trust_scores.csv")
    required_columns = {
        "sequence_id",
        "time_index",
        "corruption_label",
        "corruption_type",
        "mahalanobis_distance",
        "trust_score",
        "mahalanobis_distance_squared",
        "trust_group",
    }
    assert required_columns.issubset(pointwise.columns)
    assert len(pointwise) == 12
    np.testing.assert_allclose(
        pointwise["mahalanobis_distance"].to_numpy(),
        np.sqrt(observation_squared.reshape(-1)),
    )
    assert pointwise["trust_score"].between(0.0, 1.0).all()
    assert set(pointwise["trust_group"]) == {"high", "uncertain", "low"}
    test_rows = pointwise.loc[pointwise["split"] == "test"]
    assert len(test_rows) == 4
    assert test_rows["corruption_label"].sum() == 2

    calibration = json.loads(
        (data_dir / "trust_calibration.json").read_text(encoding="utf-8")
    )
    expected_thresholds = np.quantile(reference_squared, [0.90, 0.99], method="linear")
    assert calibration["high_threshold_squared"] == pytest.approx(expected_thresholds[0])
    assert calibration["low_threshold_squared"] == pytest.approx(expected_thresholds[1])
    metrics = pd.read_csv(table_dir / "detection_metrics.csv")
    assert set(metrics["split"]) == {"train", "val"}
    assert "test" not in set(metrics["split"])
    assert set(metrics["threshold_name"]) == {"q90", "q99"}
    assert metadata["holdout_metrics_computed"] is False
    assert metadata["plot_sequence_id"] == "val_a"

    with pytest.raises(FileExistsError):
        run_e1_evaluation(config_path, run_id="e1_run", project_root=tmp_path)
