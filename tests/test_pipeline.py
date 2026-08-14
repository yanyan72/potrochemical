"""End-to-end tests for Milestone 0 artifact generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.generate_phase1 import run_generation


def test_pipeline_writes_consistent_non_overwriting_artifacts(
    small_config: dict,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(small_config, handle, sort_keys=False)

    data_root = tmp_path / "data"
    figures_root = tmp_path / "figures"
    metadata = run_generation(
        config_path,
        data_root=data_root,
        figures_root=figures_root,
        run_id="test_run",
        project_root=tmp_path,
    )

    data_dir = data_root / "test_run"
    figure_dir = figures_root / "test_run"
    expected_files = [
        data_dir / "simulated_timeseries.csv",
        data_dir / "corruption_mask.npz",
        data_dir / "corruption_events.json",
        data_dir / "metadata.json",
        data_dir / "config_snapshot.yaml",
        figure_dir / "corruption_examples.png",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected_files)

    frame = pd.read_csv(data_dir / "simulated_timeseries.csv")
    with np.load(data_dir / "corruption_mask.npz") as payload:
        mask = payload["mask"]
    with (data_dir / "metadata.json").open(encoding="utf-8") as handle:
        saved_metadata = json.load(handle)
    with (data_dir / "corruption_events.json").open(encoding="utf-8") as handle:
        events = json.load(handle)

    assert len(frame) == mask.shape[0] * mask.shape[1]
    missing_rows = frame["corruption_type"] == "missing"
    assert (
        frame.loc[
            :,
            [
                "temperature_clean",
                "pressure_clean",
                "vibration_clean",
                "concentration_clean",
                "quality",
            ],
        ]
        .notna()
        .all()
        .all()
    )
    for feature_name in (
        "temperature",
        "pressure",
        "vibration",
        "concentration",
    ):
        expected_missing = missing_rows & (
            frame["corruption_feature"] == feature_name
        )
        np.testing.assert_array_equal(
            frame[f"{feature_name}_obs"].isna().to_numpy(),
            expected_missing.to_numpy(),
        )
    np.testing.assert_array_equal(
        frame["corruption_magnitude"].isna().to_numpy(),
        missing_rows.to_numpy(),
    )
    assert set(frame.loc[~frame["is_corrupted"], "corruption_type"]) == {"none"}
    assert set(frame.loc[~frame["is_corrupted"], "corruption_feature"]) == {"none"}
    np.testing.assert_array_equal(
        frame["is_corrupted"].to_numpy(dtype=bool),
        mask.any(axis=2).reshape(-1),
    )
    assert saved_metadata["data_source"] == "synthetic"
    assert saved_metadata["corruption_reference_split"] == "clean_train_only"
    assert saved_metadata["code_version"] == "unavailable_no_git_repository"
    assert saved_metadata["data_csv_sha256"] == metadata["data_csv_sha256"]
    assert saved_metadata["corrupted_rows"] == int(mask.any(axis=2).sum())
    assert {event["corruption_type"] for event in events} == {
        "spike",
        "bias",
        "drift",
        "missing",
        "random_replacement",
    }
    for event in events:
        if event["corruption_type"] != "random_replacement":
            continue
        observed_row = frame.loc[
            (frame["sequence_id"] == event["sequence_id"])
            & (frame["time_index"] == event["start_time_index"])
        ].iloc[0]
        assert np.isclose(
            observed_row[f"{event['corruption_feature']}_obs"],
            event["replacement_value"],
            rtol=0.0,
            atol=np.finfo(float).eps,
        )
    assert frame.groupby("sequence_id")["split"].nunique().max() == 1

    with pytest.raises(FileExistsError):
        run_generation(
            config_path,
            data_root=data_root,
            figures_root=figures_root,
            run_id="test_run",
            project_root=tmp_path,
        )
