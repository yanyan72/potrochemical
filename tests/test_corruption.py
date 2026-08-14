"""Tests for artificial corruption masks and metadata."""

from __future__ import annotations

import numpy as np

from src.corruption import inject_corruptions
from src.simulator import generate_clean_sequences


def test_corruption_mask_metadata_and_exact_ratio(small_config: dict) -> None:
    simulation = generate_clean_sequences(small_config)
    original = simulation.clean_values.copy()
    corruption = inject_corruptions(
        simulation.clean_values,
        simulation.clean_values[:6],
        simulation.sequence_ids,
        simulation.feature_names,
        small_config["corruption"],
        seed=small_config["project"]["seed"]
        + small_config["corruption"]["seed_offset"],
    )

    np.testing.assert_array_equal(simulation.clean_values, original)
    assert corruption.mask.shape == original.shape
    assert corruption.mask.dtype == np.bool_
    np.testing.assert_array_equal(
        corruption.mask,
        corruption.observed_values != original,
    )
    row_mask = corruption.mask.any(axis=2)
    expected_rows = round(row_mask.size * small_config["corruption"]["ratio"])
    assert int(row_mask.sum()) == expected_rows
    assert int(corruption.mask.sum()) == expected_rows
    assert (corruption.mask.sum(axis=2) <= 1).all()
    assert {event["corruption_type"] for event in corruption.events} == {
        "spike",
        "bias",
        "drift",
        "missing",
        "random_replacement",
    }

    reconstructed = np.zeros_like(corruption.mask)
    missing_reconstructed = np.zeros_like(corruption.mask)
    for event in corruption.events:
        sequence_index = event["sequence_index"]
        feature_index = event["feature_index"]
        start = event["start_time_index"]
        end = event["end_time_index_exclusive"]
        reconstructed[sequence_index, start:end, feature_index] = True
        assert event["duration"] == end - start
        assert event["magnitude_unit"] in {"K", "MPa", "mm/s", "fraction"}
        if event["corruption_type"] == "spike":
            assert event["duration"] == 1
        if event["corruption_type"] == "missing":
            missing_reconstructed[
                sequence_index, start:end, feature_index
            ] = True
            assert event["missing_value_encoding"] == "NaN"
            assert event["magnitude_start"] is None
            assert np.isnan(
                corruption.observed_values[
                    sequence_index, start:end, feature_index
                ]
            ).all()
        if event["corruption_type"] == "random_replacement":
            assert event["duration"] == 1
            assert event["replacement_source"] == "clean_train_empirical"
            assert np.isfinite(event["replacement_value"])
            assert event["replacement_value"] in simulation.clean_values[
                :6, :, feature_index
            ]
    missing_mask = np.isnan(corruption.observed_values)
    assert np.array_equal(missing_mask, missing_reconstructed)
    assert np.isfinite(corruption.observed_values[~missing_mask]).all()
    np.testing.assert_array_equal(reconstructed, corruption.mask)


def test_corruption_is_reproducible(small_config: dict) -> None:
    simulation = generate_clean_sequences(small_config)
    seed = 9876
    first = inject_corruptions(
        simulation.clean_values,
        simulation.clean_values[:6],
        simulation.sequence_ids,
        simulation.feature_names,
        small_config["corruption"],
        seed=seed,
    )
    second = inject_corruptions(
        simulation.clean_values,
        simulation.clean_values[:6],
        simulation.sequence_ids,
        simulation.feature_names,
        small_config["corruption"],
        seed=seed,
    )
    np.testing.assert_allclose(
        first.observed_values,
        second.observed_values,
        equal_nan=True,
    )
    np.testing.assert_array_equal(first.mask, second.mask)
    assert first.events == second.events


def test_missing_and_random_replacement_semantics(small_config: dict) -> None:
    simulation = generate_clean_sequences(small_config)
    training_reference = simulation.clean_values[:6]
    corruption = inject_corruptions(
        simulation.clean_values,
        training_reference,
        simulation.sequence_ids,
        simulation.feature_names,
        small_config["corruption"],
        seed=2468,
    )

    missing_events = [
        event
        for event in corruption.events
        if event["corruption_type"] == "missing"
    ]
    replacement_events = [
        event
        for event in corruption.events
        if event["corruption_type"] == "random_replacement"
    ]
    assert missing_events
    assert replacement_events

    for event in missing_events:
        observed = corruption.observed_values[
            event["sequence_index"],
            event["start_time_index"] : event["end_time_index_exclusive"],
            event["feature_index"],
        ]
        assert np.isnan(observed).all()
        assert event["magnitude_start"] is None
        assert event["missing_value_encoding"] == "NaN"

    for event in replacement_events:
        sequence_index = event["sequence_index"]
        time_index = event["start_time_index"]
        feature_index = event["feature_index"]
        observed = corruption.observed_values[
            sequence_index, time_index, feature_index
        ]
        clean = simulation.clean_values[sequence_index, time_index, feature_index]
        assert np.isfinite(observed)
        assert observed != clean
        assert observed in training_reference[:, :, feature_index]
        assert event["replacement_source"] == "clean_train_empirical"
