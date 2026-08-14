"""Tests for clean time-series simulation."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from src.config import validate_config
from src.simulator import FEATURE_NAMES, generate_clean_sequences


def test_simulator_shape_ranges_and_split_integrity(small_config: dict) -> None:
    validate_config(small_config)
    result = generate_clean_sequences(small_config)
    n_sequences = 12
    sequence_length = 64

    assert result.clean_values.shape == (n_sequences, sequence_length, 4)
    assert result.quality.shape == (n_sequences, sequence_length)
    assert len(result.frame) == n_sequences * sequence_length
    assert np.isfinite(result.clean_values).all()
    assert np.isfinite(result.quality).all()
    assert ((result.quality >= 0.0) & (result.quality <= 1.0)).all()
    assert ((result.clean_values[:, :, 0] >= 300.0) & (result.clean_values[:, :, 0] <= 345.0)).all()
    assert ((result.clean_values[:, :, 1] >= 1.5) & (result.clean_values[:, :, 1] <= 3.0)).all()
    assert ((result.clean_values[:, :, 2] >= 0.05) & (result.clean_values[:, :, 2] <= 2.0)).all()
    assert ((result.clean_values[:, :, 3] >= 0.60) & (result.clean_values[:, :, 3] <= 0.98)).all()

    split_counts = result.frame.groupby("split")["sequence_id"].nunique().to_dict()
    assert split_counts == {"test": 3, "train": 6, "val": 3}
    assert result.frame.groupby("sequence_id")["split"].nunique().max() == 1
    assert result.frame.groupby("sequence_id")["condition"].nunique().max() == 1
    assert tuple(result.feature_names) == FEATURE_NAMES


def test_simulator_is_reproducible_and_seed_sensitive(small_config: dict) -> None:
    first = generate_clean_sequences(small_config)
    second = generate_clean_sequences(small_config)
    np.testing.assert_array_equal(first.clean_values, second.clean_values)
    np.testing.assert_array_equal(first.quality, second.quality)
    pd.testing.assert_frame_equal(first.frame, second.frame)

    changed_config = deepcopy(small_config)
    changed_config["project"]["seed"] += 1
    changed = generate_clean_sequences(changed_config)
    assert not np.array_equal(first.clean_values, changed.clean_values)
