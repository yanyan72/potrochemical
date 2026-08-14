"""Shared test fixtures for Milestone 0."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest


@pytest.fixture
def small_config() -> dict[str, Any]:
    """Return a compact valid configuration for fast deterministic tests."""

    config: dict[str, Any] = {
        "project": {"name": "test_project", "seed": 123, "experiment_name": "test"},
        "data": {
            "num_train_sequences": 6,
            "num_val_sequences": 3,
            "num_test_sequences": 3,
            "sequence_length": 64,
            "delta_t": 1.0,
            "features": [
                "temperature",
                "pressure",
                "vibration",
                "concentration",
            ],
        },
        "simulation": {
            "normal_ratio": 0.40,
            "temperature_abnormal_ratio": 0.25,
            "vibration_abnormal_ratio": 0.15,
            "compound_abnormal_ratio": 0.20,
            "ar_rho": 0.90,
            "cycle_length": 32,
            "process_noise_scale": 1.0,
            "quality_noise_std": 0.0001,
            "base_degradation_rate": 0.00012,
            "temperature_condition_offset": 10.0,
            "vibration_condition_offset": 0.35,
        },
        "corruption": {
            "enabled": True,
            "ratio": 0.125,
            "seed_offset": 1000,
            "types": [
                "spike",
                "bias",
                "drift",
                "missing",
                "random_replacement",
            ],
            "replacement_strategy": "empirical_train_value",
            "min_segment_length": 4,
            "max_segment_length": 9,
            "magnitude_std": {
                "spike": [4.0, 6.0],
                "bias": [2.0, 3.0],
                "drift": [3.0, 4.0],
            },
        },
        "output": {"data_root": "data/processed", "figures_root": "results/figures"},
    }
    return deepcopy(config)
