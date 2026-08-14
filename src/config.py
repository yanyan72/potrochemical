"""Configuration loading and validation helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_FEATURES = (
    "temperature",
    "pressure",
    "vibration",
    "concentration",
)
SUPPORTED_CORRUPTIONS = (
    "spike",
    "bias",
    "drift",
    "missing",
    "random_replacement",
)
MAGNITUDE_CORRUPTIONS = ("spike", "bias", "drift")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and validate Milestone 0 fields."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    validate_config(config)
    return config


def _positive_int(mapping: dict[str, Any], key: str, *, minimum: int = 1) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"'{key}' must be an integer >= {minimum}.")
    return value


def validate_config(config: dict[str, Any]) -> None:
    """Validate fields required by the first simulator milestone."""

    for section in ("project", "data", "simulation", "corruption"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing mapping section: '{section}'.")

    project = config["project"]
    seed = project.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("'project.seed' must be a non-negative integer.")

    data = config["data"]
    for key in ("num_train_sequences", "num_val_sequences", "num_test_sequences"):
        _positive_int(data, key)
    _positive_int(data, "sequence_length", minimum=8)
    delta_t = data.get("delta_t")
    if not isinstance(delta_t, (int, float)) or delta_t <= 0:
        raise ValueError("'data.delta_t' must be positive.")
    features = tuple(data.get("features", ()))
    if features != SUPPORTED_FEATURES:
        raise ValueError(
            "Milestone 0 requires features in this order: "
            f"{list(SUPPORTED_FEATURES)}."
        )

    simulation = config["simulation"]
    ratio_keys = (
        "normal_ratio",
        "temperature_abnormal_ratio",
        "vibration_abnormal_ratio",
        "compound_abnormal_ratio",
    )
    ratios = [simulation.get(key) for key in ratio_keys]
    if any(not isinstance(value, (int, float)) or value < 0 for value in ratios):
        raise ValueError("All condition ratios must be non-negative numbers.")
    if abs(sum(float(value) for value in ratios) - 1.0) > 1e-9:
        raise ValueError("Simulation condition ratios must sum to 1.0.")

    corruption = config["corruption"]
    ratio = corruption.get("ratio")
    if not isinstance(ratio, (int, float)) or not 0 <= ratio < 1:
        raise ValueError("'corruption.ratio' must be in [0, 1).")
    types = corruption.get("types")
    if not isinstance(types, list) or not types:
        raise ValueError("'corruption.types' must be a non-empty list.")
    unsupported = set(types) - set(SUPPORTED_CORRUPTIONS)
    if unsupported:
        raise ValueError(
            "Milestone 0 supports spike, bias, drift, missing, and "
            "random_replacement; unsupported: "
            f"{sorted(unsupported)}"
        )
    if len(types) != len(set(types)):
        raise ValueError("'corruption.types' must not contain duplicates.")
    _positive_int(corruption, "min_segment_length", minimum=2)
    _positive_int(corruption, "max_segment_length", minimum=2)
    if corruption["min_segment_length"] > corruption["max_segment_length"]:
        raise ValueError("min_segment_length cannot exceed max_segment_length.")

    magnitude_std = corruption.get("magnitude_std")
    if not isinstance(magnitude_std, dict):
        raise ValueError("'corruption.magnitude_std' must be a mapping.")
    for corruption_type in MAGNITUDE_CORRUPTIONS:
        bounds = magnitude_std.get(corruption_type)
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or not all(isinstance(value, (int, float)) for value in bounds)
            or bounds[0] <= 0
            or bounds[1] < bounds[0]
        ):
            raise ValueError(
                f"magnitude_std.{corruption_type} must be [positive_min, max]."
            )
    if corruption.get("replacement_strategy") != "empirical_train_value":
        raise ValueError(
            "'corruption.replacement_strategy' must be "
            "'empirical_train_value'."
        )


def config_hash(config: dict[str, Any]) -> str:
    """Return a stable SHA-256 hash of a configuration mapping."""

    payload = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
