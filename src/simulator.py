"""Reproducible multivariate petrochemical time-series simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


FEATURE_NAMES = (
    "temperature",
    "pressure",
    "vibration",
    "concentration",
)
FEATURE_UNITS = {
    "temperature": "K",
    "pressure": "MPa",
    "vibration": "mm/s",
    "concentration": "fraction",
    "quality": "dimensionless_fraction",
    "time_index": "abstract_sampling_step",
}
CONDITIONS = ("normal", "high_temperature", "high_vibration", "compound")


@dataclass(frozen=True)
class SimulationResult:
    """Clean simulator output in long-table and dense-array forms."""

    frame: pd.DataFrame
    clean_values: np.ndarray
    quality: np.ndarray
    sequence_ids: tuple[str, ...]
    feature_names: tuple[str, ...] = FEATURE_NAMES


def _condition_probabilities(simulation: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            simulation["normal_ratio"],
            simulation["temperature_abnormal_ratio"],
            simulation["vibration_abnormal_ratio"],
            simulation["compound_abnormal_ratio"],
        ],
        dtype=float,
    )


def _ar1_process(
    rng: np.random.Generator,
    length: int,
    rho: float,
) -> np.ndarray:
    innovations = rng.normal(0.0, np.sqrt(1.0 - rho**2), size=length)
    values = np.empty(length, dtype=float)
    values[0] = innovations[0]
    for index in range(1, length):
        values[index] = rho * values[index - 1] + innovations[index]
    return values


def _simulate_one_sequence(
    rng: np.random.Generator,
    length: int,
    delta_t: float,
    condition: str,
    simulation: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(length, dtype=float)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    latent = _ar1_process(rng, length, rho=float(simulation["ar_rho"]))
    slow = np.sin(2.0 * np.pi * time / float(simulation["cycle_length"]) + phase)

    temperature_offset = 0.0
    pressure_offset = 0.0
    vibration_offset = 0.0
    concentration_offset = 0.0
    if condition == "high_temperature":
        temperature_offset = float(simulation["temperature_condition_offset"])
        pressure_offset = 0.04
        concentration_offset = -0.01
    elif condition == "high_vibration":
        vibration_offset = float(simulation["vibration_condition_offset"])
        pressure_offset = 0.02
    elif condition == "compound":
        temperature_offset = 0.75 * float(
            simulation["temperature_condition_offset"]
        )
        pressure_offset = 0.09
        vibration_offset = 0.75 * float(
            simulation["vibration_condition_offset"]
        )
        concentration_offset = -0.025

    process_noise = float(simulation["process_noise_scale"])
    temperature = (
        315.0
        + temperature_offset
        + 1.8 * latent
        + 0.9 * slow
        + rng.normal(0.0, 0.25 * process_noise, length)
    )
    pressure = (
        2.20
        + pressure_offset
        + 0.055 * latent
        + 0.025 * slow
        + rng.normal(0.0, 0.008 * process_noise, length)
    )
    vibration = (
        0.40
        + vibration_offset
        + 0.055 * np.abs(latent)
        + 0.025 * slow
        + rng.normal(0.0, 0.012 * process_noise, length)
    )
    concentration = (
        0.85
        + concentration_offset
        - 0.010 * latent
        - 0.006 * slow
        + rng.normal(0.0, 0.0025 * process_noise, length)
    )

    temperature = np.clip(temperature, 300.0, 345.0)
    pressure = np.clip(pressure, 1.5, 3.0)
    vibration = np.clip(vibration, 0.05, 2.0)
    concentration = np.clip(concentration, 0.60, 0.98)
    features = np.column_stack(
        (temperature, pressure, vibration, concentration)
    )

    quality = np.empty(length, dtype=float)
    quality[0] = rng.uniform(0.975, 0.995)
    quality_noise_std = float(simulation["quality_noise_std"])
    base_rate = float(simulation["base_degradation_rate"])
    for index in range(length - 1):
        rate = (
            base_rate
            + 1.5e-5 * max(temperature[index] - 315.0, 0.0)
            + 2.5e-4 * max(pressure[index] - 2.20, 0.0)
            + 3.5e-4 * max(vibration[index] - 0.40, 0.0)
            + 1.5e-3 * max(0.85 - concentration[index], 0.0)
        )
        innovation = rng.normal(0.0, quality_noise_std)
        quality[index + 1] = np.clip(
            quality[index] - rate * delta_t + innovation,
            0.0,
            1.0,
        )
    return features, quality


def generate_clean_sequences(config: dict[str, Any]) -> SimulationResult:
    """Generate clean sequences using only the supplied seed and configuration."""

    project = config["project"]
    data = config["data"]
    simulation = config["simulation"]
    rng = np.random.default_rng(int(project["seed"]))

    split_counts = (
        ("train", int(data["num_train_sequences"])),
        ("val", int(data["num_val_sequences"])),
        ("test", int(data["num_test_sequences"])),
    )
    sequence_length = int(data["sequence_length"])
    delta_t = float(data["delta_t"])
    total_sequences = sum(count for _, count in split_counts)
    conditions = rng.choice(
        CONDITIONS,
        size=total_sequences,
        p=_condition_probabilities(simulation),
    )

    clean_values = np.empty(
        (total_sequences, sequence_length, len(FEATURE_NAMES)),
        dtype=float,
    )
    quality = np.empty((total_sequences, sequence_length), dtype=float)
    sequence_ids: list[str] = []
    split_labels: list[str] = []
    sequence_index = 0
    for split, count in split_counts:
        for split_index in range(count):
            sequence_id = f"{split}_{split_index:04d}"
            sequence_ids.append(sequence_id)
            split_labels.append(split)
            sequence_features, sequence_quality = _simulate_one_sequence(
                rng,
                sequence_length,
                delta_t,
                str(conditions[sequence_index]),
                simulation,
            )
            clean_values[sequence_index] = sequence_features
            quality[sequence_index] = sequence_quality
            sequence_index += 1

    frame = pd.DataFrame(
        {
            "sequence_id": np.repeat(sequence_ids, sequence_length),
            "time_index": np.tile(np.arange(sequence_length), total_sequences),
            "split": np.repeat(split_labels, sequence_length),
            "condition": np.repeat(conditions, sequence_length),
            "quality": quality.reshape(-1),
        }
    )
    for feature_index, feature_name in enumerate(FEATURE_NAMES):
        frame[f"{feature_name}_clean"] = clean_values[:, :, feature_index].reshape(
            -1
        )

    return SimulationResult(
        frame=frame,
        clean_values=clean_values,
        quality=quality,
        sequence_ids=tuple(sequence_ids),
    )
