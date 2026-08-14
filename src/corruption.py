"""Artificial sensor corruption with exact masks and event metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


FEATURE_UNITS = {
    "temperature": "K",
    "pressure": "MPa",
    "vibration": "mm/s",
    "concentration": "fraction",
}


@dataclass(frozen=True)
class CorruptionResult:
    """Polluted observations, feature-level mask, and event annotations."""

    observed_values: np.ndarray
    mask: np.ndarray
    events: tuple[dict[str, Any], ...]


def _balanced_budgets(total: int, types: list[str]) -> dict[str, int]:
    base, remainder = divmod(total, len(types))
    return {
        corruption_type: base + int(index < remainder)
        for index, corruption_type in enumerate(types)
    }


def _choose_duration(
    rng: np.random.Generator,
    remaining: int,
    minimum: int,
    maximum: int,
) -> int:
    upper = min(remaining, maximum)
    if upper <= minimum:
        return upper
    duration = int(rng.integers(minimum, upper + 1))
    leftover = remaining - duration
    if 0 < leftover < minimum:
        duration += leftover
    return duration


def _find_open_segment(
    rng: np.random.Generator,
    occupied: np.ndarray,
    duration: int,
) -> tuple[int, int]:
    n_sequences, sequence_length = occupied.shape
    if duration > sequence_length:
        raise ValueError("Corruption segment exceeds sequence length.")
    for _ in range(2_000):
        sequence_index = int(rng.integers(0, n_sequences))
        start = int(rng.integers(0, sequence_length - duration + 1))
        if not occupied[sequence_index, start : start + duration].any():
            return sequence_index, start
    for sequence_index in range(n_sequences):
        for start in range(sequence_length - duration + 1):
            if not occupied[sequence_index, start : start + duration].any():
                return sequence_index, start
    raise RuntimeError(
        "Unable to place non-overlapping corruption events; lower the ratio "
        "or segment length."
    )


def inject_corruptions(
    clean_values: np.ndarray,
    reference_values: np.ndarray,
    sequence_ids: tuple[str, ...],
    feature_names: tuple[str, ...],
    corruption_config: dict[str, Any],
    *,
    seed: int,
) -> CorruptionResult:
    """Inject five labeled corruption types into clean sensor observations.

    The configured ratio is measured over sequence-time rows. A corrupted row
    contains exactly one affected feature, so row-level labels remain unambiguous.
    Magnitude scales and replacement candidates come only from reference_values,
    which the pipeline supplies from clean training sequences.
    """

    if clean_values.ndim != 3:
        raise ValueError("clean_values must have shape [sequence, time, feature].")
    n_sequences, sequence_length, n_features = clean_values.shape
    if len(sequence_ids) != n_sequences or len(feature_names) != n_features:
        raise ValueError("Sequence or feature labels do not match clean_values.")
    if not np.isfinite(clean_values).all():
        raise ValueError("Milestone 0 clean values must all be finite.")
    if reference_values.ndim != 3 or reference_values.shape[2] != n_features:
        raise ValueError(
            "reference_values must have shape [reference_sequence, time, feature]."
        )
    if reference_values.shape[0] == 0 or not np.isfinite(reference_values).all():
        raise ValueError("reference_values must be non-empty and finite.")

    ratio = float(corruption_config["ratio"])
    corruption_types = list(corruption_config["types"])
    total_rows = n_sequences * sequence_length
    target_rows = int(round(total_rows * ratio))
    if target_rows == 0:
        return CorruptionResult(
            observed_values=clean_values.copy(),
            mask=np.zeros_like(clean_values, dtype=bool),
            events=(),
        )
    if target_rows < len(corruption_types):
        raise ValueError(
            "Target corruption rows must be at least the number of corruption types."
        )

    rng = np.random.default_rng(seed)
    observed = clean_values.copy()
    mask = np.zeros_like(clean_values, dtype=bool)
    occupied_rows = np.zeros((n_sequences, sequence_length), dtype=bool)
    feature_scales = np.std(reference_values, axis=(0, 1), ddof=1)
    feature_scales = np.where(feature_scales > 1e-12, feature_scales, 1.0)
    replacement_candidates = reference_values.reshape(-1, n_features)
    budgets = _balanced_budgets(target_rows, corruption_types)
    min_length = int(corruption_config["min_segment_length"])
    max_length = int(corruption_config["max_segment_length"])
    magnitude_ranges = corruption_config["magnitude_std"]
    events: list[dict[str, Any]] = []

    for corruption_type in corruption_types:
        remaining = budgets[corruption_type]
        while remaining > 0:
            duration = (
                1
                if corruption_type in {"spike", "random_replacement"}
                else _choose_duration(rng, remaining, min_length, max_length)
            )
            sequence_index, start = _find_open_segment(
                rng,
                occupied_rows,
                duration,
            )
            end = start + duration
            feature_index = int(rng.integers(0, n_features))
            clean_slice = clean_values[sequence_index, start:end, feature_index]
            replacement_value: float | None = None
            replacement_source: str | None = None
            missing_value_encoding: str | None = None
            if corruption_type == "missing":
                observed[sequence_index, start:end, feature_index] = np.nan
                offsets = np.full(duration, np.nan, dtype=float)
                missing_value_encoding = "NaN"
            elif corruption_type == "random_replacement":
                candidates = replacement_candidates[:, feature_index]
                original_value = float(clean_slice[0])
                replacement_value = float(
                    candidates[int(rng.integers(0, len(candidates)))]
                )
                for _ in range(100):
                    if replacement_value != original_value:
                        break
                    replacement_value = float(
                        candidates[int(rng.integers(0, len(candidates)))]
                    )
                if replacement_value == original_value:
                    replacement_value = float(
                        np.nextafter(replacement_value, np.inf)
                    )
                observed[sequence_index, start, feature_index] = replacement_value
                offsets = np.asarray([replacement_value - original_value])
                replacement_source = "clean_train_empirical"
            else:
                lower, upper = magnitude_ranges[corruption_type]
                signed_scale = float(rng.choice((-1.0, 1.0))) * float(
                    rng.uniform(lower, upper)
                )
                final_magnitude = signed_scale * float(
                    feature_scales[feature_index]
                )
                if corruption_type == "drift":
                    offsets = np.linspace(
                        0.1 * final_magnitude,
                        final_magnitude,
                        duration,
                        dtype=float,
                    )
                else:
                    offsets = np.full(duration, final_magnitude, dtype=float)
                observed[sequence_index, start:end, feature_index] += offsets

            mask[sequence_index, start:end, feature_index] = True
            occupied_rows[sequence_index, start:end] = True
            events.append(
                {
                    "event_id": f"event_{len(events):05d}",
                    "corruption_type": corruption_type,
                    "sequence_index": sequence_index,
                    "sequence_id": sequence_ids[sequence_index],
                    "feature_index": feature_index,
                    "corruption_feature": feature_names[feature_index],
                    "start_time_index": start,
                    "end_time_index_exclusive": end,
                    "duration": duration,
                    "magnitude_start": (
                        None if np.isnan(offsets[0]) else float(offsets[0])
                    ),
                    "magnitude_end": (
                        None if np.isnan(offsets[-1]) else float(offsets[-1])
                    ),
                    "magnitude_unit": FEATURE_UNITS.get(
                        feature_names[feature_index],
                        "feature_native_unit",
                    ),
                    "replacement_value": replacement_value,
                    "replacement_source": replacement_source,
                    "missing_value_encoding": missing_value_encoding,
                }
            )
            remaining -= duration

    changed = observed != clean_values
    if not np.array_equal(changed, mask):
        raise RuntimeError("Internal error: corruption mask does not match changes.")
    if int(mask.any(axis=2).sum()) != target_rows:
        raise RuntimeError("Internal error: achieved corruption row count is wrong.")
    if np.any(mask.sum(axis=2) > 1):
        raise RuntimeError("Internal error: a row contains multiple corruptions.")
    missing_mask = np.isnan(observed)
    if np.any(missing_mask & ~mask):
        raise RuntimeError("Internal error: unmasked missing observations exist.")
    if not np.isfinite(observed[~missing_mask]).all():
        raise RuntimeError("Internal error: non-missing observations must be finite.")
    return CorruptionResult(
        observed_values=observed,
        mask=mask,
        events=tuple(events),
    )
