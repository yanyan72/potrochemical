"""Independent, illustrative storage/transport monitoring benchmark; no quality law."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .corruption import inject_corruptions

SPLITS = ("train_fit", "train_cal", "val", "test")
METHODS = (
    "single_marginal", "pooled_marginal", "regime_marginal",
    "pooled_conditional", "regime_conditional",
)


@dataclass(frozen=True)
class LogisticsData:
    """Sequence-major arrays; hidden clean values and masks are evaluation-only."""

    clean: np.ndarray
    observed: np.ndarray
    mask: np.ndarray
    corruption_type: np.ndarray
    sequence_ids: np.ndarray
    splits: np.ndarray
    regimes: np.ndarray
    events: tuple[dict[str, Any], ...]


def validate_logistics_config(config: dict[str, Any]) -> None:
    """Reject invalid or scientifically ambiguous benchmark configurations."""

    if config.get("data_source") != "synthetic":
        raise ValueError("This generator only supports explicitly synthetic data.")
    seeds = config["seeds"]
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or s < 0 for s in seeds)):
        raise ValueError("seeds must be unique non-negative integers.")
    p = len(config["features"])
    if p < 2 or len(set(config["features"])) != p or len(config["units"]) != p:
        raise ValueError("Provide >=2 distinct features with matching units.")
    length = config["sequence_length"]
    if type(length) is not int or length < 8:
        raise ValueError("sequence_length must be an integer >=8.")
    if set(config["sequences_per_regime"]) != set(SPLITS):
        raise ValueError("Four disjoint fit/cal/val/test sequence splits are required.")
    for count in config["sequences_per_regime"].values():
        if type(count) is not int or count < 2:
            raise ValueError("Each regime/split needs >=2 complete sequences.")
    loading = np.asarray(config["factor_loadings"], dtype=float)
    if loading.shape != (p,) or not np.all(np.isfinite(loading) & (np.abs(loading) < 1)):
        raise ValueError("factor_loadings must be finite and strictly between -1 and 1.")
    if not 0 <= config["ar_rho"] < 1 or not 0 < config["threshold_quantile"] < 1:
        raise ValueError("Invalid correlation or calibration quantile.")
    regimes = config["regimes"]
    if len(regimes) < 2 or config["single_reference_regime"] not in regimes:
        raise ValueError("Need >=2 regimes and an existing single reference regime.")
    for profile in regimes.values():
        mean, std = np.asarray(profile["mean"]), np.asarray(profile["std"])
        if (mean.shape != (p,) or std.shape != (p,)
                or not np.isfinite(mean).all() or not np.isfinite(std).all()
                or np.any(std <= 0)):
            raise ValueError("Regime means/scales must be finite, matched and positive.")
    methods = config["methods"]
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError("Unknown or duplicate scoring methods.")
    c = config["corruption"]
    allowed = {"spike", "bias", "drift", "missing", "random_replacement"}
    if not 0 <= c["ratio"] < 1 or not c["types"] or set(c["types"]) - allowed:
        raise ValueError("Invalid corruption ratio/types.")
    if len(set(c["types"])) != len(c["types"]):
        raise ValueError("Duplicate corruption types.")
    if not 2 <= c["min_segment_length"] <= c["max_segment_length"] <= length:
        raise ValueError("Invalid corruption duration.")
    if c["replacement_strategy"] != "empirical_train_value":
        raise ValueError("Replacement samples must come from training only.")
    for kind in ("spike", "bias", "drift"):
        bounds = np.asarray(c["magnitude_std"][kind], dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 < bounds[0] <= bounds[1]:
            raise ValueError("Invalid corruption magnitude range.")


def generate_logistics(config: dict[str, Any], seed: int) -> LogisticsData:
    """Generate balanced fixed-regime sequences with split-independent random streams."""

    validate_logistics_config(config)
    length = config["sequence_length"]
    loading = np.asarray(config["factor_loadings"])
    rho = config["ar_rho"]
    clean, ids, splits, regimes = [], [], [], []
    for split_index, split in enumerate(SPLITS):
        for regime_index, (regime, profile) in enumerate(config["regimes"].items()):
            for seq in range(config["sequences_per_regime"][split]):
                rng = np.random.default_rng(np.random.SeedSequence([seed, split_index, regime_index, seq]))
                latent = np.empty(length)
                latent[0] = rng.normal()
                for t in range(1, length):
                    latent[t] = rho * latent[t - 1] + np.sqrt(1 - rho**2) * rng.normal()
                values = latent[:, None] * loading + rng.normal(size=(length, len(loading))) * np.sqrt(1 - loading**2)
                clean.append(np.asarray(profile["mean"]) + np.asarray(profile["std"]) * values)
                ids.append(f"{split}_{regime}_{seq:03d}")
                splits.append(split)
                regimes.append(regime)
    clean_array = np.asarray(clean)
    split_array = np.asarray(splits)
    id_array = np.asarray(ids)
    observed = clean_array.copy()
    mask = np.zeros_like(clean_array, dtype=bool)
    types = np.full(clean_array.shape[:2], "none", dtype="U24")
    reference = clean_array[split_array == "train_fit"]
    events: list[dict[str, Any]] = []
    for split_index, split in enumerate(SPLITS):
        indices = np.flatnonzero(split_array == split)
        corruption = inject_corruptions(
            clean_array[indices], reference, tuple(id_array[indices]),
            tuple(config["features"]), config["corruption"],
            seed=int(np.random.SeedSequence([seed, 10000, split_index]).generate_state(1)[0]),
        )
        observed[indices] = corruption.observed_values
        mask[indices] = corruption.mask
        for original in corruption.events:
            event = dict(original)
            index = int(indices[event["sequence_index"]])
            event.update(sequence_index=index, split=split, event_id=f"{split}_{event['event_id']}")
            event["magnitude_unit"] = config["units"][event["feature_index"]]
            types[index, event["start_time_index"]:event["end_time_index_exclusive"]] = event["corruption_type"]
            events.append(event)
    return LogisticsData(clean_array, observed, mask, types, id_array, split_array,
                         np.asarray(regimes), tuple(events))
