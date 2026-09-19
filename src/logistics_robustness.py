"""Paired synthetic bias and delayed-regime stress scenarios; no quality target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .logistics_simulator import LogisticsData, validate_logistics_config


@dataclass(frozen=True)
class StressData:
    """Validation arrays with hidden true state separate from available records."""

    clean: np.ndarray
    observed: np.ndarray
    mask: np.ndarray
    true_regimes: np.ndarray
    recorded_regimes: np.ndarray
    transition: np.ndarray
    sequence_ids: np.ndarray
    events: tuple[dict[str, Any], ...]


def validate_stress_config(config: dict[str, Any]) -> None:
    """Validate the fixed one-factor comparisons before generating any output."""

    validate_logistics_config(config)
    stress = config["stress"]
    length, p = config["sequence_length"], len(config["features"])
    for key in ("segment_length", "transition_interval", "transition_window"):
        if type(stress[key]) is not int or not 1 <= stress[key] <= length:
            raise ValueError(f"Invalid {key}.")
    if stress["transition_window"] > stress["transition_interval"]:
        raise ValueError("Transition windows must not overlap.")
    if stress["transition_interval"] >= length:
        raise ValueError("At least one transition must occur.")
    scenarios = stress["scenarios"]
    names = [s["name"] for s in scenarios]
    if len(set(names)) != len(names) or stress["baseline"] not in names:
        raise ValueError("Unique scenarios and existing baseline required.")
    by_name = {s["name"]: s for s in scenarios}
    factors = ("switching", "magnitude", "ratio", "channels", "delay")
    for s in scenarios:
        if not s["name"] or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in s["name"]):
            raise ValueError("Scenario names must be safe lowercase identifiers.")
        if type(s["switching"]) is not bool:
            raise ValueError("switching must be boolean.")
        if not np.isfinite(s["magnitude"]) or s["magnitude"] <= 0 or not 0 < s["ratio"] < 1:
            raise ValueError("Positive finite magnitude and fractional ratio required.")
        if type(s["channels"]) is not int or not 1 <= s["channels"] <= p:
            raise ValueError("Invalid affected channel count.")
        if type(s["delay"]) is not int or not 0 <= s["delay"] < stress["transition_interval"]:
            raise ValueError("Delay must be shorter than a regime segment.")
        if s["delay"] and not s["switching"]:
            raise ValueError("Delay requires switching.")
        if s["name"] == stress["baseline"]:
            if s["compare_to"] is not None:
                raise ValueError("Baseline has no comparator.")
        else:
            comparator = by_name.get(s["compare_to"])
            if comparator is None or sum(s[k] != comparator[k] for k in factors) != 1:
                raise ValueError("Each comparison must change exactly one factor.")


def training_regime_scales(data: LogisticsData) -> dict[str, np.ndarray]:
    """Estimate physical-unit channel scales using only clean train_fit sequences."""

    return {str(regime): data.clean[(data.splits == "train_fit") & (data.regimes == regime)]
            .reshape(-1, data.clean.shape[-1]).std(axis=0, ddof=1)
            for regime in np.unique(data.regimes[data.splits == "train_fit"])}


def make_stress_scenario(data: LogisticsData, config: dict[str, Any],
                         scenario: dict[str, Any], seed: int) -> StressData:
    """Create paired val-only scenarios without exposing labels to the score API.

    Steady validation innovations are reused across all scenarios. Switching only
    remaps their configured mean/scale; it introduces no empirical dynamics claim.
    Corruption is a block bias, scaled by train_fit std of the true state at t.
    """

    val = data.splits == "val"
    clean = data.clean[val].copy()
    ids, initial = data.sequence_ids[val], data.regimes[val]
    n, length, p = clean.shape
    regimes = list(config["regimes"])
    true = np.empty((n, length), dtype=data.regimes.dtype)
    transition = np.zeros((n, length), dtype=bool)
    interval, window = (config["stress"][k] for k in ("transition_interval", "transition_window"))
    for i, regime in enumerate(initial):
        profile = config["regimes"][regime]
        z = (clean[i] - profile["mean"]) / profile["std"]
        state_indices = np.full(length, regimes.index(regime))
        if scenario["switching"]:
            state_indices = (state_indices + np.arange(length) // interval) % len(regimes)
            for start in range(interval, length, interval):
                transition[i, start:min(length, start + window)] = True
        for index, name in enumerate(regimes):
            selected = state_indices == index
            true[i, selected] = name
            target = config["regimes"][name]
            if scenario["switching"]:
                clean[i, selected] = np.asarray(target["mean"]) + z[selected] * target["std"]
    lagged = np.maximum(np.arange(length) - scenario["delay"], 0)
    recorded = true[:, lagged].copy()
    scales = training_regime_scales(data)
    observed, mask = clean.copy(), np.zeros_like(clean, dtype=bool)
    events = []
    segment = config["stress"]["segment_length"]
    for i in range(n):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 31000, i]))
        blocks = rng.permutation(np.arange(0, length, segment))
        budget = int(round(scenario["ratio"] * length))
        for start in blocks:
            if budget == 0:
                break
            start = int(start)
            end = min(start + segment, length, start + budget)
            # Always draw all channels and signs to keep k=1/k=2 scenarios paired.
            order = rng.permutation(p)
            signs = rng.choice([-1., 1.], size=p)
            selected = order[:scenario["channels"]]
            per_step_scale = np.asarray([scales[r] for r in true[i, start:end]])
            bias = scenario["magnitude"] * signs[selected] * per_step_scale[:, selected]
            observed[i, start:end, selected] += bias.T
            mask[i, start:end, selected] = True
            events.append({"sequence_id": str(ids[i]), "sequence_index": i,
                           "start": start, "end_exclusive": end,
                           "channels": selected.tolist(), "signs": signs[selected].tolist(),
                           "magnitude_std": scenario["magnitude"],
                           "scale_source": "clean_train_fit_per_true_regime",
                           "corruption_type": "segment_bias", "split": "val"})
            budget -= end - start
    return StressData(clean, observed, mask, true, recorded, transition, ids, tuple(events))
