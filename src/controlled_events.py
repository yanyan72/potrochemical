"""Independent validation innovations and paired duration/count/onset experiments."""
from __future__ import annotations
from typing import Any
import numpy as np
from .logistics_robustness import StressData


def validate_controlled(config: dict[str, Any], source: dict[str, Any]) -> None:
    """Reject mixed-factor contrasts, impossible placement and ambiguous budgets."""
    if config["data_source"] != "synthetic" or config["threshold_quantile"] != source["threshold_quantile"]:
        raise ValueError("Use synthetic data and the source nominal budget.")
    if (not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"])
            or not set(config["seeds"]) <= set(source["seeds"])):
        raise ValueError("Require unique source seeds.")
    if set(config["methods"]) != {"regime_marginal", "regime_conditional"} or len(config["methods"]) != 2:
        raise ValueError("Require both distinct residual methods.")
    for key in ("validation_namespace", "post_event_steps", "max_duration", "max_events", "minimum_gap"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Positive integer required: {key}")
    if not np.isfinite(config["cusum_k"]) or config["cusum_k"] <= 0:
        raise ValueError("Positive finite CUSUM allowance required.")
    length, interval = source["sequence_length"], source["stress"]["transition_interval"]
    if (config["max_events"]*config["max_duration"] +
            (config["max_events"]-1)*config["minimum_gap"] + config["post_event_steps"] > length):
        raise ValueError("Event template cannot fit.")
    scenarios = config["scenarios"]
    by_name = {s["name"]: s for s in scenarios}
    if not scenarios or len(by_name) != len(scenarios):
        raise ValueError("Require unique scenarios.")
    fields = ("duration", "count", "magnitude", "switching", "delay", "placement", "offset")
    if sum(s["compare_to"] is None for s in scenarios) != 1:
        raise ValueError("Exactly one reference scenario required.")
    for s in scenarios:
        if not s["name"] or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in s["name"]):
            raise ValueError("Unsafe scenario name.")
        for name, maximum in (("duration", config["max_duration"]), ("count", config["max_events"])):
            if type(s[name]) is not int or not 1 <= s[name] <= maximum:
                raise ValueError("Invalid event duration/count.")
        if not np.isfinite(s["magnitude"]) or s["magnitude"] <= 0 or type(s["switching"]) is not bool:
            raise ValueError("Invalid magnitude/switching.")
        if type(s["delay"]) is not int or not 0 <= s["delay"] < interval or (s["delay"] and not s["switching"]):
            raise ValueError("Invalid delay.")
        if type(s["offset"]) is not int or s["placement"] not in {"random", "transition"}:
            raise ValueError("Invalid placement.")
        if s["placement"] == "random" and s["offset"] != 0:
            raise ValueError("Random placement has no fixed offset.")
        if s["placement"] == "transition":
            starts = np.arange(interval, length, interval) + s["offset"]
            if (not s["switching"] or s["count"] != len(starts) or starts.min() < 0
                    or starts.max()+s["duration"]+config["post_event_steps"] > length
                    or np.any(np.diff(starts) < s["duration"]+config["minimum_gap"])):
                raise ValueError("Invalid transition-centered events.")
        if s["compare_to"] is not None:
            previous = by_name.get(s["compare_to"])
            if previous is None or sum(s[k] != previous[k] for k in fields) != 1:
                raise ValueError("Each contrast must change exactly one factor.")


def validation_innovations(source: dict[str, Any], seed: int, namespace: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate fresh val-only normalized trajectories; never generate training/test."""
    length, count = source["sequence_length"], source["sequences_per_regime"]["val"]
    loading = np.asarray(source["factor_loadings"])
    rho = source["ar_rho"]
    values, ids, initial = [], [], []
    for ri, regime in enumerate(source["regimes"]):
        for i in range(count):
            rng = np.random.default_rng(np.random.SeedSequence([seed, namespace, ri, i]))
            latent = np.empty(length)
            latent[0] = rng.normal()
            for t in range(1, length):
                latent[t] = rho*latent[t-1] + np.sqrt(1-rho**2)*rng.normal()
            values.append(latent[:, None]*loading + rng.normal(size=(length, len(loading)))*np.sqrt(1-loading**2))
            ids.append(f"valf_{namespace}_{seed}_{regime}_{i:03d}")
            initial.append(regime)
    return np.asarray(values), np.asarray(ids), np.asarray(initial)


def candidate_starts(length: int, *, max_duration: int, max_events: int, gap: int,
                     post_steps: int, rng: np.random.Generator) -> np.ndarray:
    """Sample an ordered hard-core template, then randomize order for nested subsets.

    Uniform sorted slack positions induce gaps around nonoverlapping max-duration
    events. Marginal starts are constrained, not iid uniform on the full timeline.
    """
    slack = length-post_steps-max_duration-(max_events-1)*(max_duration+gap)
    if slack < 0:
        raise ValueError("Insufficient timeline for event template.")
    order = np.sort(rng.choice(slack+max_events, max_events, replace=False))
    starts = order + np.arange(max_events)*(max_duration+gap-1)
    return starts[rng.permutation(max_events)]


def controlled_scenario(innovations: np.ndarray, ids: np.ndarray, initial: np.ndarray,
                        source: dict[str, Any], config: dict[str, Any], scenario: dict[str, Any],
                        scales: dict[str, list[float]], seed: int) -> StressData:
    """Map fresh innovations to regimes and inject exact paired single-channel biases."""
    n, length, p = innovations.shape
    names = list(source["regimes"])
    interval = source["stress"]["transition_interval"]
    transition_window = source["stress"]["transition_window"]
    truth_state = np.empty((n, length), dtype="U64")
    transition = np.zeros((n, length), bool)
    clean = np.empty_like(innovations)
    boundaries = np.arange(interval, length, interval)
    for i, first in enumerate(initial):
        indices = np.full(length, names.index(first))
        if scenario["switching"]:
            indices = (indices + np.arange(length)//interval) % len(names)
            for t in boundaries:
                transition[i, t:t+transition_window] = True
        for ri, regime in enumerate(names):
            selected = indices == ri
            profile = source["regimes"][regime]
            clean[i, selected] = np.asarray(profile["mean"])+innovations[i, selected]*profile["std"]
            truth_state[i, selected] = regime
    records = truth_state[:, np.maximum(np.arange(length)-scenario["delay"], 0)].copy()
    observed, mask, events = clean.copy(), np.zeros_like(clean, bool), []
    for i in range(n):
        rng = np.random.default_rng(np.random.SeedSequence([seed, config["validation_namespace"], 53000, i]))
        starts = candidate_starts(length, max_duration=config["max_duration"], max_events=config["max_events"],
                                 gap=config["minimum_gap"], post_steps=config["post_event_steps"], rng=rng)
        if scenario["placement"] == "transition":
            starts = boundaries + scenario["offset"]
        for slot, start in enumerate(starts[:scenario["count"]]):
            start, end = int(start), int(start)+scenario["duration"]
            # Separate slot stream keeps channel/sign identical across count/duration/placement contrasts.
            rng_slot = np.random.default_rng(np.random.SeedSequence([seed, config["validation_namespace"], 54000, i, slot]))
            channel = int(rng_slot.integers(p))
            sign = float(rng_slot.choice([-1., 1.]))
            scale = np.asarray([scales[r][channel] for r in truth_state[i, start:end]])
            observed[i, start:end, channel] += sign*scenario["magnitude"]*scale
            mask[i, start:end, channel] = True
            nearest = int(boundaries[np.argmin(abs(boundaries-start))]) if scenario["switching"] else None
            events.append({"sequence_index": i, "sequence_id": str(ids[i]), "slot": slot,
                "start": start, "end_exclusive": end, "channels": [channel], "signs": [sign],
                "duration": scenario["duration"], "magnitude_std": scenario["magnitude"],
                "offset_from_nearest_transition": start-nearest if nearest is not None else None,
                "scale_source": "E1c_train_fit_per_true_regime", "split": "val_new",
                "corruption_type": "segment_bias"})
    return StressData(clean, observed, mask, truth_state, records, transition, ids, tuple(events))
