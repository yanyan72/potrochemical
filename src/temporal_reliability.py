"""Causal signed-residual EWMA with sequence-disjoint empirical calibration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


def signed_residuals(values: np.ndarray, regimes: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    """Compute signed marginal/conditional residuals from observations and records."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 3 or regimes.shape != x.shape[:2] or x.shape[-1] != len(model["scaler_center"]):
        raise ValueError("Expected sequence/time/channel values and sequence/time records.")
    if np.isinf(x).any() or model["method"] not in {"regime_marginal", "regime_conditional"}:
        raise ValueError("Use NaN for missing; only regime references are supported.")
    normalized = (x - model["scaler_center"]) / model["scaler_scale"]
    residual = np.full_like(x, np.nan)
    for key, ref in model["references"].items():
        selected = regimes == key
        delta = normalized[selected] - ref["center"]
        if model["method"] == "regime_marginal":
            residual[selected] = delta / np.sqrt(np.diag(ref["covariance"]))
        else:
            precision = np.asarray(ref["precision"])
            result = np.full_like(delta, np.nan)
            complete = np.isfinite(delta).all(axis=1)
            result[complete] = (delta[complete] @ precision) / np.sqrt(np.diag(precision))
            residual[selected] = result
    return residual


def causal_ewma(residuals: np.ndarray, records: np.ndarray, alpha: float) -> np.ndarray:
    """Filter signed residuals, resetting at sequence starts, record changes and gaps.

    Missing current residuals remain NaN. Their state resets to zero; no previous
    residual or score is substituted. The first valid update is alpha * residual.
    """
    residuals = np.asarray(residuals, dtype=float)
    if (residuals.ndim != 3 or records.shape != residuals.shape[:2]
            or not 0 < alpha <= 1 or np.isinf(residuals).any()):
        raise ValueError("Invalid sequence arrays or alpha.")
    result = np.full_like(residuals, np.nan)
    state = np.zeros((residuals.shape[0], residuals.shape[2]))
    for t in range(residuals.shape[1]):
        if t:
            state[records[:, t] != records[:, t - 1]] = 0.
        valid = np.isfinite(residuals[:, t])
        state[~valid] = 0.
        update = alpha * residuals[:, t] + (1 - alpha) * state
        state[valid] = update[valid]
        result[:, t] = np.where(valid, state, np.nan)
    return result


def calibrate_temporal(values: np.ndarray, records: np.ndarray, reference: dict[str, Any],
                       *, alpha: float, quantile: float) -> dict[str, Any]:
    """Calibrate absolute filtered residual quantiles on held-out training sequences."""
    if not 0 < quantile < 1:
        raise ValueError("Invalid quantile.")
    filtered = causal_ewma(signed_residuals(values, records, reference), records, alpha)
    thresholds = {}
    for regime in reference["references"]:
        selected = np.abs(filtered[records == regime])
        if len(selected) < 2 or not np.isfinite(selected).all():
            raise ValueError("Each calibration regime needs finite independent training data.")
        threshold = np.quantile(selected, quantile, axis=0)
        if not np.isfinite(threshold).all() or np.any(threshold <= 0):
            raise ValueError("Degenerate temporal thresholds.")
        thresholds[regime] = threshold.tolist()
    return {"reference": deepcopy(reference), "alpha": alpha, "quantile": quantile,
            "initial_state": "zero", "reset_on_record_change": True, "thresholds": thresholds}


def score_temporal(values: np.ndarray, records: np.ndarray, model: dict[str, Any]) -> dict[str, np.ndarray]:
    """Score complete trajectories causally; hidden fault labels are not arguments."""
    filtered = causal_ewma(signed_residuals(values, records, model["reference"]), records, model["alpha"])
    ratio = np.full_like(filtered, np.nan)
    for regime, threshold in model["thresholds"].items():
        selected = records == regime
        ratio[selected] = np.abs(filtered[selected]) / threshold
    available = np.isfinite(ratio)
    with np.errstate(over="ignore"):
        reliability = np.exp(-.5 * ratio ** 2)
    return {"deviation_ratio": ratio, "reliability": reliability, "available": available,
            "alarm": available & (ratio > 1.)}


def event_evaluation(alarm: np.ndarray, truth: np.ndarray, events: list[dict[str, Any]],
                     *, post_steps: int) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Return event/channel rows and a deduplicated post-event clean-cell mask.

    A fresh onset requires false->true during the event. A continuing pre-event
    alarm counts for any-active hit but not automatically as a fresh detection.
    Misses receive the event duration in capped delay; late alarms never count.
    """
    if alarm.shape != truth.shape or alarm.ndim != 3 or type(post_steps) is not int or post_steps < 1:
        raise ValueError("Matching sequence arrays and positive post window required.")
    fresh = alarm & ~np.concatenate((np.zeros_like(alarm[:, :1]), alarm[:, :-1]), axis=1)
    tail = np.zeros_like(truth, dtype=bool)
    rows = []
    n, length, p = alarm.shape
    for number, event in enumerate(events):
        i, start, end = event["sequence_index"], event["start"], event["end_exclusive"]
        if not (0 <= i < n and 0 <= start < end <= length):
            raise ValueError("Event outside sequence.")
        for channel in event["channels"]:
            if not 0 <= channel < p or not truth[i, start:end, channel].all():
                raise ValueError("Event and truth mask disagree.")
            active_hits = np.flatnonzero(alarm[i, start:end, channel])
            fresh_hits = np.flatnonzero(fresh[i, start:end, channel])
            delay = int(fresh_hits[0]) if len(fresh_hits) else float("nan")
            rows.append({"event_index": number, "sequence_index": i, "channel": channel,
                         "start": start, "end_exclusive": end, "duration": end - start,
                         "any_active_hit": bool(len(active_hits)), "fresh_onset_hit": bool(len(fresh_hits)),
                         "pre_active": bool(start > 0 and alarm[i, start - 1, channel]),
                         "fresh_delay_detected": delay,
                         "fresh_delay_capped": delay if len(fresh_hits) else end - start})
            tail[i, end:min(length, end + post_steps), channel] = True
    tail &= ~truth
    return rows, tail
