"""Empirically calibrated continuous two-sided CUSUM residual baseline."""
from __future__ import annotations
from copy import deepcopy
from typing import Any
import numpy as np
from .temporal_reliability import signed_residuals


def causal_cusum(residuals: np.ndarray, records: np.ndarray, k: float) -> np.ndarray:
    """Return max of positive/negative sums; reset at records/gaps, not alarms."""
    x = np.asarray(residuals, dtype=float)
    if (x.ndim != 3 or records.shape != x.shape[:2] or not np.isfinite(k)
            or k <= 0 or np.isinf(x).any()):
        raise ValueError("Invalid sequence arrays or CUSUM reference allowance.")
    up = np.zeros((x.shape[0], x.shape[2]))
    down = np.zeros_like(up)
    result = np.full_like(x, np.nan)
    for t in range(x.shape[1]):
        if t:
            changed = records[:, t] != records[:, t-1]
            up[changed], down[changed] = 0., 0.
        valid = np.isfinite(x[:, t])
        up[~valid], down[~valid] = 0., 0.
        up[valid] = np.maximum(0., up[valid] + x[:, t][valid] - k)
        down[valid] = np.maximum(0., down[valid] - x[:, t][valid] - k)
        result[:, t] = np.where(valid, np.maximum(up, down), np.nan)
    return result


def calibrate_cusum(values: np.ndarray, records: np.ndarray, reference: dict[str, Any],
                    *, k: float, quantile: float) -> dict[str, Any]:
    """Fit per-regime/channel quantiles on independent clean training trajectories."""
    if not 0 < quantile < 1:
        raise ValueError("Invalid quantile.")
    statistic = causal_cusum(signed_residuals(values, records, reference), records, k)
    thresholds = {}
    for regime in reference["references"]:
        sample = statistic[records == regime]
        if len(sample) < 2 or not np.isfinite(sample).all():
            raise ValueError("Need finite calibration support in each regime.")
        threshold = np.quantile(sample, quantile, axis=0)
        if not np.isfinite(threshold).all() or np.any(threshold <= 0):
            raise ValueError("Degenerate CUSUM threshold.")
        thresholds[regime] = threshold.tolist()
    return {"reference": deepcopy(reference), "k": float(k), "quantile": quantile,
            "thresholds": thresholds, "initial_state": "zero", "reset_on_alarm": False,
            "reset_on_record_change": True, "reset_on_gap": "per_channel"}


def score_cusum(values: np.ndarray, records: np.ndarray, model: dict[str, Any]) -> dict[str, np.ndarray]:
    """Map the continuous two-sided statistic to a consistency score, not probability."""
    statistic = causal_cusum(signed_residuals(values, records, model["reference"]), records, model["k"])
    ratio = np.full_like(statistic, np.nan)
    for regime, threshold in model["thresholds"].items():
        selected = records == regime
        ratio[selected] = statistic[selected] / threshold
    available = np.isfinite(ratio)
    with np.errstate(over="ignore"):
        reliability = np.exp(-.5 * ratio**2)
    return {"deviation_ratio": ratio, "reliability": reliability, "available": available,
            "alarm": available & (ratio > 1.)}
