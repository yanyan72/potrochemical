"""Calibrated channel-wise marginal or Gaussian conditional consistency scores."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf

from .logistics_simulator import METHODS


def channel_deviation(values: np.ndarray, reference: dict[str, Any], kind: str) -> np.ndarray:
    """Return standardized absolute channel deviations, preserving unavailable scores.

    Conditional residual: |P(x-mu)|_j / sqrt(P_jj), P=Sigma^-1.
    Marginal residual: |x_j-mu_j| / sqrt(Sigma_jj).
    Conditional scores need every contemporaneous channel; no missing-value truth is used.
    """

    delta = np.asarray(values, dtype=float) - np.asarray(reference["center"])
    if kind == "marginal":
        return np.abs(delta) / np.sqrt(np.diag(reference["covariance"]))
    if kind != "conditional":
        raise ValueError("Unknown residual kind.")
    precision = np.asarray(reference["precision"])
    result = np.full(delta.shape, np.nan)
    complete = np.isfinite(delta).all(axis=1)
    result[complete] = np.abs(delta[complete] @ precision) / np.sqrt(np.diag(precision))
    return result


def fit_sensor_reference(
    fit_values: np.ndarray, fit_regimes: np.ndarray,
    calibration_values: np.ndarray, calibration_regimes: np.ndarray,
    *, method: str, single_regime: str, quantile: float,
) -> dict[str, Any]:
    """Fit on reliable training rows and calibrate on disjoint training sequences.

    Inputs intentionally contain no corruption labels. The caller must enforce
    sequence-disjoint fit/calibration membership; val/test are never accepted here.
    """

    fit_values = np.asarray(fit_values, dtype=float)
    calibration_values = np.asarray(calibration_values, dtype=float)
    if method not in METHODS or not 0 < quantile < 1:
        raise ValueError("Invalid method or quantile.")
    if (fit_values.ndim != 2 or calibration_values.ndim != 2
            or fit_values.shape[1] != calibration_values.shape[1]
            or len(fit_values) != len(fit_regimes)
            or len(calibration_values) != len(calibration_regimes)
            or not np.isfinite(fit_values).all() or not np.isfinite(calibration_values).all()):
        raise ValueError("Finite matching reference matrices and regime vectors required.")
    if len(fit_values) < 2:
        raise ValueError("Insufficient fitting data.")
    center = fit_values.mean(axis=0)
    scale = fit_values.std(axis=0, ddof=1)
    if np.any(scale <= np.finfo(float).eps):
        raise ValueError("Constant channels cannot be calibrated.")
    x = (fit_values - center) / scale
    cal = (calibration_values - center) / scale
    scope, kind = method.split("_")
    keys = sorted(set(fit_regimes)) if scope == "regime" else ["all"]
    references = {}
    for key in keys:
        if scope == "pooled":
            select, cselect = np.ones(len(x), bool), np.ones(len(cal), bool)
        else:
            label = key if scope == "regime" else single_regime
            select, cselect = fit_regimes == label, calibration_regimes == label
        if select.sum() < 2 or cselect.sum() < 2:
            raise ValueError("Each reference requires fitting and calibration rows.")
        estimator = LedoitWolf().fit(x[select])
        ref = {
            "center": estimator.location_.tolist(),
            "covariance": estimator.covariance_.tolist(),
            "precision": estimator.precision_.tolist(),
            "fit_rows": int(select.sum()), "calibration_rows": int(cselect.sum()),
        }
        threshold = np.quantile(channel_deviation(cal[cselect], ref, kind), quantile, axis=0)
        if not np.isfinite(threshold).all() or np.any(threshold <= 0):
            raise ValueError("Degenerate calibration thresholds.")
        ref["threshold"] = threshold.tolist()
        references[key] = ref
    return {"method": method, "quantile": quantile, "scaler_center": center.tolist(),
            "scaler_scale": scale.tolist(), "references": references}


def score_sensors(values: np.ndarray, regimes: np.ndarray, model: dict[str, Any]) -> dict[str, np.ndarray]:
    """Score observed channels without labels; uncovered regimes are undetermined."""

    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or len(x) != len(regimes) or x.shape[1] != len(model["scaler_center"]):
        raise ValueError("Observation dimensions do not match reference.")
    if np.isinf(x).any():
        raise ValueError("Use NaN for missing values, never infinity.")
    normalized = (x - model["scaler_center"]) / model["scaler_scale"]
    ratio = np.full(x.shape, np.nan)
    scope, kind = model["method"].split("_")
    for key, ref in model["references"].items():
        selected = regimes == key if scope == "regime" else np.ones(len(x), bool)
        ratio[selected] = channel_deviation(normalized[selected], ref, kind) / ref["threshold"]
    available = np.isfinite(ratio)
    score = np.full(x.shape, np.nan)
    with np.errstate(over="ignore"):
        score[available] = np.exp(-0.5 * ratio[available] ** 2)
    return {"deviation_ratio": ratio, "reliability": score, "available": available,
            "alarm": available & (ratio > 1.0)}
