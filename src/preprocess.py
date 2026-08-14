"""Leakage-safe imputation and standardization for multivariate sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PreprocessorParams:
    """Parameters fitted exclusively from clean training observations."""

    feature_names: tuple[str, ...]
    imputation_strategy: str
    imputation_values: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    std_ddof: int
    fit_source: str = "clean_train"
    fit_split: str = "train"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the parameters."""

        return {
            "feature_names": list(self.feature_names),
            "imputation_strategy": self.imputation_strategy,
            "imputation_values": self.imputation_values.tolist(),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "std_ddof": self.std_ddof,
            "fit_source": self.fit_source,
            "fit_split": self.fit_split,
        }


def fit_preprocessor(
    clean_train_values: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    imputation_strategy: str = "feature_median",
    std_ddof: int = 1,
    minimum_scale: float = 1e-12,
) -> PreprocessorParams:
    """Fit imputation and scaling parameters from clean training values only."""

    values = np.asarray(clean_train_values, dtype=float)
    if values.ndim < 2:
        raise ValueError("clean_train_values must have shape [..., feature].")
    if values.shape[-1] != len(feature_names):
        raise ValueError("Feature labels do not match clean_train_values.")
    if not feature_names or len(feature_names) != len(set(feature_names)):
        raise ValueError("feature_names must be non-empty and unique.")
    if not np.isfinite(values).all():
        raise ValueError("clean_train_values must be finite.")
    if imputation_strategy != "feature_median":
        raise ValueError("Only 'feature_median' is supported in Milestone 1.")
    if not isinstance(std_ddof, int) or isinstance(std_ddof, bool) or std_ddof < 0:
        raise ValueError("std_ddof must be a non-negative integer.")

    flattened = values.reshape(-1, values.shape[-1])
    if flattened.shape[0] <= std_ddof:
        raise ValueError("Not enough clean training rows for the requested std_ddof.")
    imputation_values = np.median(flattened, axis=0)
    means = np.mean(flattened, axis=0)
    scales = np.std(flattened, axis=0, ddof=std_ddof)
    if not np.isfinite(scales).all() or np.any(scales <= minimum_scale):
        raise ValueError("Every feature must have a finite, non-zero training scale.")

    return PreprocessorParams(
        feature_names=tuple(feature_names),
        imputation_strategy=imputation_strategy,
        imputation_values=imputation_values,
        means=means,
        scales=scales,
        std_ddof=std_ddof,
    )


def impute_missing(
    values: np.ndarray,
    params: PreprocessorParams,
) -> np.ndarray:
    """Replace NaNs with fitted feature medians without modifying the input."""

    array = np.asarray(values, dtype=float)
    if array.ndim < 2 or array.shape[-1] != len(params.feature_names):
        raise ValueError("values must have shape [..., feature].")
    if np.isinf(array).any():
        raise ValueError("values must not contain positive or negative infinity.")
    imputed = array.copy()
    missing = np.isnan(imputed)
    if missing.any():
        feature_indices = np.nonzero(missing)[-1]
        imputed[missing] = params.imputation_values[feature_indices]
    if not np.isfinite(imputed).all():
        raise RuntimeError("Imputation did not produce finite values.")
    return imputed


def transform(
    values: np.ndarray,
    params: PreprocessorParams,
) -> np.ndarray:
    """Impute and standardize values using already-fitted training parameters."""

    imputed = impute_missing(values, params)
    standardized = (imputed - params.means) / params.scales
    if not np.isfinite(standardized).all():
        raise RuntimeError("Standardization did not produce finite values.")
    return standardized
