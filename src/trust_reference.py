"""Stable normal-reference estimation for later trust scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class TrustReferenceParams:
    """Center and shrinkage covariance fitted from a normal train reference set."""

    feature_names: tuple[str, ...]
    location: np.ndarray
    covariance: np.ndarray
    precision: np.ndarray
    shrinkage: float
    estimator: str
    assume_centered: bool
    fit_split: str
    fit_condition: str
    fit_source: str
    num_reference_rows: int

    def diagnostics(self) -> dict[str, float]:
        """Return deterministic numerical stability diagnostics."""

        covariance_symmetry_error = float(
            np.max(np.abs(self.covariance - self.covariance.T))
        )
        precision_symmetry_error = float(
            np.max(np.abs(self.precision - self.precision.T))
        )
        identity = np.eye(len(self.feature_names))
        inverse_residual = float(
            np.max(np.abs(self.covariance @ self.precision - identity))
        )
        eigenvalues = np.linalg.eigvalsh(self.covariance)
        return {
            "covariance_symmetry_max_abs_error": covariance_symmetry_error,
            "precision_symmetry_max_abs_error": precision_symmetry_error,
            "covariance_min_eigenvalue": float(eigenvalues.min()),
            "covariance_max_eigenvalue": float(eigenvalues.max()),
            "covariance_condition_number": float(np.linalg.cond(self.covariance)),
            "inverse_identity_max_abs_error": inverse_residual,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable parameter representation."""

        return {
            "feature_names": list(self.feature_names),
            "location": self.location.tolist(),
            "covariance": self.covariance.tolist(),
            "precision": self.precision.tolist(),
            "shrinkage": self.shrinkage,
            "estimator": self.estimator,
            "assume_centered": self.assume_centered,
            "fit_split": self.fit_split,
            "fit_condition": self.fit_condition,
            "fit_source": self.fit_source,
            "num_reference_rows": self.num_reference_rows,
            "diagnostics": self.diagnostics(),
        }


def fit_trust_reference(
    standardized_clean_reference: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    assume_centered: bool = False,
    fit_split: str = "train",
    fit_condition: str = "normal",
    fit_source: str = "standardized_clean_train_normal",
) -> TrustReferenceParams:
    """Fit a Ledoit-Wolf reference model to standardized clean normal train rows."""

    values = np.asarray(standardized_clean_reference, dtype=float)
    if values.ndim != 2:
        raise ValueError(
            "standardized_clean_reference must have shape [row, feature]."
        )
    if not feature_names or len(feature_names) != len(set(feature_names)):
        raise ValueError("feature_names must be non-empty and unique.")
    if values.shape[1] != len(feature_names):
        raise ValueError("Feature labels do not match the reference matrix.")
    if values.shape[0] < 2:
        raise ValueError("At least two reference rows are required.")
    if not np.isfinite(values).all():
        raise ValueError("The reference matrix must be finite.")
    original = values.copy()
    estimator = LedoitWolf(
        assume_centered=assume_centered,
        store_precision=True,
    ).fit(values)
    if not np.array_equal(values, original):
        raise RuntimeError("Ledoit-Wolf unexpectedly modified the input matrix.")

    params = TrustReferenceParams(
        feature_names=tuple(feature_names),
        location=np.asarray(estimator.location_, dtype=float),
        covariance=np.asarray(estimator.covariance_, dtype=float),
        precision=np.asarray(estimator.precision_, dtype=float),
        shrinkage=float(estimator.shrinkage_),
        estimator="ledoit_wolf",
        assume_centered=assume_centered,
        fit_split=fit_split,
        fit_condition=fit_condition,
        fit_source=fit_source,
        num_reference_rows=int(values.shape[0]),
    )
    diagnostics = params.diagnostics()
    if diagnostics["covariance_min_eigenvalue"] <= 0.0:
        raise RuntimeError("Shrinkage covariance must be positive definite.")
    if not all(np.isfinite(value) for value in diagnostics.values()):
        raise RuntimeError("Reference diagnostics must all be finite.")
    return params
