"""Trust-distance primitives built from a frozen normal reference model."""

from __future__ import annotations

import numpy as np


def squared_mahalanobis_distance(
    standardized_values: np.ndarray,
    location: np.ndarray,
    precision: np.ndarray,
    *,
    negative_tolerance: float = 1e-10,
) -> np.ndarray:
    """Return squared Mahalanobis distances while preserving leading dimensions.

    The final input axis is interpreted as the feature axis. ``location`` and
    ``precision`` must come from a previously fitted reference model; this
    function performs no fitting and therefore cannot alter train/validation/test
    boundaries.
    """

    values = np.asarray(standardized_values, dtype=float)
    center = np.asarray(location, dtype=float)
    inverse_covariance = np.asarray(precision, dtype=float)

    if values.ndim < 1:
        raise ValueError("standardized_values must have a feature axis.")
    if center.ndim != 1 or center.size == 0:
        raise ValueError("location must be a non-empty one-dimensional array.")
    if values.shape[-1] != center.shape[0]:
        raise ValueError("The input feature dimension does not match location.")
    if inverse_covariance.shape != (center.size, center.size):
        raise ValueError("precision must have shape [feature, feature].")
    if not np.isfinite(values).all():
        raise ValueError("standardized_values must all be finite.")
    if not np.isfinite(center).all() or not np.isfinite(inverse_covariance).all():
        raise ValueError("location and precision must all be finite.")
    if not isinstance(negative_tolerance, (int, float)) or isinstance(
        negative_tolerance, bool
    ):
        raise ValueError("negative_tolerance must be a non-negative number.")
    if not np.isfinite(negative_tolerance) or negative_tolerance < 0.0:
        raise ValueError("negative_tolerance must be a non-negative finite number.")
    if not np.allclose(
        inverse_covariance,
        inverse_covariance.T,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("precision must be symmetric.")
    if float(np.linalg.eigvalsh(inverse_covariance).min()) <= 0.0:
        raise ValueError("precision must be positive definite.")

    centered = values - center
    distances = np.einsum(
        "...i,ij,...j->...",
        centered,
        inverse_covariance,
        centered,
        optimize=True,
    )
    if not np.isfinite(distances).all():
        raise RuntimeError("Squared Mahalanobis distances must all be finite.")

    scale = max(1.0, float(np.max(np.abs(distances), initial=0.0)))
    if float(np.min(distances, initial=0.0)) < -negative_tolerance * scale:
        raise RuntimeError("Squared Mahalanobis distance is materially negative.")
    return np.maximum(distances, 0.0)
