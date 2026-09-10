"""Trust-distance primitives built from a frozen normal reference model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrustCalibration:
    """Train-reference quantiles and scale for deterministic trust scoring."""

    high_quantile: float
    low_quantile: float
    high_threshold_squared: float
    low_threshold_squared: float
    quantile_method: str
    tau: float
    tau_source: str
    fit_source: str
    num_reference_distances: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable calibration representation."""

        return {
            "high_quantile": self.high_quantile,
            "low_quantile": self.low_quantile,
            "high_threshold_squared": self.high_threshold_squared,
            "low_threshold_squared": self.low_threshold_squared,
            "quantile_method": self.quantile_method,
            "tau": self.tau,
            "tau_source": self.tau_source,
            "fit_source": self.fit_source,
            "num_reference_distances": self.num_reference_distances,
        }


def _validated_squared_distances(
    squared_distances: np.ndarray,
    *,
    require_one_dimensional: bool = False,
) -> np.ndarray:
    values = np.asarray(squared_distances, dtype=float)
    if values.ndim == 0 or (require_one_dimensional and values.ndim != 1):
        expected = "one-dimensional " if require_one_dimensional else "non-scalar "
        raise ValueError(f"squared_distances must be a {expected}array.")
    if values.size == 0:
        raise ValueError("squared_distances must not be empty.")
    if not np.isfinite(values).all():
        raise ValueError("squared_distances must all be finite.")
    if (values < 0.0).any():
        raise ValueError("squared_distances must all be non-negative.")
    return values


def fit_trust_calibration(
    reference_squared_distances: np.ndarray,
    *,
    high_quantile: float = 0.90,
    low_quantile: float = 0.99,
    quantile_method: str = "linear",
    tau_source: str = "q90",
    fit_source: str = "normal_train_clean_reference_distances",
) -> TrustCalibration:
    """Fit q90/q99 and trust scale using only normal train reference distances."""

    reference = _validated_squared_distances(
        reference_squared_distances,
        require_one_dimensional=True,
    )
    if reference.size < 2:
        raise ValueError("At least two reference distances are required.")
    if not 0.0 < high_quantile < low_quantile < 1.0:
        raise ValueError("Quantiles must satisfy 0 < high_quantile < low_quantile < 1.")
    if quantile_method != "linear":
        raise ValueError("Only the deterministic 'linear' quantile method is supported.")
    if tau_source != "q90":
        raise ValueError("Only tau_source='q90' is supported in the E1 baseline.")
    if not isinstance(fit_source, str) or not fit_source:
        raise ValueError("fit_source must be a non-empty string.")

    high_threshold, low_threshold = np.quantile(
        reference,
        [high_quantile, low_quantile],
        method=quantile_method,
    )
    high_threshold = float(high_threshold)
    low_threshold = float(low_threshold)
    if high_threshold <= 0.0:
        raise ValueError("The q90 reference scale must be positive.")
    if low_threshold <= high_threshold:
        raise ValueError("The q99 threshold must be greater than q90.")
    return TrustCalibration(
        high_quantile=float(high_quantile),
        low_quantile=float(low_quantile),
        high_threshold_squared=high_threshold,
        low_threshold_squared=low_threshold,
        quantile_method=quantile_method,
        tau=high_threshold,
        tau_source=tau_source,
        fit_source=fit_source,
        num_reference_distances=int(reference.size),
    )


def assign_trust_groups(
    squared_distances: np.ndarray,
    calibration: TrustCalibration,
) -> np.ndarray:
    """Assign high, uncertain, or low trust with inclusive documented boundaries."""

    values = _validated_squared_distances(squared_distances)
    groups = np.full(values.shape, "low", dtype="<U9")
    groups[values <= calibration.low_threshold_squared] = "uncertain"
    groups[values <= calibration.high_threshold_squared] = "high"
    return groups


def exponential_trust_score(
    squared_distances: np.ndarray,
    calibration: TrustCalibration,
) -> np.ndarray:
    """Map squared distance monotonically to [0, 1] using frozen train tau."""

    values = _validated_squared_distances(squared_distances)
    if not np.isfinite(calibration.tau) or calibration.tau <= 0.0:
        raise ValueError("calibration.tau must be positive and finite.")
    scores = np.exp(-values / (2.0 * calibration.tau))
    if not np.isfinite(scores).all():
        raise RuntimeError("Trust scores must all be finite.")
    return np.clip(scores, 0.0, 1.0)


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
