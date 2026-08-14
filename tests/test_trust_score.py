"""Tests for squared Mahalanobis distance computation."""

from __future__ import annotations

import numpy as np
import pytest

from src.trust_score import squared_mahalanobis_distance


def test_squared_mahalanobis_matches_explicit_quadratic_form() -> None:
    values = np.asarray([[1.0, 2.0], [3.0, -1.0], [0.5, 0.25]])
    location = np.asarray([0.5, -0.5])
    precision = np.asarray([[2.0, 0.4], [0.4, 1.5]])
    expected = np.asarray(
        [
            (row - location) @ precision @ (row - location)
            for row in values
        ]
    )

    actual = squared_mahalanobis_distance(values, location, precision)

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_squared_mahalanobis_preserves_leading_shape_and_inputs() -> None:
    values = np.asarray(
        [
            [[0.0, 0.0], [1.0, 2.0]],
            [[-1.0, 0.5], [3.0, -2.0]],
        ]
    )
    location = np.asarray([0.0, 0.0])
    precision = np.asarray([[1.0, 0.2], [0.2, 2.0]])
    original_values = values.copy()
    original_location = location.copy()
    original_precision = precision.copy()

    actual = squared_mahalanobis_distance(values, location, precision)

    assert actual.shape == (2, 2)
    assert actual[0, 0] == pytest.approx(0.0)
    assert np.isfinite(actual).all()
    assert (actual >= 0.0).all()
    np.testing.assert_array_equal(values, original_values)
    np.testing.assert_array_equal(location, original_location)
    np.testing.assert_array_equal(precision, original_precision)


def test_squared_mahalanobis_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="feature dimension"):
        squared_mahalanobis_distance(
            np.ones((3, 2)),
            np.zeros(3),
            np.eye(3),
        )
    with pytest.raises(ValueError, match="finite"):
        squared_mahalanobis_distance(
            np.asarray([[np.nan, 0.0]]),
            np.zeros(2),
            np.eye(2),
        )
    with pytest.raises(ValueError, match="symmetric"):
        squared_mahalanobis_distance(
            np.ones((3, 2)),
            np.zeros(2),
            np.asarray([[1.0, 1.0], [0.0, 1.0]]),
        )
    with pytest.raises(ValueError, match="positive definite"):
        squared_mahalanobis_distance(
            np.ones((3, 2)),
            np.zeros(2),
            np.asarray([[1.0, 0.0], [0.0, -1.0]]),
        )
