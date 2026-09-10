"""Tests for squared Mahalanobis distance computation."""

from __future__ import annotations

import numpy as np
import pytest

from src.trust_score import (
    assign_trust_groups,
    exponential_trust_score,
    fit_trust_calibration,
    squared_mahalanobis_distance,
)


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


def test_calibration_matches_train_reference_quantiles_without_observations() -> None:
    reference = np.linspace(0.1, 20.0, 800)
    external_observations = np.asarray([1e9, 2e9, 3e9])
    original_reference = reference.copy()

    first = fit_trust_calibration(reference)
    external_observations *= -1.0
    second = fit_trust_calibration(reference)

    expected = np.quantile(reference, [0.90, 0.99], method="linear")
    assert first.high_threshold_squared == pytest.approx(expected[0])
    assert first.low_threshold_squared == pytest.approx(expected[1])
    assert first.tau == pytest.approx(expected[0])
    assert first.to_dict() == second.to_dict()
    np.testing.assert_array_equal(reference, original_reference)


def test_trust_groups_use_documented_inclusive_boundaries() -> None:
    calibration = fit_trust_calibration(np.arange(1.0, 101.0))
    q90 = calibration.high_threshold_squared
    q99 = calibration.low_threshold_squared
    values = np.asarray([0.0, q90, np.nextafter(q90, np.inf), q99, np.nextafter(q99, np.inf)])

    groups = assign_trust_groups(values, calibration)

    np.testing.assert_array_equal(
        groups,
        np.asarray(["high", "high", "uncertain", "uncertain", "low"]),
    )


def test_exponential_trust_is_monotonic_bounded_and_stable() -> None:
    calibration = fit_trust_calibration(np.arange(1.0, 101.0))
    values = np.asarray([0.0, 1.0, calibration.tau, 1e3, 1e12])

    scores = exponential_trust_score(values, calibration)

    assert scores[0] == pytest.approx(1.0)
    assert scores[2] == pytest.approx(np.exp(-0.5))
    assert np.isfinite(scores).all()
    assert ((0.0 <= scores) & (scores <= 1.0)).all()
    assert (np.diff(scores) <= 0.0).all()


def test_calibration_and_mapping_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        fit_trust_calibration(np.ones((2, 2)))
    with pytest.raises(ValueError, match="non-negative"):
        fit_trust_calibration(np.asarray([1.0, -1.0]))
    with pytest.raises(ValueError, match="Quantiles"):
        fit_trust_calibration(np.arange(1.0, 10.0), high_quantile=0.99, low_quantile=0.90)
    calibration = fit_trust_calibration(np.arange(1.0, 101.0))
    with pytest.raises(ValueError, match="non-negative"):
        assign_trust_groups(np.asarray([-1.0, 1.0]), calibration)
