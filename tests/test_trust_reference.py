"""Tests for clean-normal Ledoit-Wolf reference estimation."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.covariance import LedoitWolf

from src.trust_reference import fit_trust_reference


def test_trust_reference_matches_sklearn_and_is_stable() -> None:
    rng = np.random.default_rng(31415)
    latent = rng.normal(size=(300, 2))
    values = np.column_stack(
        (
            latent[:, 0],
            0.8 * latent[:, 0] + 0.2 * latent[:, 1],
            latent[:, 1],
            -0.3 * latent[:, 0] + 0.5 * latent[:, 1] + rng.normal(0, 0.05, 300),
        )
    )
    original = values.copy()
    params = fit_trust_reference(values, ("a", "b", "c", "d"))
    expected = LedoitWolf(assume_centered=False, store_precision=True).fit(values)

    np.testing.assert_array_equal(values, original)
    np.testing.assert_allclose(params.location, expected.location_)
    np.testing.assert_allclose(params.covariance, expected.covariance_)
    np.testing.assert_allclose(params.precision, expected.precision_)
    assert params.shrinkage == pytest.approx(expected.shrinkage_)
    np.testing.assert_allclose(params.covariance, params.covariance.T, atol=1e-14)
    np.testing.assert_allclose(params.precision, params.precision.T, atol=1e-14)
    np.testing.assert_allclose(
        params.covariance @ params.precision,
        np.eye(4),
        atol=1e-10,
    )
    diagnostics = params.diagnostics()
    assert diagnostics["covariance_min_eigenvalue"] > 0.0
    assert diagnostics["covariance_condition_number"] >= 1.0
    assert all(np.isfinite(value) for value in diagnostics.values())


def test_trust_reference_rejects_invalid_reference() -> None:
    with pytest.raises(ValueError, match="shape"):
        fit_trust_reference(np.ones((2, 3, 4)), ("a", "b", "c", "d"))
    with pytest.raises(ValueError, match="finite"):
        fit_trust_reference(
            np.asarray([[1.0, 2.0], [np.nan, 3.0]]),
            ("a", "b"),
        )
    with pytest.raises(ValueError, match="two reference"):
        fit_trust_reference(np.asarray([[1.0, 2.0]]), ("a", "b"))
