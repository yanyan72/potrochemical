"""Tests for leakage-safe missing-value imputation and standardization."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocess import fit_preprocessor, impute_missing, transform


def test_fit_and_transform_use_clean_train_parameters_only() -> None:
    clean_train = np.asarray(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
        ]
    )
    clean_train_original = clean_train.copy()
    params = fit_preprocessor(clean_train, ("a", "b"), std_ddof=1)

    flattened = clean_train.reshape(-1, 2)
    np.testing.assert_array_equal(params.imputation_values, np.median(flattened, axis=0))
    np.testing.assert_array_equal(params.means, np.mean(flattened, axis=0))
    np.testing.assert_array_equal(params.scales, np.std(flattened, axis=0, ddof=1))
    np.testing.assert_array_equal(clean_train, clean_train_original)

    validation_or_test = np.asarray([[[np.nan, 1000.0], [-999.0, np.nan]]])
    validation_original = validation_or_test.copy()
    imputed = impute_missing(validation_or_test, params)
    standardized = transform(validation_or_test, params)
    expected_imputed = np.asarray(
        [[[params.imputation_values[0], 1000.0], [-999.0, params.imputation_values[1]]]]
    )
    np.testing.assert_array_equal(imputed, expected_imputed)
    np.testing.assert_allclose(standardized, (expected_imputed - params.means) / params.scales)
    assert np.isfinite(standardized).all()
    np.testing.assert_allclose(validation_or_test, validation_original, equal_nan=True)

    changed_external_values = validation_or_test * 1_000_000.0
    unchanged_params = fit_preprocessor(clean_train, ("a", "b"), std_ddof=1)
    assert params.to_dict() == unchanged_params.to_dict()
    assert changed_external_values.shape == validation_or_test.shape


def test_preprocessor_rejects_nonfinite_training_and_zero_scale() -> None:
    with pytest.raises(ValueError, match="finite"):
        fit_preprocessor(np.asarray([[1.0, np.nan], [2.0, 3.0]]), ("a", "b"))
    with pytest.raises(ValueError, match="non-zero"):
        fit_preprocessor(np.asarray([[1.0, 2.0], [1.0, 3.0]]), ("a", "b"))

    params = fit_preprocessor(np.asarray([[1.0, 2.0], [2.0, 4.0]]), ("a", "b"))
    with pytest.raises(ValueError, match="infinity"):
        transform(np.asarray([[np.inf, 1.0]]), params)
