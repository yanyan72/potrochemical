"""Tests for synthetic E1 trust-detection evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate_trust import (
    binary_detection_metrics,
    build_condition_false_positive_table,
    build_corruption_type_recall_table,
    build_detection_metrics_table,
)


def test_binary_detection_metrics_matches_known_confusion_matrix() -> None:
    metrics = binary_detection_metrics(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.1, 0.8, 0.9, 0.2]),
        threshold=0.5,
    )
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["false_positive_rate"] == pytest.approx(0.5)
    assert metrics["false_negative_rate"] == pytest.approx(0.5)


def test_tables_honor_evaluation_splits_and_report_types_and_conditions() -> None:
    frame = pd.DataFrame(
        {
            "split": ["train"] * 4 + ["val"] * 4 + ["test"] * 4,
            "condition": ["normal", "normal", "hot", "hot"] * 3,
            "corruption_label": [0, 1, 0, 1] * 3,
            "corruption_type": ["none", "spike", "none", "drift"] * 3,
            "mahalanobis_distance_squared": [0.1, 2.0, 3.0, 4.0] * 3,
        }
    )
    thresholds = {"q90": 1.0, "q99": 3.5}

    overall = build_detection_metrics_table(
        frame,
        evaluation_splits=("train", "val"),
        thresholds=thresholds,
    )
    by_type = build_corruption_type_recall_table(
        frame,
        evaluation_splits=("train", "val"),
        thresholds=thresholds,
    )
    by_condition = build_condition_false_positive_table(
        frame,
        evaluation_splits=("train", "val"),
        thresholds=thresholds,
    )

    assert set(overall["split"]) == {"train", "val"}
    assert "test" not in set(overall["split"])
    assert set(overall["threshold_name"]) == {"q90", "q99"}
    assert set(by_type["corruption_type"]) == {"spike", "drift"}
    assert set(by_condition["condition"]) == {"normal", "hot"}
    hot_q90 = by_condition.loc[
        (by_condition["split"] == "val")
        & (by_condition["condition"] == "hot")
        & (by_condition["threshold_name"] == "q90")
    ].iloc[0]
    assert hot_q90["false_positive_rate"] == pytest.approx(1.0)


def test_binary_metrics_reject_degenerate_or_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="Both clean and corrupted"):
        binary_detection_metrics(np.zeros(3), np.arange(3.0), threshold=1.0)
    with pytest.raises(ValueError, match="equal shape"):
        binary_detection_metrics(np.zeros(3), np.arange(2.0), threshold=1.0)
    with pytest.raises(ValueError, match="finite"):
        binary_detection_metrics(
            np.asarray([0, 1]),
            np.asarray([0.0, np.nan]),
            threshold=1.0,
        )
