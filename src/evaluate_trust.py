"""Evaluation tables and diagnostic plotting for synthetic trust scores."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "petrochemical_traceability_mpl"),
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def _validated_binary_inputs(
    corruption_labels: np.ndarray,
    anomaly_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(corruption_labels)
    scores = np.asarray(anomaly_scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("Labels and scores must be one-dimensional with equal shape.")
    if labels.size == 0:
        raise ValueError("Labels and scores must not be empty.")
    if not np.isfinite(scores).all():
        raise ValueError("Anomaly scores must all be finite.")
    unique_labels = set(np.unique(labels).tolist())
    if not unique_labels.issubset({0, 1, False, True}):
        raise ValueError("Corruption labels must be binary.")
    return labels.astype(bool), scores


def binary_detection_metrics(
    corruption_labels: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate binary metrics for the rule ``score > threshold``."""

    labels, scores = _validated_binary_inputs(corruption_labels, anomaly_scores)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    if labels.all() or not labels.any():
        raise ValueError("Both clean and corrupted labels are required for evaluation.")
    predicted = scores > threshold
    true_positive = int(np.sum(predicted & labels))
    false_positive = int(np.sum(predicted & ~labels))
    false_negative = int(np.sum(~predicted & labels))
    true_negative = int(np.sum(~predicted & ~labels))
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_positive_rate = false_positive / (false_positive + true_negative)
    false_negative_rate = false_negative / recall_denominator
    return {
        "num_samples": int(labels.size),
        "num_corrupted": int(labels.sum()),
        "corruption_prevalence": float(labels.mean()),
        "threshold": float(threshold),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(labels.astype(int), scores)),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
    }


def build_detection_metrics_table(
    frame: pd.DataFrame,
    *,
    evaluation_splits: Iterable[str],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Build overall threshold metrics for explicitly allowed splits."""

    rows: list[dict[str, float | int | str]] = []
    for split in evaluation_splits:
        subset = frame.loc[frame["split"] == split]
        if subset.empty:
            raise ValueError(f"No rows found for evaluation split '{split}'.")
        for threshold_name, threshold in thresholds.items():
            metrics = binary_detection_metrics(
                subset["corruption_label"].to_numpy(),
                subset["mahalanobis_distance_squared"].to_numpy(),
                threshold,
            )
            rows.append(
                {
                    "split": split,
                    "threshold_name": threshold_name,
                    "prediction_rule": "mahalanobis_distance_squared > threshold",
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def build_corruption_type_recall_table(
    frame: pd.DataFrame,
    *,
    evaluation_splits: Iterable[str],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Report detection recall separately for every injected corruption type."""

    corruption_types = sorted(
        value for value in frame["corruption_type"].unique() if value != "none"
    )
    rows: list[dict[str, float | int | str]] = []
    for split in evaluation_splits:
        split_frame = frame.loc[frame["split"] == split]
        for corruption_type in corruption_types:
            type_mask = (
                split_frame["corruption_label"].astype(bool)
                & (split_frame["corruption_type"] == corruption_type)
            )
            num_corrupted = int(type_mask.sum())
            if num_corrupted == 0:
                continue
            scores = split_frame["mahalanobis_distance_squared"]
            for threshold_name, threshold in thresholds.items():
                num_detected = int(np.sum(type_mask & (scores > threshold)))
                rows.append(
                    {
                        "split": split,
                        "threshold_name": threshold_name,
                        "corruption_type": corruption_type,
                        "num_corrupted": num_corrupted,
                        "num_detected": num_detected,
                        "recall": num_detected / num_corrupted,
                    }
                )
    return pd.DataFrame(rows)


def build_condition_false_positive_table(
    frame: pd.DataFrame,
    *,
    evaluation_splits: Iterable[str],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Measure false alarms on uncorrupted rows for each clean process condition."""

    rows: list[dict[str, float | int | str]] = []
    for split in evaluation_splits:
        split_frame = frame.loc[
            (frame["split"] == split) & (~frame["corruption_label"].astype(bool))
        ]
        for condition in sorted(split_frame["condition"].unique()):
            condition_frame = split_frame.loc[split_frame["condition"] == condition]
            scores = condition_frame["mahalanobis_distance_squared"]
            for threshold_name, threshold in thresholds.items():
                false_positives = int(np.sum(scores > threshold))
                rows.append(
                    {
                        "split": split,
                        "threshold_name": threshold_name,
                        "condition": condition,
                        "num_uncorrupted": int(len(condition_frame)),
                        "false_positive": false_positives,
                        "false_positive_rate": false_positives / len(condition_frame),
                    }
                )
    return pd.DataFrame(rows)


def plot_distance_trust_timeseries(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    plot_split: str,
    high_threshold_squared: float,
    low_threshold_squared: float,
) -> tuple[Path, str]:
    """Plot a deterministic representative sequence from one evaluation split."""

    candidates = frame.loc[frame["split"] == plot_split]
    if candidates.empty:
        raise ValueError(f"No rows found for plot split '{plot_split}'.")
    counts = (
        candidates.groupby("sequence_id", sort=True)["corruption_label"]
        .sum()
        .sort_values(ascending=False, kind="stable")
    )
    selected_sequence_id = str(counts.index[0])
    selected = candidates.loc[
        candidates["sequence_id"].astype(str) == selected_sequence_id
    ].sort_values("time_index")
    if selected["time_index"].duplicated().any():
        raise ValueError("The selected sequence has duplicate time indices.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    time = selected["time_index"].to_numpy(dtype=int)
    distance = selected["mahalanobis_distance"].to_numpy(dtype=float)
    trust = selected["trust_score"].to_numpy(dtype=float)
    corrupted = selected["corruption_label"].to_numpy(dtype=bool)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    axes[0].plot(time, distance, color="tab:blue", linewidth=1.5, label="Mahalanobis distance")
    axes[0].axhline(
        np.sqrt(high_threshold_squared),
        color="tab:orange",
        linestyle="--",
        label="sqrt(q90)",
    )
    axes[0].axhline(
        np.sqrt(low_threshold_squared),
        color="tab:red",
        linestyle="--",
        label="sqrt(q99)",
    )
    axes[0].scatter(
        time[corrupted],
        distance[corrupted],
        color="black",
        marker="x",
        s=28,
        label="Injected corruption",
        zorder=3,
    )
    axes[0].set_ylabel("Mahalanobis distance")
    axes[0].set_yscale("symlog", linthresh=0.2)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right", ncols=2)

    axes[1].plot(time, trust, color="tab:green", linewidth=1.5, label="Trust score")
    axes[1].scatter(
        time[corrupted],
        trust[corrupted],
        color="black",
        marker="x",
        s=28,
        label="Injected corruption",
        zorder=3,
    )
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("Trust score [0, 1]")
    axes[1].set_xlabel("Time index (abstract sampling step)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")
    fig.suptitle(
        f"Synthetic E1 distance and trust timeline — {selected_sequence_id}",
        fontsize=13,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output, selected_sequence_id
