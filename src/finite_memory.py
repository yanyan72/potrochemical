"""Causal full-window signed residual averages and coverage-aware evaluation."""
from __future__ import annotations

from copy import deepcopy
from typing import Any
import numpy as np

from .temporal_reliability import signed_residuals, event_evaluation
from .run_logistics_robustness import stress_metrics


def causal_sma(residuals: np.ndarray, records: np.ndarray, window_size: int) -> np.ndarray:
    """Average W consecutive same-record finite residuals; otherwise return NaN.

    Sequence boundaries, recorded-regime changes and per-channel gaps reset the
    required history. No padded zeros, partial windows or future values are used.
    """
    values = np.asarray(residuals, dtype=float)
    if (values.ndim != 3 or records.shape != values.shape[:2]
            or type(window_size) is not int or window_size < 1 or np.isinf(values).any()):
        raise ValueError("Require sequence arrays and a positive integer window.")
    result = np.full_like(values, np.nan)
    counts = np.zeros((values.shape[0], values.shape[2]), dtype=int)
    for t in range(values.shape[1]):
        if t:
            counts[records[:, t] != records[:, t-1]] = 0
        valid = np.isfinite(values[:, t])
        counts = np.where(valid, np.minimum(counts + 1, window_size), 0)
        if t >= window_size - 1:
            means = values[:, t-window_size+1:t+1].mean(axis=1)
            result[:, t] = np.where(counts == window_size, means, np.nan)
    return result


def calibrate_sma(values: np.ndarray, records: np.ndarray, reference: dict[str, Any],
                  *, window_size: int, quantile: float) -> dict[str, Any]:
    """Fit per-regime/channel thresholds only on scoreable calibration windows."""
    if not 0 < quantile < 1:
        raise ValueError("Invalid quantile.")
    filtered = causal_sma(signed_residuals(values, records, reference), records, window_size)
    thresholds, counts = {}, {}
    for regime in reference["references"]:
        selected = np.abs(filtered[records == regime])
        thresholds[regime], counts[regime] = [], []
        for j in range(values.shape[-1]):
            sample = selected[:, j][np.isfinite(selected[:, j])]
            if len(sample) < 2:
                raise ValueError("Insufficient complete calibration windows.")
            threshold = float(np.quantile(sample, quantile))
            if not np.isfinite(threshold) or threshold <= 0:
                raise ValueError("Degenerate window threshold.")
            thresholds[regime].append(threshold)
            counts[regime].append(len(sample))
    return {"reference": deepcopy(reference), "window_size": window_size, "quantile": quantile,
            "warmup": "full_window_required", "reset_on_record_change": True,
            "reset_on_gap": "per_channel", "thresholds": thresholds, "calibration_counts": counts}


def score_sma(values: np.ndarray, records: np.ndarray, model: dict[str, Any]) -> dict[str, np.ndarray]:
    """Return signed-mean consistency scores with explicit startup unavailability."""
    filtered = causal_sma(signed_residuals(values, records, model["reference"]),
                          records, model["window_size"])
    ratio = np.full_like(filtered, np.nan)
    for regime, threshold in model["thresholds"].items():
        selected = records == regime
        ratio[selected] = np.abs(filtered[selected]) / threshold
    available = np.isfinite(ratio)
    with np.errstate(over="ignore"):
        reliability = np.exp(-.5 * ratio ** 2)
    return {"deviation_ratio": ratio, "reliability": reliability, "available": available,
            "alarm": available & (ratio > 1.)}


def coverage_metrics(truth: np.ndarray, scored: dict[str, np.ndarray], selected: np.ndarray,
                     common_rows: np.ndarray, tail: np.ndarray, *, support: str) -> dict[str, Any]:
    """Compare common/own supports while keeping whole-timeline miss accounting."""
    if support not in {"own", "common"}:
        raise ValueError("Unknown support.")
    rows = selected & (common_rows if support == "common" else True)
    result = stress_metrics(truth[rows], {k: v[rows] for k, v in scored.items()})
    full_truth, full_alarm = truth[selected], scored["alarm"][selected]
    tp = int((full_truth & full_alarm).sum())
    fp = int((~full_truth & full_alarm).sum())
    fn = int((full_truth & ~full_alarm).sum())
    positives = int(full_truth.sum())
    previous = np.concatenate((np.zeros_like(truth[:, :1]), scored["alarm"][:, :-1]), axis=1)
    onsets = scored["alarm"] & ~previous & ~truth
    # All comparison diagnostics use the same complete-row support as point metrics.
    eligible = rows & scored["available"].all(axis=-1)
    cell_support = np.broadcast_to(eligible[:, :, None], truth.shape)
    tail_support = tail & cell_support
    negatives = cell_support & ~truth
    result.update(
        selected_cells=int(full_truth.size),
        native_score_coverage=float(scored["available"][selected].mean()),
        common_row_fraction=float(common_rows[selected].mean()),
        all_timeline_recall=tp/positives if positives else float("nan"),
        all_timeline_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else float("nan"),
        unavailable_positive_fraction=float((full_truth & ~scored["available"][selected]).sum()/positives)
            if positives else float("nan"),
        post_event_cells=int(tail_support.sum()),
        post_event_fpr=float(scored["alarm"][tail_support].mean()) if tail_support.any() else float("nan"),
        false_onsets_per_1000_clean_cells=float(1000*onsets[negatives].sum()/negatives.sum())
            if negatives.any() else float("nan"),
    )
    return result


def coverage_events(scored: dict[str, np.ndarray], truth: np.ndarray,
                    events: list[dict[str, Any]], *, post_steps: int) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Evaluate all events without discarding partially/fully unavailable windows."""
    rows, tail = event_evaluation(scored["alarm"], truth, events, post_steps=post_steps)
    for row in rows:
        available = scored["available"][row["sequence_index"], row["start"]:row["end_exclusive"], row["channel"]]
        row["scoreable_fraction"] = float(available.mean())
        row["fully_unavailable"] = not bool(available.any())
    return rows, tail
