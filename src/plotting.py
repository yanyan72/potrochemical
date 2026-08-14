"""Plotting helpers for simulator and corruption quality assurance."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "petrochemical_traceability_mpl"),
)

import matplotlib.pyplot as plt
import numpy as np


def plot_corruption_examples(
    clean_values: np.ndarray,
    observed_values: np.ndarray,
    events: tuple[dict[str, Any], ...],
    output_path: str | Path,
    *,
    context_steps: int = 6,
) -> Path:
    """Plot one representative event for each configured corruption type."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected_events: list[dict[str, Any]] = []
    corruption_types = (
        "spike",
        "bias",
        "drift",
        "missing",
        "random_replacement",
    )
    for corruption_type in corruption_types:
        candidates = [
            event for event in events if event["corruption_type"] == corruption_type
        ]
        event = next(
            (
                candidate
                for candidate in candidates
                if not any(
                    other["event_id"] != candidate["event_id"]
                    and other["sequence_index"] == candidate["sequence_index"]
                    and other["feature_index"] == candidate["feature_index"]
                    and other["start_time_index"]
                    < candidate["end_time_index_exclusive"] + context_steps
                    and other["end_time_index_exclusive"]
                    > candidate["start_time_index"] - context_steps
                    for other in events
                )
            ),
            candidates[0] if candidates else None,
        )
        if event is None:
            raise ValueError(f"No {corruption_type} event available for plotting.")
        selected_events.append(event)

    fig, axes = plt.subplots(5, 1, figsize=(11, 14), constrained_layout=True)
    for axis, event in zip(axes, selected_events, strict=True):
        sequence_index = int(event["sequence_index"])
        feature_index = int(event["feature_index"])
        start = int(event["start_time_index"])
        end = int(event["end_time_index_exclusive"])
        window_start = max(0, start - context_steps)
        window_end = min(clean_values.shape[1], end + context_steps)
        time = np.arange(window_start, window_end)
        axis.plot(
            time,
            clean_values[sequence_index, window_start:window_end, feature_index],
            label="Clean truth",
            color="tab:blue",
            linewidth=1.8,
        )
        axis.plot(
            time,
            observed_values[
                sequence_index, window_start:window_end, feature_index
            ],
            label="Corrupted observation",
            color="tab:red",
            linewidth=1.4,
            marker=(
                "o"
                if event["corruption_type"] in {"spike", "random_replacement"}
                else None
            ),
            markersize=3,
        )
        axis.axvspan(
            start - 0.5,
            end - 0.5,
            color="tab:orange",
            alpha=0.20,
            label="Injected mask",
        )
        axis.set_title(
            f"{event['corruption_type'].upper()} — "
            f"{event['sequence_id']} / {event['corruption_feature']}"
        )
        axis.set_ylabel("Native unit")
        axis.grid(alpha=0.25)
    axes[0].legend(loc="best", ncols=3)
    axes[-1].set_xlabel("Time index (abstract sampling step)")
    fig.suptitle(
        "Milestone 0: reproducible artificial sensor corruption examples",
        fontsize=14,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output
