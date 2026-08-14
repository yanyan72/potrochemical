"""Command-line pipeline for Milestone 0 simulation and corruption."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import config_hash, load_config
from .corruption import CorruptionResult, inject_corruptions
from .plotting import plot_corruption_examples
from .simulator import FEATURE_UNITS, SimulationResult, generate_clean_sequences


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_version(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return completed.stdout.strip()
    return "unavailable_no_git_repository"


def _row_annotations(
    simulation: SimulationResult,
    corruption: CorruptionResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_sequences, sequence_length, _ = corruption.mask.shape
    row_mask = corruption.mask.any(axis=2)
    corruption_type = np.full(
        (n_sequences, sequence_length),
        "none",
        dtype=object,
    )
    corruption_feature = np.full(
        (n_sequences, sequence_length),
        "none",
        dtype=object,
    )
    corruption_magnitude = np.zeros((n_sequences, sequence_length), dtype=float)
    for event in corruption.events:
        sequence_index = int(event["sequence_index"])
        feature_index = int(event["feature_index"])
        start = int(event["start_time_index"])
        end = int(event["end_time_index_exclusive"])
        corruption_type[sequence_index, start:end] = event["corruption_type"]
        corruption_feature[sequence_index, start:end] = event[
            "corruption_feature"
        ]
        corruption_magnitude[sequence_index, start:end] = (
            corruption.observed_values[sequence_index, start:end, feature_index]
            - simulation.clean_values[sequence_index, start:end, feature_index]
        )
    return row_mask, corruption_type, corruption_feature, corruption_magnitude


def build_long_table(
    simulation: SimulationResult,
    corruption: CorruptionResult,
) -> pd.DataFrame:
    """Combine clean truth, observations, and row-level corruption labels."""

    frame = simulation.frame.copy()
    for feature_index, feature_name in enumerate(simulation.feature_names):
        frame[f"{feature_name}_obs"] = corruption.observed_values[
            :, :, feature_index
        ].reshape(-1)
    row_mask, row_type, row_feature, row_magnitude = _row_annotations(
        simulation,
        corruption,
    )
    frame["is_corrupted"] = row_mask.reshape(-1)
    frame["corruption_type"] = row_type.reshape(-1)
    frame["corruption_feature"] = row_feature.reshape(-1)
    frame["corruption_magnitude"] = row_magnitude.reshape(-1)
    ordered_columns = [
        "sequence_id",
        "time_index",
        "split",
        "condition",
        *[f"{name}_clean" for name in simulation.feature_names],
        *[f"{name}_obs" for name in simulation.feature_names],
        "quality",
        "is_corrupted",
        "corruption_type",
        "corruption_feature",
        "corruption_magnitude",
    ]
    return frame[ordered_columns]


def _make_run_id(configuration_hash: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"milestone0_{timestamp}_{configuration_hash[:8]}"


def run_generation(
    config_path: str | Path,
    *,
    data_root: str | Path | None = None,
    figures_root: str | Path | None = None,
    run_id: str | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Run the complete Milestone 0 pipeline and return saved path metadata."""

    project_root_path = Path(project_root).resolve()
    config_path_obj = Path(config_path).resolve()
    config = load_config(config_path_obj)
    configuration_hash = config_hash(config)
    actual_run_id = run_id or _make_run_id(configuration_hash)

    configured_data_root = Path(config["output"]["data_root"])
    configured_figures_root = Path(config["output"]["figures_root"])
    data_root_path = Path(data_root) if data_root is not None else configured_data_root
    figures_root_path = (
        Path(figures_root) if figures_root is not None else configured_figures_root
    )
    if not data_root_path.is_absolute():
        data_root_path = project_root_path / data_root_path
    if not figures_root_path.is_absolute():
        figures_root_path = project_root_path / figures_root_path
    run_data_dir = data_root_path / actual_run_id
    run_figure_dir = figures_root_path / actual_run_id
    if run_data_dir.exists() or run_figure_dir.exists():
        raise FileExistsError(
            f"Run ID already exists and will not be overwritten: {actual_run_id}"
        )
    run_data_dir.mkdir(parents=True, exist_ok=False)
    run_figure_dir.mkdir(parents=True, exist_ok=False)

    simulation = generate_clean_sequences(config)
    sequence_split = (
        simulation.frame[["sequence_id", "split"]]
        .drop_duplicates("sequence_id")
        .set_index("sequence_id")
        .reindex(simulation.sequence_ids)["split"]
        .to_numpy()
    )
    train_indices = np.flatnonzero(sequence_split == "train")
    if train_indices.size == 0:
        raise RuntimeError("At least one clean training sequence is required.")
    corruption_seed = int(config["project"]["seed"]) + int(
        config["corruption"]["seed_offset"]
    )
    corruption = inject_corruptions(
        simulation.clean_values,
        simulation.clean_values[train_indices],
        simulation.sequence_ids,
        simulation.feature_names,
        config["corruption"],
        seed=corruption_seed,
    )
    long_table = build_long_table(simulation, corruption)

    csv_path = run_data_dir / "simulated_timeseries.csv"
    mask_path = run_data_dir / "corruption_mask.npz"
    events_path = run_data_dir / "corruption_events.json"
    metadata_path = run_data_dir / "metadata.json"
    config_snapshot_path = run_data_dir / "config_snapshot.yaml"
    figure_path = run_figure_dir / "corruption_examples.png"

    long_table.to_csv(csv_path, index=False, float_format="%.17g")
    np.savez_compressed(
        mask_path,
        mask=corruption.mask,
        sequence_ids=np.asarray(simulation.sequence_ids),
        feature_names=np.asarray(simulation.feature_names),
    )
    with events_path.open("w", encoding="utf-8") as handle:
        json.dump(corruption.events, handle, ensure_ascii=False, indent=2)
    with config_snapshot_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    plot_corruption_examples(
        simulation.clean_values,
        corruption.observed_values,
        corruption.events,
        figure_path,
    )

    row_mask = corruption.mask.any(axis=2)
    metadata: dict[str, Any] = {
        "dataset_id": actual_run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "synthetic",
        "seed": int(config["project"]["seed"]),
        "corruption_seed": corruption_seed,
        "corruption_reference_split": "clean_train_only",
        "config_sha256": configuration_hash,
        "data_csv_sha256": _sha256_file(csv_path),
        "code_version": _code_version(project_root_path),
        "time_unit": "abstract_sampling_step",
        "feature_units": FEATURE_UNITS,
        "num_sequences": len(simulation.sequence_ids),
        "sequence_length": int(config["data"]["sequence_length"]),
        "num_rows": len(long_table),
        "num_features": len(simulation.feature_names),
        "requested_corruption_ratio": float(config["corruption"]["ratio"]),
        "achieved_corruption_ratio": float(row_mask.mean()),
        "corrupted_rows": int(row_mask.sum()),
        "event_count": len(corruption.events),
        "event_count_by_type": {
            name: sum(
                event["corruption_type"] == name for event in corruption.events
            )
            for name in config["corruption"]["types"]
        },
        "corrupted_row_count_by_type": {
            str(name): int(count)
            for name, count in long_table.loc[
                long_table["is_corrupted"], "corruption_type"
            ].value_counts().sort_index().items()
        },
        "split_sequence_counts": {
            str(name): int(count)
            for name, count in long_table.groupby("split")[
                "sequence_id"
            ].nunique().items()
        },
        "condition_sequence_counts": {
            str(name): int(count)
            for name, count in long_table.groupby("condition")[
                "sequence_id"
            ].nunique().items()
        },
        "files": {
            "data_csv": str(csv_path.resolve()),
            "mask_npz": str(mask_path.resolve()),
            "events_json": str(events_path.resolve()),
            "config_snapshot": str(config_snapshot_path.resolve()),
            "example_figure": str(figure_path.resolve()),
            "metadata_json": str(metadata_path.resolve()),
        },
        "scientific_scope_note": (
            "Synthetic Milestone 0 data only; no industrial performance, "
            "trust model, forecasting model, physics model, blockchain, or UI."
        ),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reproducible synthetic time series with labeled corruption."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--data-root", help="Optional output data root override.")
    parser.add_argument("--figures-root", help="Optional figure root override.")
    parser.add_argument("--run-id", help="Optional unique run ID; existing runs fail.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = _parse_args()
    metadata = run_generation(
        args.config,
        data_root=args.data_root,
        figures_root=args.figures_root,
        run_id=args.run_id,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
