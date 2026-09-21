"""Five-seed controlled events with frozen point/EWMA and calibrated CUSUM."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .config import config_hash
from .run_logistics_e1b import ROOT, _json, _sha
from .run_finite_memory_e1e import audited_source, METRICS
from .run_temporal_e1d import aggregate, EVENT_METRICS
from .finite_memory import coverage_metrics, coverage_events
from .temporal_reliability import score_temporal
from .cusum_reliability import calibrate_cusum, score_cusum
from .controlled_events import validate_controlled, validation_innovations, controlled_scenario

GROUPS = ["scenario", "method", "temporal", "view", "regime", "window"]
EVENTS = EVENT_METRICS + ["scoreable_fraction", "fully_unavailable", "delay_fraction_capped"]


def differences(frame: pd.DataFrame, keys: list[str], metrics: list[str],
                *, column: str, a_value: str, b_value: str) -> pd.DataFrame:
    """Compute exact-key paired differences and reject mismatched supports."""
    a = frame[frame[column] == a_value].set_index(keys)[metrics].sort_index()
    b = frame[frame[column] == b_value].set_index(keys)[metrics].sort_index()
    if not a.index.equals(b.index):
        raise ValueError("Unmatched paired results.")
    result = (a-b).reset_index()
    result["comparison"] = f"{a_value}_minus_{b_value}"
    return result


def figure(summary: pd.DataFrame, events: pd.DataFrame, out: Path) -> None:
    """Show duration response and tail cost with seed standard deviations."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    rows = summary[(summary.view == "observed") & (summary.regime == "all") & (summary.window == "all")]
    names = ["duration4", "steady_d12_n2_m1", "duration24"]
    for (method, temporal), group in rows.groupby(["method", "temporal"]):
        if not set(names) <= set(group.scenario):
            continue
        points = group.set_index("scenario").loc[names]
        ev = events[(events.method == method) & (events.temporal == temporal)
                    & (events.view == "observed")].set_index("scenario").loc[names]
        for ax, data, metric, label in [
            (axes[0], points, "all_timeline_recall", "Cell recall"),
            (axes[1], ev, "fresh_onset_hit", "Fresh event detection"),
            (axes[2], points, "post_event_fpr", "Post-event clean-cell FPR")]:
            ax.errorbar([4, 12, 24], data[f"{metric}_mean"], yerr=data[f"{metric}_std"], marker="o",
                        capsize=2, label=f"{method.removeprefix('regime_')}/{temporal}")
            ax.set(xlabel="Duration / steps (2 events per sequence)", ylabel=label)
            ax.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    fig.suptitle("New synthetic validation; fixed event starts; same nominal q99; error bars: seed SD")
    fig.savefig(out/"controlled_events.png", dpi=180)
    fig.savefig(out/"controlled_events.pdf")
    plt.close(fig)


def run_controlled(config_path: str | Path, *, output_root: str | Path | None = None,
                   run_id: str | None = None) -> Path:
    """Freeze sources, generate new val-only events, and save complete evidence."""
    config = yaml.safe_load(Path(config_path).read_text())
    source, baseline = (ROOT/config[k] for k in ("source_run", "baseline_run"))
    source_meta, baseline_meta = audited_source(source), audited_source(baseline)
    source_config = yaml.safe_load((source/"config.yaml").read_text())
    validate_controlled(config, source_config)
    if ((ROOT/baseline_meta["source_run"]).resolve() != source.resolve()
            or baseline_meta["source_metadata_sha256"] != _sha(source/"metadata.json")):
        raise ValueError("Baseline provenance mismatch.")
    identifier = run_id or datetime.now(timezone.utc).strftime("controlled_%Y%m%dT%H%M%S%fZ")
    if Path(identifier).name != identifier or identifier in {".", ".."}:
        raise ValueError("Unsafe run_id.")
    out = (Path(output_root) if output_root is not None else ROOT/config["output_root"])/identifier
    out.mkdir(parents=True, exist_ok=False)
    (out/"config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    _json(out/"input_manifest.json", {name: {"path": config[key], "metadata_sha256": _sha(path/"metadata.json"),
           "verified_outputs": meta["output_sha256"]} for name, key, path, meta in [
           ("training_source", "source_run", source, source_meta),
           ("frozen_baselines", "baseline_run", baseline, baseline_meta)]})
    metric_rows, event_rows_all, audits, designs = [], [], [], []
    for seed in config["seeds"]:
        src, old, dest = source/f"seed_{seed}", baseline/f"seed_{seed}", out/f"seed_{seed}"
        dest.mkdir()
        references = json.loads((src/"references.json").read_text())
        scales = json.loads((src/"scales.json").read_text())
        old_models = json.loads((old/"models.json").read_text())
        with np.load(src/"base_data.npz", allow_pickle=False) as base:
            cal = base["splits"] == "train_cal"
            cal_values = base["clean"][cal]
            cal_records = np.repeat(base["regimes"][cal, None], cal_values.shape[1], axis=1)
            cal_ids = base["sequence_ids"][cal].tolist()
        if old_models["calibration_ids"] != cal_ids:
            raise ValueError("Calibration membership mismatch.")
        models = {}
        for method in config["methods"]:
            for temporal, key in (("point", "a1"), ("ewma", "a0.2")):
                model = old_models["models"][f"{method}__{key}"]
                if model["reference"] != references[method] or model["quantile"] != config["threshold_quantile"]:
                    raise ValueError("Frozen reference/calibration budget mismatch.")
                models[f"{method}__{temporal}"] = model
            models[f"{method}__cusum"] = calibrate_cusum(cal_values, cal_records, references[method],
                                                       k=config["cusum_k"], quantile=config["threshold_quantile"])
        for key, model in models.items():
            method, temporal = key.split("__")
            scorer = score_cusum if temporal == "cusum" else score_temporal
            calibrated = scorer(cal_values, cal_records, model)
            for regime in model["thresholds"]:
                for j, threshold in enumerate(model["thresholds"][regime]):
                    selected = cal_records == regime
                    audits.append({"seed": seed, "method": method, "temporal": temporal, "regime": regime,
                        "channel": j, "threshold": threshold, "n_calibration_cells": int(selected.sum()),
                        "calibration_fpr": float(calibrated["alarm"][selected, j].mean())})
        innovations, ids, initial = validation_innovations(source_config, seed, config["validation_namespace"])
        _json(dest/"models.json", {"calibration_ids": cal_ids, "validation_ids": ids.tolist(), "models": models})
        np.savez_compressed(dest/"innovations.npz", innovations=innovations, sequence_ids=ids, initial_regimes=initial)
        for scenario in config["scenarios"]:
            data = controlled_scenario(innovations, ids, initial, source_config, config, scenario, scales, seed)
            directory = dest/scenario["name"]
            directory.mkdir()
            np.savez_compressed(directory/"validation_data.npz", clean=data.clean, observed=data.observed,
                mask=data.mask, true_regimes=data.true_regimes, recorded_regimes=data.recorded_regimes,
                transition=data.transition, sequence_ids=data.sequence_ids)
            _json(directory/"events.json", data.events)
            for event in data.events:
                designs.append({"seed": seed, "scenario": scenario["name"], **event})
            scores = {}
            for view in ("clean", "observed"):
                scored_all = {key: (score_cusum if key.endswith("__cusum") else score_temporal)(
                    getattr(data, view), data.recorded_regimes, model) for key, model in models.items()}
                common = np.logical_and.reduce([s["available"].all(axis=-1) for s in scored_all.values()])
                truth = data.mask if view == "observed" else np.zeros_like(data.mask)
                for key, scored in scored_all.items():
                    method, temporal = key.split("__")
                    for name, values in scored.items():
                        scores[f"{key}__{view}__{name}"] = values
                    event_rows, tail = coverage_events(scored, data.mask, list(data.events),
                                                       post_steps=config["post_event_steps"])
                    for event in event_rows:
                        original = data.events[event["event_index"]]
                        offset = original["offset_from_nearest_transition"]
                        phase = ("steady" if offset is None else
                                 "far" if abs(offset) > source_config["stress"]["transition_window"] else
                                 "before" if offset < 0 else "after" if offset > 0 else "at")
                        event_rows_all.append({"seed": seed, "scenario": scenario["name"], "method": method,
                            "temporal": temporal, "view": view, "onset_phase": phase,
                            "offset_from_nearest_transition": offset,
                            **event, "delay_fraction_capped": event["fresh_delay_capped"]/event["duration"]})
                    for regime in ["all", *source_config["regimes"]]:
                        state = np.ones(truth.shape[:2], bool) if regime == "all" else data.true_regimes == regime
                        for window, selected in (("all", state), ("transition", state & data.transition),
                                                 ("stable", state & ~data.transition)):
                            if selected.any():
                                result = coverage_metrics(truth, scored, selected, common, tail, support="common")
                                metric_rows.append({"seed": seed, "scenario": scenario["name"], "method": method,
                                    "temporal": temporal, "view": view, "regime": regime, "window": window, **result})
            np.savez_compressed(directory/"scores.npz", **scores)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out/"metrics_by_seed.csv", index=False)
    summary = aggregate(metrics, GROUPS, METRICS)
    summary.to_csv(out/"summary.csv", index=False)
    pd.DataFrame(audits).to_csv(out/"calibration_audit.csv", index=False)
    pd.DataFrame(designs).to_csv(out/"event_design.csv", index=False)
    events = pd.DataFrame(event_rows_all)
    events.to_csv(out/"events.csv", index=False)
    event_keys = ["seed", "scenario", "method", "temporal", "view"]
    event_seed = events.groupby(event_keys)[EVENTS].mean().reset_index()
    event_seed = event_seed.merge(events.groupby(event_keys).size().rename("event_channel_count").reset_index(),
                                   on=event_keys, validate="one_to_one")
    event_seed.to_csv(out/"events_by_seed.csv", index=False)
    event_summary = aggregate(event_seed, event_keys[1:], EVENTS)
    event_summary.to_csv(out/"event_summary.csv", index=False)
    phase_keys = event_keys+["onset_phase"]
    phase_seed = events.groupby(phase_keys)[EVENTS].mean().reset_index()
    phase_seed = phase_seed.merge(events.groupby(phase_keys).size().rename("event_channel_count").reset_index(),
                                  on=phase_keys, validate="one_to_one")
    phase_seed.to_csv(out/"event_phase_by_seed.csv", index=False)
    aggregate(phase_seed, phase_keys[1:], EVENTS).to_csv(out/"event_phase_summary.csv", index=False)
    for frame, keys, fields, prefix in [
        (metrics, ["seed", *[g for g in GROUPS if g != "temporal"]], METRICS, "method_paired"),
        (event_seed, ["seed", "scenario", "method", "view"], EVENTS, "event_paired")]:
        diffs = [differences(frame, keys, fields, column="temporal", a_value="cusum", b_value=b)
                 for b in ("point", "ewma")]
        table = pd.concat(diffs, ignore_index=True)
        table.to_csv(out/f"{prefix}_by_seed.csv", index=False)
        aggregate(table, [*keys[1:], "comparison"], fields).to_csv(out/f"{prefix}_summary.csv", index=False)
    # Scenario pairs compare all-time aggregates; transition strata differ for steady/switching.
    full = metrics[(metrics.regime == "all") & (metrics.window == "all")]
    keys = ["seed", "method", "temporal", "view"]
    scenario_deltas = [differences(full, keys, METRICS, column="scenario", a_value=s["name"],
                                   b_value=s["compare_to"]) for s in config["scenarios"] if s["compare_to"]]
    table = pd.concat(scenario_deltas, ignore_index=True)
    table.to_csv(out/"scenario_paired_by_seed.csv", index=False)
    aggregate(table, [*keys[1:], "comparison"], METRICS).to_csv(out/"scenario_paired_summary.csv", index=False)
    figure(summary, event_summary, out)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "src", "tests", "configs"],
                                    cwd=ROOT, text=True).strip()
    _json(out/"metadata.json", {"run_id": identifier, "data_source": "synthetic",
        "evaluation_split": "new_development_validation", "test_evaluated": False, "quality_evaluated": False,
        "code_commit": commit, "implementation_worktree_dirty": bool(dirty), "seeds": config["seeds"],
        "config_sha256": config_hash(config), "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_namespace": config["validation_namespace"],
        "source_sha256": {str(p.relative_to(ROOT)): _sha(p) for p in sorted((ROOT/"src").glob("*.py"))},
        "environment": {"python": platform.python_version(), **{p: importlib.metadata.version(p) for p in
                         ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "PyYAML")}},
        "output_sha256": {str(p.relative_to(out)): _sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
        "limitations": ["synthetic same-family new validation not final blind test", "clean oracle training",
            "continuous CUSUM has no alarm reset; no ARL guarantee", "nominal not realized matched FPR",
            "constrained random starts and ideal abrupt regime transitions", "no quality or real-data evidence"]})
    return out


def main() -> None:
    """Execute the controlled event protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(run_controlled(args.config, output_root=args.output_root, run_id=args.run_id))


if __name__ == "__main__":
    main()
