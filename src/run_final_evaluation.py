"""Once-frozen release evaluation on the retained E1c synthetic test sequences."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import config_hash
from .controlled_events import controlled_scenario, validate_controlled
from .finite_memory import coverage_events, coverage_metrics
from .run_controlled_e1f import EVENTS, GROUPS
from .run_finite_memory_e1e import audited_source, METRICS
from .run_logistics_e1b import ROOT, _json, _sha
from .run_temporal_e1d import aggregate
from .run_transition_e1g import record_windows
from .scoring_component import make_bundle
from .temporal_reliability import score_temporal

MODEL_KEYS = ["regime_marginal__point", "regime_marginal__ewma",
              "regime_conditional__point", "regime_conditional__ewma"]


def retained_innovations(base: Any, config: dict[str, Any],
                         membership: dict[str, list[str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select test by explicit membership and normalize using configured constants."""
    all_ids = base["sequence_ids"].tolist()
    if len(all_ids) != len(set(all_ids)) or set(membership) != {"train_fit", "train_cal", "val", "test"}:
        raise ValueError("Invalid split identities.")
    declared = [i for values in membership.values() for i in values]
    if len(declared) != len(set(declared)) or set(declared) != set(all_ids):
        raise ValueError("Overlapping or incomplete split membership.")
    for split, ids in membership.items():
        if base["sequence_ids"][base["splits"] == split].tolist() != ids:
            raise ValueError("Stored split membership mismatch.")
    test = base["splits"] == "test"
    ids, initial = base["sequence_ids"][test], base["regimes"][test]
    clean = base["clean"][test]
    expected = len(config["regimes"])*config["sequences_per_regime"]["test"]
    if clean.shape != (expected, config["sequence_length"], len(config["features"])) or not np.isfinite(clean).all():
        raise ValueError("Invalid retained trajectories.")
    z = np.empty_like(clean)
    for i, regime in enumerate(initial):
        profile = config["regimes"][regime]
        z[i] = (clean[i]-profile["mean"])/profile["std"]
    return z, ids, initial


def validate_release(config: dict[str, Any]) -> None:
    """Enforce the predeclared release choices; no test-driven method selection."""
    expected = {"release": "scoring_v0.1_synthetic", "seeds": [42, 43, 44, 45, 46],
                "models": MODEL_KEYS, "default_model": "regime_marginal__ewma", "demo_seed": 42,
                "test_event_namespace": 20260927, "record_window_steps": 12}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Release choices differ from the frozen protocol.")


def run_final(config_path: str | Path, *, output_root: str | Path | None = None,
              run_id: str | None = None) -> Path:
    """Evaluate all frozen methods and save auditable test scores without fitting."""
    config = yaml.safe_load(Path(config_path).read_text())
    validate_release(config)
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "src", "tests", "configs",
                                    config["protocol"]], cwd=ROOT, text=True).strip()
    if dirty:
        raise ValueError("Commit implementation, tests, config and protocol before test evaluation.")
    training, models = (ROOT/config[k] for k in ("training_run", "model_run"))
    train_meta, model_meta = audited_source(training), audited_source(models)
    source_config = yaml.safe_load((training/"config.yaml").read_text())
    event_config = yaml.safe_load((models/"config.yaml").read_text())
    validate_controlled(event_config, source_config)
    manifest = json.loads((models/"input_manifest.json").read_text())["training_source"]
    if ((ROOT/manifest["path"]).resolve() != training.resolve() or
            manifest["metadata_sha256"] != _sha(training/"metadata.json")):
        raise ValueError("Model/training provenance mismatch.")
    event_config = deepcopy(event_config)
    event_config["validation_namespace"] = config["test_event_namespace"]
    identifier = run_id or datetime.now(timezone.utc).strftime("final_%Y%m%dT%H%M%S%fZ")
    if Path(identifier).name != identifier or identifier in {".", ".."}:
        raise ValueError("Unsafe run identifier.")
    out = (Path(output_root) if output_root is not None else ROOT/config["output_root"])/identifier
    out.mkdir(parents=True, exist_ok=False)
    (out/"config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (out/"event_config.yaml").write_text(yaml.safe_dump(event_config, sort_keys=False))
    (out/"protocol.md").write_bytes((ROOT/config["protocol"]).read_bytes())
    _json(out/"input_manifest.json", {k: {"path": config[k], "metadata_sha256": _sha(p/"metadata.json"),
           "verified_outputs": m["output_sha256"]} for k, p, m in
          (("training_run", training, train_meta), ("model_run", models, model_meta))})
    rows, event_rows, identity_rows = [], [], []
    for seed in config["seeds"]:
        train_dir, model_dir, dest = training/f"seed_{seed}", models/f"seed_{seed}", out/f"seed_{seed}"
        dest.mkdir()
        saved = json.loads((model_dir/"models.json").read_text())
        membership = json.loads((train_dir/"membership.json").read_text())
        if (saved["calibration_ids"] != membership["train_cal"] or
                set(saved["validation_ids"]) & set(membership["test"])):
            raise ValueError("Calibration/development/test identities overlap or disagree.")
        references = json.loads((train_dir/"references.json").read_text())
        bundles = {}
        for key in config["models"]:
            model = saved["models"][key]
            method, temporal = key.split("__")
            if model["reference"] != references[method] or model["alpha"] != (1. if temporal == "point" else .2):
                raise ValueError("Frozen model mismatch.")
            bundle = make_bundle(model, source_config["features"], source_config["units"],
                                 {"seed": seed, "source": str((model_dir/"models.json").relative_to(ROOT)),
                                  "source_sha256": _sha(model_dir/"models.json"), "model_key": key,
                                  "fit_ids": membership["train_fit"], "calibration_ids": membership["train_cal"]})
            bundles[key] = bundle
            _json(dest/f"{key}.json", bundle)
        with np.load(train_dir/"base_data.npz", allow_pickle=False) as base:
            z, ids, initial = retained_innovations(base, source_config, membership)
        scales = json.loads((train_dir/"scales.json").read_text())
        _json(dest/"membership.json", {**membership, "e1f_development": saved["validation_ids"]})
        identity_rows.append({"seed": seed, "test_sequences": len(ids), "length": z.shape[1],
                              "split_disjoint": True, "parameters_refit": False})
        for scenario in event_config["scenarios"]:
            data = controlled_scenario(z, ids, initial, source_config, event_config, scenario, scales, seed, split="test")
            directory = dest/scenario["name"]
            directory.mkdir()
            np.savez_compressed(directory/"test_data.npz", clean=data.clean, observed=data.observed,
                mask=data.mask, true_regimes=data.true_regimes, recorded_regimes=data.recorded_regimes,
                transition=data.transition, sequence_ids=data.sequence_ids)
            _json(directory/"events.json", data.events)
            record_window = record_windows(data.recorded_regimes, config["record_window_steps"])
            scores_to_save = {}
            for view in ("clean", "observed"):
                values = getattr(data, view)
                truth = data.mask if view == "observed" else np.zeros_like(data.mask)
                all_scores = {key: score_temporal(values, data.recorded_regimes, bundle["model"])
                              for key, bundle in bundles.items()}
                common = np.logical_and.reduce([s["available"].all(axis=-1) for s in all_scores.values()])
                for key, scored in all_scores.items():
                    method, temporal = key.split("__")
                    for field, value in scored.items():
                        scores_to_save[f"{key}__{view}__{field}"] = value
                    event_metrics, tail = coverage_events(scored, data.mask, list(data.events),
                                                          post_steps=event_config["post_event_steps"])
                    for event in event_metrics:
                        event_rows.append({"seed": seed, "scenario": scenario["name"], "method": method,
                            "temporal": temporal, "view": view, **event,
                            "delay_fraction_capped": event["fresh_delay_capped"]/event["duration"]})
                    for regime in ["all", *source_config["regimes"]]:
                        state = np.ones(truth.shape[:2], bool) if regime == "all" else data.true_regimes == regime
                        for window, mask in (("all", np.ones_like(state)), ("transition", data.transition),
                                ("stable", ~data.transition), ("record_transition", record_window),
                                ("record_stable", ~record_window)):
                            selected = state & mask
                            if selected.any():
                                rows.append({"seed": seed, "scenario": scenario["name"], "method": method,
                                    "temporal": temporal, "view": view, "regime": regime, "window": window,
                                    **coverage_metrics(truth, scored, selected, common, tail, support="common")})
            np.savez_compressed(directory/"scores.npz", **scores_to_save)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out/"metrics_by_seed.csv", index=False)
    aggregate(metrics, GROUPS, METRICS).to_csv(out/"summary.csv", index=False)
    events = pd.DataFrame(event_rows)
    events.to_csv(out/"events.csv", index=False)
    keys = ["seed", "scenario", "method", "temporal", "view"]
    event_seed = events.groupby(keys)[EVENTS].mean().reset_index()
    event_seed.to_csv(out/"events_by_seed.csv", index=False)
    aggregate(event_seed, keys[1:], EVENTS).to_csv(out/"event_summary.csv", index=False)
    pd.DataFrame(identity_rows).to_csv(out/"identity_audit.csv", index=False)
    _json(out/"metadata.json", {"run_id": identifier, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "synthetic", "evaluation_split": "retained_e1c_test", "test_evaluated": True,
        "quality_evaluated": False, "parameters_refit": False, "default_selected_before_test": config["default_model"],
        "seeds": config["seeds"], "independent_test_sequences_per_seed": 12,
        "paired_scenario_count": len(event_config["scenarios"]), "config_sha256": config_hash(config),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "implementation_worktree_dirty": False, "protocol_sha256": _sha(ROOT/config["protocol"]),
        "source_sha256": {str(p.relative_to(ROOT)): _sha(p) for p in sorted((ROOT/"src").glob("*.py"))},
        "environment": {"python": platform.python_version(), **{k: importlib.metadata.version(k) for k in
                        ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "PyYAML")}},
        "limitations": ["same-family synthetic retained data, not external industrial validation",
            "public stored test, not externally blinded", "single-channel biases only in final protocol",
            "clean oracle training", "abstract time and abrupt regime changes", "no quality prediction",
            "test consumed; do not tune or select methods with these results"],
        "output_sha256": {str(p.relative_to(out)): _sha(p) for p in sorted(out.rglob("*")) if p.is_file()}})
    return out


def main() -> None:
    """Run the precommitted final evaluation into a new output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scoring_release.yaml")
    parser.add_argument("--output-root")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(run_final(args.config, output_root=args.output_root, run_id=args.run_id))


if __name__ == "__main__":
    main()
