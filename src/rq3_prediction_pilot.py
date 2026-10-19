"""Frozen five-seed RQ3 ridge development study; no test prediction interface."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.linalg import eigh

from .rq3_paired_data import load_inputs
from .rq3_release_audit import sha256, strict_audit
from .sensor_reliability import fit_sensor_reference
from .temporal_reliability import calibrate_temporal, score_temporal

GROUPS = ("mean", "B0", "B1", "B2", "B3", "B2Q", "B3Q", "B2R")
SOURCE_FILES = ("rq3_prediction_pilot.py", "rq3_release_audit.py", "rq3_paired_data.py",
                "sensor_reliability.py", "temporal_reliability.py", "logistics_simulator.py", "corruption.py")


def write_json(path: Path, obj: Any) -> None:
    """Write finite JSON metadata or parameters."""
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Persist a nonempty, uniformly shaped result table."""
    if not rows:
        raise ValueError("empty result table")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def development_inputs(root: Path, seed: int, split: str, **view: Any) -> dict[str, np.ndarray]:
    """Allow only reference, training and development validation prediction inputs."""
    if split not in ("reference_fit", "reference_cal", "train", "val"):
        raise ValueError("prediction pilot excludes test")
    return load_inputs(root, seed, split, **view)


def targets(root: Path, seed: int, split: str, ids: np.ndarray,
            names: list[str]) -> np.ndarray:
    """Load ID-aligned development targets, never hidden truth or test targets."""
    if split not in ("train", "val"):
        raise ValueError("target loader excludes test and references")
    with (root / f"seed_{seed}" / split / "targets.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if [r["sequence_id"] for r in rows] != list(ids):
        raise ValueError("target ID mismatch")
    y = np.asarray([[float(r[n]) for n in names] for r in rows])
    if not np.isfinite(y).all():
        raise ValueError("nonfinite target")
    return y


def records(view: dict[str, np.ndarray], regimes: list[str]) -> np.ndarray:
    """Convert frozen integer records to the existing scorer's string labels."""
    r = view["recorded_regime"]
    if not np.isin(r, np.arange(len(regimes))).all():
        raise ValueError("unknown frozen regime")
    return np.asarray(regimes)[r]


def fit_score(root: Path, seed: int, regimes: list[str], cfg: dict[str, Any]) -> dict[str, Any]:
    """Fit normal reference and calibrate on disjoint visible-prefix sequences."""
    fit = development_inputs(root, seed, "reference_fit")
    cal = development_inputs(root, seed, "reference_cal")
    ref = fit_sensor_reference(
        fit["observations"].reshape(-1, 3), records(fit, regimes).ravel(),
        cal["observations"].reshape(-1, 3), records(cal, regimes).ravel(),
        method=cfg["score_method"], single_regime=regimes[0], quantile=cfg["quantile"])
    return calibrate_temporal(cal["observations"], records(cal, regimes), ref,
                              alpha=cfg["ewma_alpha"], quantile=cfg["quantile"],
                              reset_on_record_change=True)


def score_view(view: dict[str, np.ndarray], regimes: list[str], model: dict[str, Any]) -> dict[str, np.ndarray]:
    """Compute causal scores with only the allowed visible observations/records."""
    return score_temporal(view["observations"], records(view, regimes), model)


def fit_preprocessor(train: dict[str, np.ndarray]) -> dict[str, list[float]]:
    """Fit per-channel imputation and nonlinear-feature scales on original train."""
    x = train["observations"].reshape(-1, 3)
    if not np.isfinite(x).all():
        raise ValueError("frozen training inputs must be finite")
    return {"median": np.median(x, axis=0).tolist(), "mean": x.mean(axis=0).tolist(),
            "scale": x.std(axis=0).tolist()}


def random_scores(ids: np.ndarray, bank: np.ndarray, seed: int, control_seed: int) -> np.ndarray:
    """Sample label-independent train score trajectories using stable batch keys."""
    index = [int.from_bytes(hashlib.sha256(f"{control_seed}:{seed}:{sid}".encode()).digest()[:8],
                            "big") % len(bank) for sid in ids]
    return bank[index].copy()


def features(view: dict[str, np.ndarray], scores: dict[str, np.ndarray],
             prep: dict[str, Any], group: str, random_block: np.ndarray | None = None) -> np.ndarray:
    """Flatten a fixed 118-step window; no labels, IDs or event fields enter X."""
    if group not in GROUPS or group == "mean":
        raise ValueError("unknown learned feature group")
    x = view["observations"]
    missing = ~np.isfinite(x)
    filled = np.where(missing, np.asarray(prep["median"]), x)
    parts = [filled, missing.astype(float)]
    if group != "B0":
        parts.append(np.eye(3)[view["recorded_regime"]])
    if group not in ("B0", "B1"):
        parts.append(scores["available"].astype(float))
    if group in ("B2Q", "B3Q"):
        parts.append(((filled - prep["mean"]) / prep["scale"]) ** 2)
    if group in ("B3", "B3Q"):
        parts.append(np.where(scores["available"], scores["reliability"], 0.))
    if group == "B2R":
        if random_block is None or random_block.shape != x.shape:
            raise ValueError("random control requires matched trajectory block")
        parts.append(np.where(scores["available"], random_block, 0.))
    result = np.concatenate(parts, axis=2).reshape(len(x), -1)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite feature")
    return result


def ridge_path(x: np.ndarray, y: np.ndarray, alphas: list[float]) -> dict[str, np.ndarray]:
    """Compute centered ridge candidates from one dual eigendecomposition.

    The objective is ||Y-XW||^2 + alpha*||W||^2 with an unpenalized intercept.
    Target columns share X but can subsequently select different alpha values.
    """
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("invalid ridge inputs")
    if any(a <= 0 or not np.isfinite(a) for a in alphas):
        raise ValueError("ridge alphas must be positive")
    center, scale = x.mean(axis=0), x.std(axis=0)
    constant = scale < 1e-12
    scale[constant] = 1.
    z = (x - center) / scale
    intercept = y.mean(axis=0)
    eigen, vectors = eigh(z @ z.T, check_finite=True)
    if eigen.min() < -1e-7 * max(1., eigen.max()):
        raise ValueError("non-PSD design kernel")
    eigen = np.maximum(eigen, 0.)
    projection = vectors.T @ (y - intercept)
    weights = np.stack([z.T @ (vectors @ (projection / (eigen[:, None] + a))) for a in alphas])
    return {"center": center, "scale": scale, "weights": weights,
            "intercept": intercept, "constant": constant}


def predict(x: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    """Apply train-fitted standardization and an archived ridge coefficient matrix."""
    return ((x - model["center"]) / model["scale"]) @ model["weights"] + model["intercept"]


def select_model(path: dict[str, np.ndarray], val_x: np.ndarray, val_y: np.ndarray,
                 alphas: list[float]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Select each target's alpha only by original validation MAE, stable ties."""
    losses = np.stack([np.abs(predict(val_x, {**path, "weights": w}) - val_y).mean(axis=0)
                       for w in path["weights"]])
    selected = losses.argmin(axis=0)
    model = {k: path[k] for k in ("center", "scale", "intercept")}
    model["weights"] = np.column_stack([path["weights"][i, :, target] for target, i in enumerate(selected)])
    model["alphas"] = np.asarray(alphas)[selected]
    return model, losses


def view_specs(cfg: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Enumerate paired amplitude conditions and separate record-delay pressures."""
    views: list[tuple[str, dict[str, Any]]] = [("original", {})]
    for kind in ("bias", "drift"):
        views.extend((f"{kind}_{mag:g}", {"kind": kind, "magnitude": mag}) for mag in cfg["fault_magnitudes"])
    views.append(("missing", {"kind": "missing"}))
    views.extend((f"delay_{delay}", {"record_delay": delay}) for delay in cfg["record_delays"] if delay)
    return views


def metrics(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    """Return development MAE and RMSE in original proxy units."""
    error = prediction - y
    return float(np.abs(error).mean()), float(np.sqrt(np.mean(error ** 2)))


def aggregate(rows: list[dict[str, Any]], keys: list[str], values: list[str]) -> list[dict[str, Any]]:
    """Aggregate per-seed metrics with sample SD, never treating views as seeds."""
    bins: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bins[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, group in sorted(bins.items()):
        result = dict(zip(keys, key))
        result["data_repeats"] = len(group)
        for value in values:
            a = np.asarray([r[value] for r in group])
            result[value + "_mean"] = float(a.mean())
            result[value + "_sd"] = float(a.std(ddof=1)) if len(a) > 1 else 0.
        out.append(result)
    return out


def run_pilot(root: Path, cfg: dict[str, Any], output: Path,
              implementation_commit: str) -> dict[str, Any]:
    """Fit, archive and replay the frozen development study in a fresh directory."""
    if cfg["evaluation_split"] != "val" or cfg["test_predictive_evaluation"] is not False:
        raise ValueError("pilot supports development validation only")
    if tuple(cfg["groups"]) != GROUPS or cfg["selection"] != "original_validation_mae":
        raise ValueError("unexpected group or selection protocol")
    if cfg["score_method"] != "regime_marginal" or cfg["targets"] != ["arrival_proxy", "null_proxy"]:
        raise ValueError("unexpected scorer or targets")
    if not re.fullmatch(r"[0-9a-f]{40}", implementation_commit):
        raise ValueError("explicit implementation commit required")
    source_dir = Path(__file__).resolve().parent
    if sha256(source_dir / "rq3_paired_data.py") != cfg["generator_sha256"]:
        raise ValueError("local generator differs from externally pinned source")
    data_cfg = yaml.safe_load((root / "config.yaml").read_text())
    if cfg["seeds"] != data_cfg["seeds"]:
        raise ValueError("data seed scope mismatch")
    if output.exists():
        raise FileExistsError(output)
    audit = strict_audit(root, expected_dataset_id=cfg["dataset_id"],
                         expected_source_sha256=cfg["generator_sha256"],
                         expected_manifest_sha256=cfg["data_manifest_sha256"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "strict_data_audit.json", audit)
    write_json(output / "config.json", cfg)
    write_json(output / "input_manifest.json", json.loads((root / "manifest.json").read_text()))
    source_hashes = {"src/" + name: sha256(source_dir / name) for name in SOURCE_FILES}
    metric_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    replay_max = equivalence_max = 0.
    regimes = data_cfg["regimes"]
    views = view_specs(cfg)
    for seed in cfg["seeds"]:
        directory = output / f"seed_{seed}"
        directory.mkdir()
        train = development_inputs(root, seed, "train")
        val = development_inputs(root, seed, "val")
        y_train = targets(root, seed, "train", train["sequence_id"], cfg["targets"])
        y_val = targets(root, seed, "val", val["sequence_id"], cfg["targets"])
        score_model = fit_score(root, seed, regimes, cfg)
        prep = fit_preprocessor(train)
        score_train = score_view(train, regimes, score_model)
        score_val = score_view(val, regimes, score_model)
        bank = score_train["reliability"]
        random_train = random_scores(train["sequence_id"], bank, seed, cfg["random_control_seed"])
        random_val = random_scores(val["sequence_id"], bank, seed, cfg["random_control_seed"])
        write_json(directory / "preprocessing_and_score.json", {"preprocessing": prep, "scoring": score_model})
        write_json(directory / "sequence_ids.json", {split: development_inputs(root, seed, split)["sequence_id"].tolist()
                   for split in ("reference_fit", "reference_cal", "train", "val")})
        learned: dict[str, dict[str, np.ndarray]] = {}
        archived: dict[str, np.ndarray] = {"mean": y_train.mean(axis=0)}
        for group in GROUPS[1:]:
            x_train = features(train, score_train, prep, group, random_train)
            x_val = features(val, score_val, prep, group, random_val)
            path = ridge_path(x_train, y_train, cfg["ridge_alphas"])
            model, loss = select_model(path, x_val, y_val, cfg["ridge_alphas"])
            learned[group] = model
            for key, value in model.items():
                archived[f"{group}__{key}"] = value
            for target_index, target in enumerate(cfg["targets"]):
                for i, alpha in enumerate(cfg["ridge_alphas"]):
                    selection_rows.append({"seed": seed, "group": group, "target": target,
                                           "alpha": alpha, "original_val_mae": float(loss[i, target_index]),
                                           "selected": bool(alpha == model["alphas"][target_index]),
                                           "feature_count": x_train.shape[1],
                                           "constant_train_columns": int(path["constant"].sum())})
        np.savez_compressed(directory / "ridge_models.npz", **archived)
        # Replay from serialized parameters, not just in-memory coefficient copies.
        with np.load(directory / "ridge_models.npz", allow_pickle=False) as z:
            reloaded = {g: {k: z[f"{g}__{k}"].copy() for k in ("center", "scale", "weights", "intercept", "alphas")}
                        for g in GROUPS[1:]}
            reloaded_mean = z["mean"].copy()
        persisted = json.loads((directory / "preprocessing_and_score.json").read_text())
        replay_bank = score_view(train, regimes, persisted["scoring"])["reliability"]
        replay_random_val = random_scores(val["sequence_id"], replay_bank, seed, cfg["random_control_seed"])
        with (root / f"seed_{seed}" / "val" / "events.csv").open(newline="") as f:
            events = list(csv.DictReader(f))
        if [r["sequence_id"] for r in events] != list(val["sequence_id"]):
            raise ValueError("evaluation event alignment mismatch")
        event_channel = np.asarray([int(r["channel"]) for r in events])
        event_duration = np.asarray([int(r["end_exclusive"]) - int(r["start"]) for r in events])
        all_predictions = []
        for view_name, kwargs in views:
            view = development_inputs(root, seed, "val", **kwargs)
            score = score_view(view, regimes, score_model)
            score_reloaded = score_view(view, regimes, persisted["scoring"])
            predictions = []
            for group in GROUPS:
                if group == "mean":
                    pred = np.broadcast_to(archived["mean"], y_val.shape).copy()
                    rep = np.broadcast_to(reloaded_mean, y_val.shape)
                else:
                    x = features(view, score, prep, group, random_val)
                    pred = predict(x, learned[group])
                    rep = predict(features(view, score_reloaded, persisted["preprocessing"], group, replay_random_val), reloaded[group])
                replay_max = max(replay_max, float(np.max(np.abs(pred - rep))))
                if not np.isfinite(pred).all() or not np.allclose(pred, rep, rtol=0, atol=1e-10):
                    raise ValueError("archived model replay mismatch/nonfinite")
                predictions.append(pred)
                strata = {"all": np.ones(len(y_val), bool)}
                for r in range(3):
                    strata[f"cutoff_regime:{regimes[r]}"] = val["recorded_regime"][:, -1] == r
                if view_name.startswith(("bias", "drift", "missing")):
                    for ch, name in enumerate(data_cfg["features"]):
                        strata[f"fault_channel:{name}"] = event_channel == ch
                    for duration in data_cfg["faults"]["durations"]:
                        strata[f"duration:{duration}"] = event_duration == duration
                for stratum, mask in strata.items():
                    if not mask.any():
                        continue
                    for target_index, target in enumerate(cfg["targets"]):
                        mae, rmse = metrics(y_val[mask, target_index], pred[mask, target_index])
                        metric_rows.append({"seed": seed, "view": view_name, "group": group,
                                            "target": target, "stratum": stratum,
                                            "batches": int(mask.sum()), "mae": mae, "rmse": rmse})
            equivalence_max = max(equivalence_max, float(np.max(np.abs(predictions[2] - predictions[3]))))
            if not np.allclose(predictions[2], predictions[3], rtol=0, atol=1e-8):
                raise ValueError("predeclared B1/B2 equivalence failed")
            all_predictions.append(np.stack(predictions))
            for label, sc in ((view_name, score),):
                finite = sc["reliability"][sc["available"]]
                quantiles = np.quantile(finite, [.01, .5, .99])
                diagnostic_rows.append({"seed": seed, "split": "val", "view": label,
                                        "available_fraction": float(sc["available"].mean()),
                                        "q01": float(quantiles[0]), "q50": float(quantiles[1]), "q99": float(quantiles[2])})
        q = np.quantile(bank, [.01, .5, .99])
        diagnostic_rows.append({"seed": seed, "split": "train", "view": "original",
                                "available_fraction": float(score_train["available"].mean()),
                                "q01": float(q[0]), "q50": float(q[1]), "q99": float(q[2])})
        np.savez_compressed(directory / "validation_predictions.npz", prediction=np.stack(all_predictions),
                            targets=y_val, sequence_id=val["sequence_id"],
                            views=np.asarray([v[0] for v in views]), groups=np.asarray(GROUPS),
                            target_names=np.asarray(cfg["targets"]))
        current = {(r["view"], r["group"], r["target"]): r["mae"]
                   for r in metric_rows if r["seed"] == seed and r["stratum"] == "all"}
        for better, base in (("B3", "B2"), ("B3Q", "B2Q"), ("B2R", "B2")):
            for view_name, _ in views:
                for target in cfg["targets"]:
                    diff = current[view_name, better, target] - current[view_name, base, target]
                    original_diff = current["original", better, target] - current["original", base, target]
                    paired_rows.append({"seed": seed, "view": view_name, "comparison": better + "-" + base,
                                        "target": target, "mae_difference": diff,
                                        "degradation_difference": diff - original_diff})
        print(f"seed {seed}: fitted 7 ridge groups, 2 targets, {len(views)} validation views; replay passed", flush=True)
    write_csv(output / "metrics_by_seed.csv", metric_rows)
    write_csv(output / "metrics_summary.csv", aggregate(metric_rows, ["view", "group", "target", "stratum"], ["mae", "rmse"]))
    write_csv(output / "selection.csv", selection_rows)
    write_csv(output / "score_diagnostics.csv", diagnostic_rows)
    write_csv(output / "paired_by_seed.csv", paired_rows)
    write_csv(output / "paired_summary.csv", aggregate(paired_rows, ["view", "comparison", "target"], ["mae_difference", "degradation_difference"]))
    if sha256(root / "manifest.json") != cfg["data_manifest_sha256"]:
        raise ValueError("input manifest changed during run")
    # Verify every source artifact byte again without loading test for prediction.
    for rel, expected in json.loads((root / "manifest.json").read_text()).items():
        if sha256(root / rel) != expected:
            raise ValueError("input data changed during run: " + rel)
    metadata = {"run_id": cfg["run_id"], "dataset_id": cfg["dataset_id"], "data_source": "synthetic",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "implementation_commit": implementation_commit,
                "provenance_mode": "explicit commit verified against uploaded source blobs; recorded local SHA-256",
                "source_sha256": source_hashes, "data_manifest_sha256": cfg["data_manifest_sha256"],
                "python": platform.python_version(),
                "dependencies": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "PyYAML")},
                "test_predictive_evaluation": False, "validation_used_for_selection": True,
                "prediction_splits_loaded": ["reference_fit", "reference_cal", "train", "val"],
                "base_train_batches": sum(data_cfg["counts"]["train"] for _ in cfg["seeds"]),
                "base_val_batches": sum(data_cfg["counts"]["val"] for _ in cfg["seeds"]),
                "learned_model_target_pairs": len(cfg["seeds"]) * 7 * 2,
                "model_replay_max_abs_difference": replay_max,
                "B1_B2_max_abs_prediction_difference": equivalence_max,
                "interpretation": "development validation of synthetic proxies, not final test or industrial quality evidence"}
    write_json(output / "metadata.json", metadata)
    write_json(output / "manifest.json", {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()})
    return metadata


def main() -> None:
    """Run the frozen pilot on an existing immutable dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    print(json.dumps(run_pilot(args.data, config, args.output, args.implementation_commit), indent=2))


if __name__ == "__main__":
    main()
