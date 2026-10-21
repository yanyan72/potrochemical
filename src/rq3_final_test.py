"""One-time held-out evaluation of frozen RQ3 predictors; no fitting path."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import t as student_t

from .rq3_paired_data import load_inputs
from .rq3_prediction_pilot import (aggregate, features, metrics, predict, random_scores,
                                  score_view, view_specs, write_csv, write_json)
from .rq3_lstm_pilot import PlainLSTM, check_files, infer, transform_sequence
from .rq3_release_audit import sha256

ARCHITECTURES = ("ridge", "lstm")
GROUPS = ("mean", "B0", "B2", "B3", "B2Q", "B3Q", "B2R")
COMPARISONS = (("B3", "B2"), ("B3Q", "B2Q"), ("B2R", "B2"))


def utc_now() -> str:
    """Return a provenance timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a saved CSV table."""
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def validate_sources(data: Path, ridge: Path, lstm: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Verify immutable identities, including current imported source files."""
    if tuple(cfg["architectures"]) != ARCHITECTURES or tuple(cfg["groups"]) != GROUPS:
        raise ValueError("unexpected comparison scope")
    if cfg["evaluation_split"] != "test" or cfg["refit"] is not False:
        raise ValueError("final protocol must be frozen test without refitting")
    if cfg["targets"] != ["arrival_proxy", "null_proxy"]:
        raise ValueError("target scope mismatch")
    for folder, key in ((data, "data_manifest_sha256"), (ridge, "ridge_manifest_sha256"), (lstm, "lstm_manifest_sha256")):
        if sha256(folder / "manifest.json") != cfg[key]:
            raise ValueError("manifest pin mismatch: " + key)
        check_files(folder, json.loads((folder / "manifest.json").read_text()))
    dc = yaml.safe_load((data / "config.yaml").read_text())
    if dc["dataset_id"] != cfg["dataset_id"] or dc["seeds"] != cfg["seeds"]:
        raise ValueError("dataset identity/scope mismatch")
    for folder in (ridge, lstm):
        pc = json.loads((folder / "config.json").read_text())
        if any(pc[k] != cfg[k] for k in ("seeds", "targets", "fault_magnitudes", "record_delays", "random_control_seed", "data_manifest_sha256")):
            raise ValueError("development protocol mismatch")
    base = Path(__file__).resolve().parents[1]
    sources = json.loads((lstm / "run_identity.json").read_text())["source_sha256"]
    check_files(base, sources)
    sources = {**sources, "src/rq3_final_test.py": sha256(Path(__file__))}
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    return sources


def predict_saved(data: Path, ridge: Path, lstm: Path, cfg: dict[str, Any], seed: int,
                  split: str, *, execute_frozen_test: bool = False) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Predict from archived parameters only; no target/truth values are accepted."""
    if split not in ("val", "test") or (split == "test" and not execute_frozen_test):
        raise ValueError("test prediction requires explicit frozen-test execution")
    dc = yaml.safe_load((data / "config.yaml").read_text())
    lp = json.loads((lstm / f"seed_{seed}/parameters.json").read_text())
    rp = json.loads((ridge / f"seed_{seed}/preprocessing_and_score.json").read_text())
    if lp["base"] != rp:
        raise ValueError("scoring/preprocessor archives differ")
    lc = json.loads((lstm / "config.json").read_text())
    train = load_inputs(data, seed, "train")
    bank = score_view(train, dc["regimes"], rp["scoring"])["reliability"]
    views = [load_inputs(data, seed, split, allow_test=execute_frozen_test, **kw) for _, kw in view_specs(cfg)]
    original = views[0]
    random = random_scores(original["sequence_id"], bank, seed, cfg["random_control_seed"])
    scores = [score_view(v, dc["regimes"], rp["scoring"]) for v in views]
    result = np.empty((2, len(views), len(GROUPS), len(original["sequence_id"]), len(cfg["targets"])))
    with np.load(ridge / f"seed_{seed}/ridge_models.npz", allow_pickle=False) as rstate, np.load(lstm / f"seed_{seed}/models.npz", allow_pickle=False) as lstate:
        result[:, :, 0] = rstate["mean"]
        for gi, group in enumerate(GROUPS[1:], 1):
            rm = {k: rstate[f"{group}__{k}"] for k in ("center", "scale", "weights", "intercept")}
            models = []
            for target in cfg["targets"]:
                key = group + "__" + target
                detail = lp["models"][key]
                model = PlainLSTM(detail["input_channels"], lc["hidden_size"])
                model.load_state_dict({n: torch.from_numpy(lstate[key + "__" + n].copy()) for n in model.state_dict()})
                models.append((model, detail))
            for vi, (view, score) in enumerate(zip(views, scores)):
                flat = features(view, score, rp["preprocessing"], group, random)
                result[0, vi, gi] = predict(flat, rm)
                x = transform_sequence(flat.reshape(len(flat), view["observations"].shape[1], -1), lp["groups"][group])
                for ti, (model, detail) in enumerate(models):
                    result[1, vi, gi, :, ti] = infer(model, x, detail["target_center"], detail["target_scale"], lc["batch_size"])
    if not np.isfinite(result).all():
        raise ValueError("nonfinite frozen-model predictions")
    return result, original


def replay_validation(data: Path, ridge: Path, lstm: Path, cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    """Check final inference against all existing val predictions before test use."""
    if output.exists():
        raise FileExistsError(output)
    validate_sources(data, ridge, lstm, cfg)
    checks = []
    for seed in cfg["seeds"]:
        predictions, view = predict_saved(data, ridge, lstm, cfg, seed, "val")
        for ai, folder in enumerate((ridge, lstm)):
            with np.load(folder / f"seed_{seed}/validation_predictions.npz", allow_pickle=False) as old:
                if not np.array_equal(view["sequence_id"], old["sequence_id"]):
                    raise ValueError("validation identity mismatch")
                indices = [list(old["groups"]).index(g) for g in GROUPS]
                expected = old["prediction"][:, indices]
                difference = float(np.abs(predictions[ai] - expected).max())
                if difference > cfg["validation_replay_tolerance"]:
                    raise ValueError("validation replay mismatch")
                checks.append(dict(seed=seed, architecture=ARCHITECTURES[ai], max_abs_difference=difference))
    result = {"status": "pass", "checks": checks, "test_predictive_evaluation": False,
              "model_refitting": False, "source_manifests_unchanged": True}
    write_json(output, result)
    return result


def aligned_labels(data: Path, seed: int, ids: np.ndarray, target_names: list[str], *,
                   execute_frozen_test: bool = False) -> np.ndarray:
    """Read test labels for evaluation only after predictions have been persisted."""
    if not execute_frozen_test:
        raise ValueError("test labels require explicit frozen-test execution")
    rows = read_rows(data / f"seed_{seed}/test/targets.csv")
    if [r["sequence_id"] for r in rows] != list(ids):
        raise ValueError("test target ID mismatch")
    y = np.asarray([[float(r[t]) for t in target_names] for r in rows])
    if not np.isfinite(y).all():
        raise ValueError("nonfinite test labels")
    return y


def evaluate_arrays(pred: np.ndarray, y: np.ndarray, ids: np.ndarray, recorded: np.ndarray,
                    events: list[dict[str, str]], cfg: dict[str, Any], dc: dict[str, Any], seed: int
                    ) -> tuple[list[dict], list[dict], list[dict]]:
    """Compute paired errors only after prediction; views are not independent units."""
    if [r["sequence_id"] for r in events] != list(ids):
        raise ValueError("evaluation event alignment")
    metrics_rows, paired, endpoints = [], [], []
    specs = view_specs(cfg)
    for ai, arch in enumerate(ARCHITECTURES):
        for vi, (view, _) in enumerate(specs):
            masks = {"all": np.ones(len(y), bool)}
            masks.update({"cutoff_regime:" + n: recorded[:, -1] == i for i, n in enumerate(dc["regimes"])})
            if view.startswith(("bias", "drift", "missing")):
                masks.update({"fault_channel:" + n: np.array([int(e["channel"]) == i for e in events]) for i, n in enumerate(dc["features"])})
                masks.update({f"duration:{d}": np.array([int(e["end_exclusive"]) - int(e["start"]) == d for e in events]) for d in dc["faults"]["durations"]})
            for gi, group in enumerate(GROUPS):
                for ti, target in enumerate(cfg["targets"]):
                    for name, mask in masks.items():
                        if mask.any():
                            mae, rmse = metrics(y[mask, ti], pred[ai, vi, gi, mask, ti])
                            metrics_rows.append(dict(seed=seed, architecture=arch, view=view, group=group,
                                target=target, stratum=name, batches=int(mask.sum()), mae=mae, rmse=rmse))
        for better, base in COMPARISONS:
            bi, ci = GROUPS.index(better), GROUPS.index(base)
            # Shape [view,batch,target]; average batches inside each seed first.
            errors = np.abs(pred[ai, :, bi] - y[None]) - np.abs(pred[ai, :, ci] - y[None])
            deltas = errors.mean(axis=1)
            for ti, target in enumerate(cfg["targets"]):
                for vi, (view, _) in enumerate(specs):
                    paired.append(dict(seed=seed, architecture=arch, view=view, comparison=better + "-" + base,
                        target=target, mae_difference=float(deltas[vi, ti]),
                        degradation_difference=float(deltas[vi, ti] - deltas[0, ti])))
                for scope, indices in (("original", [0]), ("pressure_equal_weight", list(range(1, len(specs)))),
                                       ("suite_equal_weight", list(range(len(specs))))):
                    endpoints.append(dict(seed=seed, architecture=arch, comparison=better + "-" + base,
                        target=target, scope=scope, views=len(indices), batches=len(y),
                        mae_difference=float(deltas[indices, ti].mean())))
    return metrics_rows, paired, endpoints


def seed_interval(values: list[float], confidence: float) -> dict[str, Any]:
    """Use only independent seed deltas for the registered Student-t interval."""
    a = np.asarray(values, dtype=float)
    if len(a) < 2 or not np.isfinite(a).all() or not 0 < confidence < 1:
        raise ValueError("interval requires finite independent replicates")
    mean, sd = float(a.mean()), float(a.std(ddof=1))
    half = float(student_t.ppf((1 + confidence) / 2, len(a) - 1) * sd / np.sqrt(len(a)))
    return dict(data_repeats=len(a), df=len(a) - 1, mae_difference_mean=mean,
                mae_difference_sd=sd, ci_low=mean - half, ci_high=mean + half)


def endpoint_summary(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep registered primary and descriptive endpoints distinct."""
    keys = ("architecture", "comparison", "target", "scope")
    bins: dict[tuple, list[dict]] = {}
    for r in rows:
        bins.setdefault(tuple(r[k] for k in keys), []).append(r)
    output = []
    for key, entries in sorted(bins.items()):
        if sorted(int(r["seed"]) for r in entries) != sorted(cfg["seeds"]):
            raise ValueError("duplicate or missing independent seed")
        interval = seed_interval([float(r["mae_difference"]) for r in entries], cfg["confidence_level"])
        primary = key == tuple(cfg["primary_" + k] for k in keys)
        direction = "lower_error" if interval["ci_high"] < 0 else "higher_error" if interval["ci_low"] > 0 else "inconclusive"
        output.append({**dict(zip(keys, key)), **interval, "primary": primary,
                       "interval_direction": direction, "confidence_level": cfg["confidence_level"]})
    return output


def initialize_run(data: Path, ridge: Path, lstm: Path, cfg: dict[str, Any], output: Path,
                   protocol_commit: str, implementation_commit: str, execute_frozen_test: bool) -> None:
    """Persist test-use status before any test prediction; prevent silent reruns."""
    if not execute_frozen_test:
        raise ValueError("explicit frozen-test execution required")
    if any(not re.fullmatch(r"[0-9a-f]{40}", x) for x in (protocol_commit, implementation_commit)):
        raise ValueError("published protocol and implementation commits required")
    sources = validate_sources(data, ridge, lstm, cfg)
    if output.exists():
        if (output / "manifest.json").exists():
            raise FileExistsError("sealed test run cannot be overwritten")
        identity = json.loads((output / "identity.json").read_text())
        if (identity["source_sha256"] != sources or identity["protocol_commit"] != protocol_commit
            or identity["implementation_commit"] != implementation_commit
            or json.loads((output / "config.json").read_text()) != cfg):
            raise ValueError("test run identity mismatch")
        return
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", cfg)
    write_json(output / "identity.json", {"protocol_commit": protocol_commit, "implementation_commit": implementation_commit,
        "source_sha256": sources, "python": platform.python_version(),
        "dependencies": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "scipy", "scikit-learn", "PyYAML")},
        "created_utc": utc_now(), "data_source": "synthetic_rebuilt", "refitting": False})
    write_json(output / "test_use.json", {"started_utc": utc_now(), "status": "test_execution_started",
        "notice": "This test is now consumed; incomplete outputs do not restore unseen status."})
    write_json(output / "source_manifests.json", {name: json.loads((folder / "manifest.json").read_text())
        for name, folder in (("data", data), ("ridge", ridge), ("lstm", lstm))})


def run_seed(data: Path, ridge: Path, lstm: Path, cfg: dict[str, Any], output: Path,
             protocol_commit: str, implementation_commit: str, seed: int, *,
             execute_frozen_test: bool = False) -> dict[str, Any]:
    """Evaluate one held-out seed once, persisting predictions before labels."""
    if seed not in cfg["seeds"]:
        raise ValueError("seed outside protocol")
    initialize_run(data, ridge, lstm, cfg, output, protocol_commit, implementation_commit, execute_frozen_test)
    sd = output / f"seed_{seed}"
    sd.mkdir(exist_ok=False)
    pred, original = predict_saved(data, ridge, lstm, cfg, seed, "test", execute_frozen_test=True)
    np.savez_compressed(sd / "predictions.npz", prediction=pred, sequence_id=original["sequence_id"],
        architectures=np.asarray(ARCHITECTURES), views=np.asarray([n for n, _ in view_specs(cfg)]),
        groups=np.asarray(GROUPS), target_names=np.asarray(cfg["targets"]))
    y = aligned_labels(data, seed, original["sequence_id"], cfg["targets"], execute_frozen_test=True)
    np.savez_compressed(sd / "evaluation_targets.npz", targets=y, sequence_id=original["sequence_id"])
    dc = yaml.safe_load((data / "config.yaml").read_text())
    events = read_rows(data / f"seed_{seed}/test/events.csv")
    rows, paired, endpoints = evaluate_arrays(pred, y, original["sequence_id"], original["recorded_regime"], events, cfg, dc, seed)
    for name, table in (("metrics_by_seed", rows), ("paired_by_seed", paired), ("endpoints_by_seed", endpoints)):
        write_csv(sd / (name + ".csv"), table)
    validate_sources(data, ridge, lstm, cfg)
    meta = {"seed": seed, "completed_utc": utc_now(), "test_predictive_evaluation": True,
            "batches": len(y), "models_refitted": 0, "source_files_unchanged": True,
            "predictions_saved_before_target_read": True}
    write_json(sd / "metadata.json", meta)
    write_json(sd / "manifest.json", {p.name: sha256(p) for p in sorted(sd.iterdir()) if p.is_file()})
    return meta


def summarize(output: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Seal complete held-out results, retaining every registered comparison."""
    if (output / "manifest.json").exists():
        raise FileExistsError("sealed test run")
    tables = {n: [] for n in ("metrics_by_seed", "paired_by_seed", "endpoints_by_seed")}
    for seed in cfg["seeds"]:
        sd = output / f"seed_{seed}"
        check_files(sd, json.loads((sd / "manifest.json").read_text()))
        for name in tables:
            tables[name].extend(read_rows(sd / (name + ".csv")))
    for name, rows in tables.items():
        write_csv(output / (name + ".csv"), rows)
    for name, raw, keys, values in (
        ("metrics_summary", "metrics_by_seed", ["architecture", "view", "group", "target", "stratum"], ["mae", "rmse"]),
        ("paired_summary", "paired_by_seed", ["architecture", "view", "comparison", "target"], ["mae_difference", "degradation_difference"])):
        converted = [{**r, **{v: float(r[v]) for v in values}} for r in tables[raw]]
        write_csv(output / (name + ".csv"), aggregate(converted, keys, values))
    endpoints = endpoint_summary(tables["endpoints_by_seed"], cfg)
    write_csv(output / "endpoints_summary.csv", endpoints)
    primary = [r for r in endpoints if r["primary"]]
    if len(primary) != 1:
        raise ValueError("exactly one prespecified primary endpoint required")
    meta = {"run_id": cfg["run_id"], "completed_utc": utc_now(), "test_predictive_evaluation": True,
        "test_status": "used", "refitting": False, "data_source": "synthetic_rebuilt", "primary_endpoint": primary[0],
        "unique_test_batches": sum(json.loads((output / f"seed_{s}/metadata.json").read_text())["batches"] for s in cfg["seeds"]),
        "independent_repeats": len(cfg["seeds"]), "scope": "current frozen synthetic mechanism and archived predictors only"}
    write_json(output / "metadata.json", meta)
    write_json(output / "manifest.json", {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()})
    return meta


def main() -> None:
    """Validate val replay or explicitly execute the published frozen test protocol."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "ridge", "lstm", "config", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--protocol-commit", default="")
    p.add_argument("--implementation-commit", default="")
    p.add_argument("--execute-frozen-test", action="store_true")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate-only", action="store_true")
    g.add_argument("--seed", type=int)
    g.add_argument("--summarize", action="store_true")
    a = p.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    if a.validate_only:
        result = replay_validation(a.data, a.ridge, a.lstm, cfg, a.output)
    elif a.summarize:
        initialize_run(a.data, a.ridge, a.lstm, cfg, a.output, a.protocol_commit, a.implementation_commit, a.execute_frozen_test)
        result = summarize(a.output, cfg)
    else:
        result = run_seed(a.data, a.ridge, a.lstm, cfg, a.output, a.protocol_commit, a.implementation_commit,
                          a.seed, execute_frozen_test=a.execute_frozen_test)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
