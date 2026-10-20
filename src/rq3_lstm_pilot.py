"""Frozen CPU LSTM development study on immutable RQ3 paired synthetic data."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import yaml

from .rq3_prediction_pilot import (SOURCE_FILES, aggregate, development_inputs, features,
    metrics, random_scores, score_view, targets, view_specs, write_csv, write_json)
from .rq3_release_audit import sha256, strict_audit

GROUPS = ("mean", "B0", "B2", "B3", "B2Q", "B3Q", "B2R")


class PlainLSTM(nn.Module):
    """Single causal LSTM with mean-visible-hidden-state scalar readout."""

    def __init__(self, inputs: int, hidden: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(inputs, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict one standardized target from a complete visible prefix."""
        h, _ = self.lstm(x)
        return self.head(h.mean(dim=1)).squeeze(-1)


def configure_runtime(cfg: dict[str, Any]) -> None:
    """Require the registered development-only architecture and CPU execution."""
    if cfg["evaluation_split"] != "val" or cfg["test_predictive_evaluation"] is not False:
        raise ValueError("LSTM pilot excludes test")
    if tuple(cfg["groups"]) != GROUPS or cfg["targets"] != ["arrival_proxy", "null_proxy"]:
        raise ValueError("unexpected group/target protocol")
    for key, value in {"device": "cpu", "num_layers": 1, "readout": "mean_hidden",
                       "optimizer": "AdamW", "loss": "standardized_target_mse",
                       "selection": "original_validation_mae_best_epoch",
                       "constant_train_features": "zero_in_all_splits"}.items():
        if cfg[key] != value:
            raise ValueError("unsupported protocol: " + key)
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)


def sequence_features(view: dict[str, np.ndarray], scores: dict[str, np.ndarray],
                      prep: dict[str, Any], group: str, random: np.ndarray) -> np.ndarray:
    """Reuse the frozen feature allowlist without adding IDs, labels or truth."""
    flat = features(view, scores, prep, group, random)
    return flat.reshape(len(flat), view["observations"].shape[1], -1)


def fit_sequence_scaler(train: np.ndarray) -> dict[str, Any]:
    """Fit channel statistics across training batches/time, freezing constants."""
    if train.ndim != 3 or not np.isfinite(train).all():
        raise ValueError("invalid training features")
    center, scale = train.mean(axis=(0, 1)), train.std(axis=(0, 1))
    constant = scale < 1e-12
    scale[constant] = 1.
    return {"center": center.tolist(), "scale": scale.tolist(), "constant": constant.tolist()}


def transform_sequence(x: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    """Apply train statistics; unseen changes in constant channels stay zero."""
    z = (x - np.asarray(scaler["center"])) / np.asarray(scaler["scale"])
    z[..., np.asarray(scaler["constant"], bool)] = 0.
    if not np.isfinite(z).all():
        raise ValueError("nonfinite transformed features")
    return z.astype(np.float32)


def infer(model: PlainLSTM, x: np.ndarray, y_center: float, y_scale: float,
          batch_size: int) -> np.ndarray:
    """Predict on CPU using an archived training target scale."""
    model.eval()
    with torch.no_grad():
        p = torch.cat([model(torch.from_numpy(x[i:i + batch_size]))
                       for i in range(0, len(x), batch_size)]).numpy()
    return p.astype(np.float64) * y_scale + y_center


def train_model(x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray,
                cfg: dict[str, Any], model_seed: int, shuffle_seed: int
                ) -> tuple[PlainLSTM, dict[str, Any], list[dict[str, Any]]]:
    """Train fixed epochs, selecting the earliest minimum original-val MAE."""
    torch.manual_seed(model_seed)
    model = PlainLSTM(x.shape[-1], cfg["hidden_size"])
    center, scale = float(y.mean()), float(y.std())
    if scale < 1e-12:
        raise ValueError("constant training target")
    tx, ty = torch.from_numpy(x), torch.from_numpy(((y - center) / scale).astype(np.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    shuffle = np.random.default_rng(shuffle_seed)
    history, best_state = [], None
    best, best_epoch = float("inf"), -1
    for epoch in range(1, cfg["epochs"] + 1):
        order = shuffle.permutation(len(x))
        model.train()
        total = 0.
        for start in range(0, len(x), cfg["batch_size"]):
            idx = order[start:start + cfg["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(tx[idx]) - ty[idx]) ** 2)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        val_mae = float(np.abs(infer(model, vx, center, scale, cfg["batch_size"]) - vy).mean())
        if val_mae < best:
            best, best_epoch = val_mae, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch, "train_online_standardized_mse": total / len(x),
                        "original_val_mae": val_mae})
    assert best_state is not None
    model.load_state_dict(best_state)
    for r in history:
        r["selected"] = r["epoch"] == best_epoch
    return model, {"target_center": center, "target_scale": scale, "selected_epoch": best_epoch,
                   "selected_val_mae": best, "model_seed": model_seed, "shuffle_seed": shuffle_seed,
                   "input_channels": x.shape[-1], "parameter_count": sum(p.numel() for p in model.parameters())}, history


def check_files(root: Path, manifest: dict[str, str]) -> None:
    """Reject changed source artifacts using byte hashes."""
    for rel, expected in manifest.items():
        if sha256(root / rel) != expected:
            raise ValueError("hash mismatch: " + str(root / rel))


def initialize(data: Path, ridge: Path, cfg: dict[str, Any], output: Path,
               commit: str) -> None:
    """Create immutable run identity, or verify identity before another seed."""
    configure_runtime(cfg)
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("explicit implementation commit required")
    source = Path(__file__).resolve().parent
    if sha256(source / "rq3_paired_data.py") != cfg["generator_sha256"]:
        raise ValueError("generator source changed")
    if sha256(data / "manifest.json") != cfg["data_manifest_sha256"]:
        raise ValueError("data manifest pin mismatch")
    if sha256(ridge / "manifest.json") != cfg["ridge_manifest_sha256"]:
        raise ValueError("ridge manifest pin mismatch")
    check_files(data, json.loads((data / "manifest.json").read_text()))
    check_files(ridge, json.loads((ridge / "manifest.json").read_text()))
    sources = {"src/" + n: sha256(source / n) for n in (*SOURCE_FILES, "rq3_lstm_pilot.py", "__init__.py", "simulator.py")}
    if output.exists():
        if (output / "manifest.json").exists():
            raise FileExistsError("completed run is immutable")
        stored = json.loads((output / "run_identity.json").read_text())
        if stored["implementation_commit"] != commit or stored["source_sha256"] != sources:
            raise ValueError("run code identity changed")
        if json.loads((output / "config.json").read_text()) != cfg:
            raise ValueError("run config changed")
        return
    audit = strict_audit(data, expected_dataset_id=cfg["dataset_id"],
                         expected_source_sha256=cfg["generator_sha256"],
                         expected_manifest_sha256=cfg["data_manifest_sha256"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "strict_data_audit.json", audit)
    write_json(output / "config.json", cfg)
    write_json(output / "input_manifest.json", json.loads((data / "manifest.json").read_text()))
    write_json(output / "run_identity.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_commit": commit, "source_sha256": sources, "python": platform.python_version(),
        "dependencies": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "scipy", "scikit-learn", "PyYAML", "pandas")},
        "data_source": "synthetic_rebuilt", "test_predictive_evaluation": False,
        "validation_used_for_selection": True, "device": "cpu", "threads": cfg["threads"],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()})


def evaluation_tables(data: Path, seed: int, cfg: dict[str, Any], pred: np.ndarray,
                      y: np.ndarray, val: dict[str, np.ndarray]) -> tuple[list[dict], list[dict]]:
    """Compute registered overall and event strata after prediction is complete."""
    dc = yaml.safe_load((data / "config.yaml").read_text())
    with (data / f"seed_{seed}/val/events.csv").open() as f:
        events = list(csv.DictReader(f))
    if [e["sequence_id"] for e in events] != list(val["sequence_id"]):
        raise ValueError("event alignment")
    rows, paired = [], []
    for vi, (view, _) in enumerate(view_specs(cfg)):
        masks = {"all": np.ones(len(y), bool)}
        masks.update({"cutoff_regime:" + r: val["recorded_regime"][:, -1] == i for i, r in enumerate(dc["regimes"])})
        if view.startswith(("bias", "drift", "missing")):
            masks.update({"fault_channel:" + n: np.array([int(e["channel"]) == i for e in events]) for i, n in enumerate(dc["features"])})
            masks.update({f"duration:{d}": np.array([int(e["end_exclusive"]) - int(e["start"]) == d for e in events]) for d in dc["faults"]["durations"]})
        for gi, group in enumerate(cfg["groups"]):
            for ti, target in enumerate(cfg["targets"]):
                for name, mask in masks.items():
                    if mask.any():
                        mae, rmse = metrics(y[mask, ti], pred[vi, gi, mask, ti])
                        rows.append(dict(seed=seed, view=view, group=group, target=target,
                                         stratum=name, batches=int(mask.sum()), mae=mae, rmse=rmse))
    lookup = {(r["view"], r["group"], r["target"]): r["mae"] for r in rows if r["stratum"] == "all"}
    for a, b in (("B3", "B2"), ("B3Q", "B2Q"), ("B2R", "B2")):
        for view, _ in view_specs(cfg):
            for target in cfg["targets"]:
                delta = lookup[view, a, target] - lookup[view, b, target]
                orig = lookup["original", a, target] - lookup["original", b, target]
                paired.append(dict(seed=seed, view=view, comparison=a + "-" + b, target=target,
                                   mae_difference=delta, degradation_difference=delta - orig))
    return rows, paired


def run_seed(data: Path, ridge: Path, cfg: dict[str, Any], output: Path,
             commit: str, seed: int) -> dict[str, Any]:
    """Train one registered seed and persist/replay all checkpoints and views."""
    if seed not in cfg["seeds"]:
        raise ValueError("seed outside protocol")
    initialize(data, ridge, cfg, output, commit)
    sd = output / f"seed_{seed}"
    sd.mkdir(exist_ok=False)
    started = time.monotonic()
    dc = yaml.safe_load((data / "config.yaml").read_text())
    frozen = json.loads((ridge / f"seed_{seed}/preprocessing_and_score.json").read_text())
    train = development_inputs(data, seed, "train")
    val = development_inputs(data, seed, "val")
    yt = targets(data, seed, "train", train["sequence_id"], cfg["targets"])
    yv = targets(data, seed, "val", val["sequence_id"], cfg["targets"])
    st = score_view(train, dc["regimes"], frozen["scoring"])
    bank = st["reliability"]
    rt = random_scores(train["sequence_id"], bank, seed, cfg["random_control_seed"])
    rv = random_scores(val["sequence_id"], bank, seed, cfg["random_control_seed"])
    views = [(name, development_inputs(data, seed, "val", **kw)) for name, kw in view_specs(cfg)]
    scores = [score_view(v, dc["regimes"], frozen["scoring"]) for _, v in views]
    all_pred = np.empty((len(views), len(GROUPS), len(yv), len(cfg["targets"])), dtype=np.float64)
    all_pred[:, 0] = yt.mean(axis=0)
    saved, history, params = {}, [], {"base": frozen, "groups": {}, "models": {}}
    for gi, group in enumerate(GROUPS[1:], 1):
        raw = sequence_features(train, st, frozen["preprocessing"], group, rt)
        scaler = fit_sequence_scaler(raw)
        x = transform_sequence(raw, scaler)
        xv = [transform_sequence(sequence_features(v, sc, frozen["preprocessing"], group, rv), scaler)
              for (_, v), sc in zip(views, scores)]
        params["groups"][group] = scaler
        for ti, target in enumerate(cfg["targets"]):
            model, detail, curve = train_model(x, yt[:, ti], xv[0], yv[:, ti], cfg,
                cfg["model_seed_offset"] + 10 * seed + ti, cfg["shuffle_seed_offset"] + 10 * seed + ti)
            key = group + "__" + target
            params["models"][key] = detail
            for name, value in model.state_dict().items():
                saved[key + "__" + name] = value.detach().numpy().copy()
            history.extend({"seed": seed, "group": group, "target": target, **r} for r in curve)
            # Each completed model is persisted for interrupted-run recovery, never silently reused.
            np.savez_compressed(sd / "models.npz", **saved)
            write_json(sd / "parameters.json", params)
            write_csv(sd / "training_history.csv", history)
            for vi in range(len(views)):
                all_pred[vi, gi, :, ti] = infer(model, xv[vi], detail["target_center"], detail["target_scale"], cfg["batch_size"])
            print(f"seed={seed} group={group} target={target}: {cfg['epochs']}-epoch budget finished, selected={detail['selected_epoch']}", flush=True)
    # Reconstruct inputs and models from serialized artifacts for every view/target.
    replay = 0.
    stored = json.loads((sd / "parameters.json").read_text())
    bank2 = score_view(train, dc["regimes"], stored["base"]["scoring"])["reliability"]
    rv2 = random_scores(val["sequence_id"], bank2, seed, cfg["random_control_seed"])
    with np.load(sd / "models.npz", allow_pickle=False) as states:
        for gi, group in enumerate(GROUPS[1:], 1):
            for ti, target in enumerate(cfg["targets"]):
                key = group + "__" + target
                detail = stored["models"][key]
                model = PlainLSTM(detail["input_channels"], cfg["hidden_size"])
                model.load_state_dict({n: torch.from_numpy(states[key + "__" + n].copy()) for n in model.state_dict()})
                for vi, (_, view) in enumerate(views):
                    sc = score_view(view, dc["regimes"], stored["base"]["scoring"])
                    x = transform_sequence(sequence_features(view, sc, stored["base"]["preprocessing"], group, rv2), stored["groups"][group])
                    p = infer(model, x, detail["target_center"], detail["target_scale"], cfg["batch_size"])
                    replay = max(replay, float(np.abs(p - all_pred[vi, gi, :, ti]).max()))
    if replay > 1e-7 or not np.isfinite(all_pred).all():
        raise ValueError("prediction replay or finite check failed")
    np.savez_compressed(sd / "validation_predictions.npz", prediction=all_pred, targets=yv,
        sequence_id=val["sequence_id"], views=np.array([n for n, _ in views]),
        groups=np.asarray(GROUPS), target_names=np.asarray(cfg["targets"]))
    write_json(sd / "sequence_ids.json", {s: development_inputs(data, seed, s)["sequence_id"].tolist()
               for s in ("reference_fit", "reference_cal", "train", "val")})
    rows, paired = evaluation_tables(data, seed, cfg, all_pred, yv, val)
    write_csv(sd / "metrics_by_seed.csv", rows)
    write_csv(sd / "paired_by_seed.csv", paired)
    check_files(data, json.loads((data / "manifest.json").read_text()))
    metadata = {"seed": seed, "implementation_commit": commit, "completed_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - started, "learned_models": len(params["models"]),
                "model_replay_max_abs_difference": replay, "test_predictive_evaluation": False,
                "prediction_splits_loaded": ["train", "val", "reference_fit", "reference_cal"],
                "data_files_unchanged": True}
    write_json(sd / "metadata.json", metadata)
    write_json(sd / "manifest.json", {p.name: sha256(p) for p in sorted(sd.iterdir()) if p.is_file()})
    return metadata


def summarize(output: Path, cfg: dict[str, Any]) -> None:
    """Aggregate only complete, verified seeds and seal the run without refitting."""
    if (output / "manifest.json").exists():
        raise FileExistsError("completed run is immutable")
    tables = {name: [] for name in ("metrics_by_seed", "paired_by_seed", "training_history")}
    for seed in cfg["seeds"]:
        sd = output / f"seed_{seed}"
        check_files(sd, json.loads((sd / "manifest.json").read_text()))
        for name in tables:
            with (sd / (name + ".csv")).open() as f:
                tables[name].extend(list(csv.DictReader(f)))
    for name, rows in tables.items():
        write_csv(output / (name + ".csv"), rows)
    for name, raw, keys, values in (
        ("metrics_summary", "metrics_by_seed", ["view", "group", "target", "stratum"], ["mae", "rmse"]),
        ("paired_summary", "paired_by_seed", ["view", "comparison", "target"], ["mae_difference", "degradation_difference"])):
        rows = [{**r, **{v: float(r[v]) for v in values}} for r in tables[raw]]
        write_csv(output / (name + ".csv"), aggregate(rows, keys, values))
    write_json(output / "metadata.json", {"run_id": cfg["run_id"], "data_source": "synthetic_rebuilt",
        "test_predictive_evaluation": False, "validation_used_for_selection": True,
        "seeds": cfg["seeds"], "learned_models": len(cfg["seeds"]) * (len(GROUPS) - 1) * len(cfg["targets"]),
        "epochs_per_model": cfg["epochs"], "data_manifest_sha256": cfg["data_manifest_sha256"],
        "model_replay_max_abs_difference": max(json.loads((output / f"seed_{s}/metadata.json").read_text())["model_replay_max_abs_difference"] for s in cfg["seeds"])})
    write_json(output / "manifest.json", {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()})


def main() -> None:
    """Run exactly one seed or finalize complete development results."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--ridge", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--implementation-commit", required=True)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--seed", type=int)
    action.add_argument("--summarize", action="store_true")
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.summarize:
        initialize(args.data, args.ridge, cfg, args.output, args.implementation_commit)
        summarize(args.output, cfg)
        print("summaries sealed", flush=True)
    else:
        print(json.dumps(run_seed(args.data, args.ridge, cfg, args.output, args.implementation_commit, args.seed), indent=2))


if __name__ == "__main__":
    main()
