"""Paper-aligned synthetic data; hidden process and prediction inputs stay separate.

This module generates data and performs structural checks only. It neither trains
a predictor nor evaluates predictive performance on the reserved test split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SPLITS = ("reference_fit", "reference_cal", "train", "val", "test")
INPUT_KEYS = {"observations", "recorded_regime", "sequence_id", "time_step", "available_time"}


def validate_config(c: dict[str, Any]) -> None:
    """Validate this three-channel illustrative protocol before writing anything."""
    if c.get("data_source") != "synthetic":
        raise ValueError("data_source must be synthetic")
    if not c.get("dataset_id") or "/" in c["dataset_id"]:
        raise ValueError("dataset_id must be a nonempty name")
    seeds = c["seeds"]
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or s < 0 for s in seeds)):
        raise ValueError("seeds must be distinct nonnegative integers")
    if type(c["namespace"]) is not int or c["namespace"] < 0:
        raise ValueError("namespace must be a nonnegative integer")
    if set(c["counts"]) != set(SPLITS) or any(
        type(v) is not int or v < 2 for v in c["counts"].values()
    ):
        raise ValueError("five complete-sequence splits, each with >=2 batches required")
    if (c["features"] != ["temperature", "relative_humidity", "vibration"]
            or c["units"] != ["K", "percent_RH", "mm_per_s"]
            or c["regimes"] != ["warehouse", "transit_smooth", "transit_rough"]):
        raise ValueError("feature/unit/regime order is part of the frozen contract")
    for key in ("sequence_length", "prediction_time", "observation_delay", "sample_time", "report_available_time"):
        if type(c[key]) is not int or c[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    length = c["sequence_length"]
    visible = c["prediction_time"] - c["observation_delay"] + 1
    if not (1 < visible <= c["prediction_time"] + 1 < c["sample_time"] == length
            < c["report_available_time"]):
        raise ValueError("require visible prefix < endpoint sample < report availability")
    for key in ("departure_range", "arrival_range"):
        v = c[key]
        if len(v) != 2 or any(type(x) is not int for x in v) or not 0 < v[0] <= v[1] < length:
            raise ValueError(f"invalid {key}")
    if not c["departure_range"][1] < visible < c["arrival_range"][0]:
        raise ValueError("all prefixes must include departure and precede arrival")
    for key, shape in (("means", (3, 3)), ("stds", (3, 3)),
                       ("batch_offset_std", (3,)), ("measurement_noise_std", (3,)),
                       ("factor_loadings", (3,))):
        a = np.asarray(c[key], dtype=float)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f"invalid shape/nonfinite {key}")
        if key.endswith("std") or key == "stds":
            if (a <= 0).any():
                raise ValueError(f"{key} must be positive")
    if not 0 <= c["ar_rho"] < 1 or np.any(np.abs(c["factor_loadings"]) >= 1):
        raise ValueError("invalid AR coefficient/loadings")
    p = c["proxy"]
    if not np.isfinite(list(p.values())).all() or any(p[k] <= 0 for k in ("coefficient", "batch_std", "assay_std", "null_std")):
        raise ValueError("invalid proxy parameters")
    f = c["faults"]
    if f["types"] != ["bias", "drift", "missing"]:
        raise ValueError("fault types must be bias, drift, missing")
    if (not f["durations"] or any(type(v) is not int or not 1 <= v <= visible for v in f["durations"])
            or not f["magnitudes"] or not np.isfinite(f["magnitudes"]).all()
            or any(v <= 0 for v in f["magnitudes"])):
        raise ValueError("invalid durations/magnitudes")
    if (not c["record_delays"] or c["record_delays"][0] != 0
            or any(type(v) is not int or not 0 <= v < visible for v in c["record_delays"])):
        raise ValueError("invalid record delays")


def _rng(c: dict[str, Any], seed: int, split: str, batch: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(
        [c["namespace"], seed, SPLITS.index(split), batch, stream]
    ))


def exposure_proxy(clean: np.ndarray, c: dict[str, Any]) -> np.ndarray:
    """Return the noise-free illustrative endpoint, not a real quality law."""
    p = c["proxy"]
    return p["intercept"] + p["coefficient"] * np.maximum(clean[..., 0] - p["threshold"], 0).sum(axis=-1)


def generate_split(c: dict[str, Any], seed: int, split: str) -> dict[str, Any]:
    """Generate independent batch streams; never inject faults into the process."""
    validate_config(c)
    if split not in SPLITS or seed not in c["seeds"]:
        raise ValueError("unknown seed/split")
    n, length = c["counts"][split], c["sequence_length"]
    visible = c["prediction_time"] - c["observation_delay"] + 1
    clean = np.empty((n, length, 3), dtype=np.float64)
    observed = np.empty((n, visible, 3), dtype=np.float64)
    regimes = np.zeros((n, length), dtype=np.int8)
    targets = np.empty((n, 2), dtype=np.float64)
    label_noise = np.empty((n, 2), dtype=np.float64)
    ids, events = [], []
    loading, means, stds = (np.asarray(c[k]) for k in ("factor_loadings", "means", "stds"))
    rho = c["ar_rho"]
    for i in range(n):
        sid = f"rq3_{c['namespace']}_{seed}_{split}_{i:06d}"
        ids.append(sid)
        timing = _rng(c, seed, split, i, 0)
        departure = int(timing.integers(c["departure_range"][0], c["departure_range"][1] + 1))
        arrival = int(timing.integers(c["arrival_range"][0], c["arrival_range"][1] + 1))
        regimes[i, departure:arrival] = timing.integers(1, 3)
        process = _rng(c, seed, split, i, 1)
        common = process.normal(size=length)
        for t in range(1, length):
            common[t] = rho * common[t - 1] + np.sqrt(1 - rho**2) * common[t]
        noise = common[:, None] * loading + process.normal(size=(length, 3)) * np.sqrt(1 - loading**2)
        offset = _rng(c, seed, split, i, 2).normal(size=3) * c["batch_offset_std"]
        z = means[regimes[i]] + stds[regimes[i]] * noise + offset
        z[:, 1] = np.clip(z[:, 1], 0, 100)
        z[:, 2] = np.maximum(z[:, 2], 0)
        clean[i] = z
        measurement = _rng(c, seed, split, i, 3).normal(size=(length, 3)) * c["measurement_noise_std"]
        observed[i] = (z + measurement)[:visible]
        label_noise[i, 0] = _rng(c, seed, split, i, 4).normal(scale=c["proxy"]["batch_std"])
        label_noise[i, 1] = _rng(c, seed, split, i, 5).normal(scale=c["proxy"]["assay_std"])
        targets[i, 0] = exposure_proxy(z, c) + label_noise[i].sum()
        targets[i, 1] = _rng(c, seed, split, i, 6).normal(c["proxy"]["null_mean"], c["proxy"]["null_std"])
        if split in ("val", "test"):
            fault = _rng(c, seed, split, i, 7)
            duration = int(fault.choice(c["faults"]["durations"]))
            start = int(fault.integers(0, visible - duration + 1))
            events.append({"sequence_id": sid, "channel": int(fault.integers(3)),
                           "start": start, "end_exclusive": start + duration,
                           "sign": int(fault.choice([-1, 1]))})
    time = np.arange(visible, dtype=np.int16)
    return {"inputs": {"observations": observed, "recorded_regime": regimes[:, :visible].copy(),
                       "sequence_id": np.asarray(ids), "time_step": time,
                       "available_time": time + c["observation_delay"]},
            "truth": {"clean_process": clean, "true_regime": regimes,
                      "sequence_id": np.asarray(ids), "label_noise": label_noise},
            "targets": targets, "events": events}


def fit_fault_scales(inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Fit observed per-regime SDs on reference_fit visible prefixes only."""
    x, r = inputs["observations"], inputs["recorded_regime"]
    scales = []
    for regime in range(3):
        values = x[r == regime]
        if len(values) < 2:
            raise ValueError("reference_fit must cover each regime")
        scales.append(values.std(axis=0, ddof=1))
    result = np.asarray(scales)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("invalid fitted scales")
    return result


def apply_view(inputs: dict[str, np.ndarray], events: list[dict[str, Any]], scales: np.ndarray,
               kind: str = "original", magnitude: float = 1.0,
               record_delay: int = 0) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Build a paired input view and separate evaluation mask without reading truth.

    Delayed regimes use only past recorded entries. Combined delay+fault is
    rejected: the frozen primary comparisons change one factor at a time.
    """
    if set(inputs) != INPUT_KEYS or kind not in ("original", "bias", "drift", "missing"):
        raise ValueError("invalid input schema/view")
    out = {k: v.copy() for k, v in inputs.items()}
    x = out["observations"]
    n, length, channels = x.shape
    if (channels != 3 or not np.isfinite(x).all() or not np.isfinite(magnitude) or magnitude <= 0
            or type(record_delay) is not int or not 0 <= record_delay < length):
        raise ValueError("invalid original values/view parameters")
    if record_delay and kind != "original":
        raise ValueError("do not combine record delay and injected faults in primary protocol")
    mask = np.zeros(x.shape, dtype=bool)
    if kind != "original":
        if scales.shape != (3, 3) or not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("invalid fault scales")
        lookup = {str(s): i for i, s in enumerate(out["sequence_id"])}
        if len(lookup) != n or len(events) != n or {e["sequence_id"] for e in events} != set(lookup):
            raise ValueError("exactly one matched event per batch required")
        for e in events:
            i = lookup[e["sequence_id"]]
            ch, start, end, sign = (int(e[k]) for k in ("channel", "start", "end_exclusive", "sign"))
            if not (0 <= ch < 3 and 0 <= start < end <= length and sign in (-1, 1)):
                raise ValueError("invalid event range/channel/sign")
            mask[i, start:end, ch] = True
            if kind == "missing":
                x[i, start:end, ch] = np.nan
            else:
                amplitude = magnitude * scales[int(out["recorded_regime"][i, start]), ch] * sign
                ramp = np.arange(1, end - start + 1) / (end - start) if kind == "drift" else 1.0
                x[i, start:end, ch] += amplitude * ramp
    if record_delay:
        out["recorded_regime"] = out["recorded_regime"][:, np.maximum(np.arange(length) - record_delay, 0)]
    return out, mask


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(root: Path, seed: int, split: str, kind: str = "original", magnitude: float = 1.0,
                record_delay: int = 0, *, allow_test: bool = False) -> dict[str, np.ndarray]:
    """Load allowlisted prediction inputs; test requires explicit protocol authorization.

    No target, event metadata, mask or evaluation_only value is returned. The
    allow_test switch is an accidental-use guard, not an access-control claim.
    """
    c = yaml.safe_load((root / "config.yaml").read_text())
    validate_config(c)
    if split not in SPLITS or seed not in c["seeds"]:
        raise ValueError("unknown seed/split")
    if split == "test" and not allow_test:
        raise ValueError("test requires explicit authorization under a frozen evaluation protocol")
    if split not in ("val", "test") and (kind != "original" or record_delay != 0):
        raise ValueError("reference and train must remain original in this protocol")
    if record_delay not in c["record_delays"] or (kind in ("bias", "drift") and magnitude not in c["faults"]["magnitudes"]):
        raise ValueError("view not included in frozen config")
    directory = root / f"seed_{seed}" / split
    inputs = _read_npz(directory / "inputs.npz")
    if set(inputs) != INPUT_KEYS or np.any(inputs["available_time"] > c["prediction_time"]):
        raise ValueError("unexpected input fields or future information")
    events = []
    if kind != "original":
        with (directory / "events.csv").open() as stream:
            events = list(csv.DictReader(stream))
    scales = np.asarray(json.loads((root / f"seed_{seed}" / "fault_scales.json").read_text())["scales"])
    return apply_view(inputs, events, scales, kind, magnitude, record_delay)[0]


def write_dataset(c: dict[str, Any], output: Path) -> dict[str, Any]:
    """Write a fresh dataset, provenance and hashes; refuse to overwrite any path."""
    validate_config(c)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
    for seed in c["seeds"]:
        for split in SPLITS:
            d = generate_split(c, seed, split)
            directory = output / f"seed_{seed}" / split
            directory.mkdir(parents=True)
            np.savez_compressed(directory / "inputs.npz", **d["inputs"])
            truth = output / "evaluation_only" / f"seed_{seed}" / split
            truth.mkdir(parents=True)
            np.savez_compressed(truth / "truth.npz", **d["truth"])
            if split == "reference_fit":
                _json(directory.parent / "fault_scales.json", {
                    "source_split": "reference_fit", "source_inputs_sha256": _sha(directory / "inputs.npz"),
                    "ddof": 1, "visible_only": True, "scales": fit_fault_scales(d["inputs"]).tolist()})
            if split in ("train", "val", "test"):
                rows = [{"sequence_id": str(sid), "arrival_proxy": float(y[0]), "null_proxy": float(y[1]),
                         "prediction_time": c["prediction_time"], "sample_time": c["sample_time"],
                         "report_available_time": c["report_available_time"]}
                        for sid, y in zip(d["inputs"]["sequence_id"], d["targets"], strict=True)]
                _csv(directory / "targets.csv", rows)
            if d["events"]:
                _csv(directory / "events.csv", d["events"])
    source = Path(__file__).resolve()
    repo = source.parent.parent
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", None
    _json(output / "metadata.json", {"dataset_id": c["dataset_id"], "data_source": "synthetic",
          "source_commit": commit, "tracked_worktree_dirty": dirty, "source_sha256": _sha(source),
          "python": platform.python_version(), "dependencies": {p: importlib.metadata.version(p) for p in ("numpy", "PyYAML")},
          "test_predictive_evaluation": False, "test_structural_audit": True,
          "target_unit": "arbitrary_proxy_unit", "time_unit": "abstract_step"})
    hashes = {str(p.relative_to(output)): _sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    _json(output / "manifest.json", hashes)
    report = audit_dataset(output)
    _json(output / "audit.json", report)
    return report


def audit_dataset(root: Path) -> dict[str, Any]:
    """Verify all artifacts and paired views structurally, without predictor metrics."""
    c = yaml.safe_load((root / "config.yaml").read_text())
    validate_config(c)
    manifest = json.loads((root / "manifest.json").read_text())
    for relative, expected in manifest.items():
        if _sha(root / relative) != expected:
            raise ValueError(f"hash mismatch: {relative}")
    files = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} - {"manifest.json", "audit.json"}
    if files != set(manifest):
        raise ValueError("manifest/file set mismatch")
    ids: set[str] = set()
    checks, batches = 0, 0
    visible = c["prediction_time"] - c["observation_delay"] + 1
    counts: dict[str, int] = {}
    for seed in c["seeds"]:
        reference = load_inputs(root, seed, "reference_fit")
        scales = fit_fault_scales(reference)
        saved = json.loads((root / f"seed_{seed}" / "fault_scales.json").read_text())
        if not np.array_equal(scales, saved["scales"]) or saved["source_inputs_sha256"] != _sha(root / f"seed_{seed}" / "reference_fit" / "inputs.npz"):
            raise ValueError("scale provenance mismatch")
        for split in SPLITS:
            inputs = load_inputs(root, seed, split, allow_test=True)
            n = c["counts"][split]
            x = inputs["observations"]
            if (x.shape != (n, visible, 3) or not np.isfinite(x).all()
                    or inputs["recorded_regime"].shape != (n, visible)
                    or not np.isin(inputs["recorded_regime"], [0, 1, 2]).all()):
                raise ValueError("invalid input shape/values")
            current = set(map(str, inputs["sequence_id"]))
            if len(current) != n or ids & current:
                raise ValueError("duplicate IDs across splits/seeds")
            ids.update(current)
            if not np.array_equal(inputs["time_step"], np.arange(visible)) or not np.array_equal(inputs["available_time"], np.arange(visible) + c["observation_delay"]):
                raise ValueError("invalid availability timeline")
            truth = _read_npz(root / "evaluation_only" / f"seed_{seed}" / split / "truth.npz")
            if (truth["clean_process"].shape != (n, c["sequence_length"], 3)
                    or not np.isfinite(truth["clean_process"]).all()
                    or not np.array_equal(truth["sequence_id"], inputs["sequence_id"])
                    or not np.array_equal(truth["true_regime"][:, :visible], inputs["recorded_regime"])):
                raise ValueError("truth/input identity mismatch")
            directory = root / f"seed_{seed}" / split
            if split in ("train", "val", "test"):
                with (directory / "targets.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                if [r["sequence_id"] for r in rows] != list(inputs["sequence_id"]):
                    raise ValueError("target/sequence identity mismatch")
                actual = np.asarray([[float(r["arrival_proxy"]), float(r["null_proxy"])] for r in rows])
                expected = exposure_proxy(truth["clean_process"], c) + truth["label_noise"].sum(axis=1)
                if not np.isfinite(actual).all() or not np.allclose(expected, actual[:, 0], rtol=0, atol=1e-12):
                    raise ValueError("invalid target formula/value")
                for row in rows:
                    if any(int(row[k]) != c[k] for k in ("prediction_time", "sample_time", "report_available_time")):
                        raise ValueError("invalid target timeline")
            if split in ("val", "test"):
                with (directory / "events.csv").open() as stream:
                    events = list(csv.DictReader(stream))
                for kind in c["faults"]["types"]:
                    for mag in ([1.0] if kind == "missing" else c["faults"]["magnitudes"]):
                        view, mask = apply_view(inputs, events, scales, kind, mag)
                        changed = np.isnan(view["observations"]) | (view["observations"] != x)
                        if not np.array_equal(changed, mask) or not np.array_equal(view["recorded_regime"], inputs["recorded_regime"]):
                            raise ValueError("paired mask/record mismatch")
                        checks += 1
                for delay in c["record_delays"]:
                    view, mask = apply_view(inputs, [], scales, record_delay=delay)
                    if not np.array_equal(view["observations"], x) or mask.any():
                        raise ValueError("record delay changed observations")
                    checks += 1
            batches += n
            counts[f"seed_{seed}/{split}"] = n
    return {"status": "pass", "data_source": "synthetic", "files_hashed": len(manifest),
            "independent_batches": batches, "full_latent_time_rows": batches * c["sequence_length"],
            "visible_input_time_rows": batches * visible, "view_checks": checks,
            "counts": counts, "test_predictive_evaluation": False,
            "scope": "structural, identity, label formula, causality, paired mask and integrity only"}


def main() -> None:
    """Generate a fresh data version, or re-audit an existing one without mutation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.audit is not None:
        if args.config is not None or args.output is not None:
            parser.error("--audit cannot be combined with generation")
        report = audit_dataset(args.audit)
    else:
        if args.config is None or args.output is None:
            parser.error("generation requires --config and --output")
        report = write_dataset(yaml.safe_load(args.config.read_text()), args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
