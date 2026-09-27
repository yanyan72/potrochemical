"""Validated, synthetic-only model bundles and causal complete-sequence scoring."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .temporal_reliability import score_temporal

SCHEMA = "petrochemical.scoring.v1"
IDENTITY = ["sequence_id", "time_step", "recorded_regime"]


def _vector(value: Any, p: int, *, positive: bool = False) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.shape != (p,) or not np.isfinite(a).all() or (positive and np.any(a <= 0)):
        raise ValueError("Invalid finite model vector or nonpositive scale/threshold.")
    return a


def validate_bundle(bundle: dict[str, Any]) -> None:
    """Reject incompatible schemas, nonfinite parameters and invalid covariance."""
    try:
        if bundle["schema"] != SCHEMA or bundle["data_source"] != "synthetic":
            raise ValueError("Only the synthetic v1 bundle is supported.")
        features, units = bundle["features"], bundle["units"]
        if (not isinstance(features, list) or len(features) < 2 or
                any(not isinstance(s, str) or not s.strip() for s in features + units) or
                len(set(features)) != len(features) or len(units) != len(features) or
                set(features) & set(IDENTITY)):
            raise ValueError("Unique features and matching units required.")
        if bundle["time_unit"] != "abstract_sampling_step" or bundle["sample_interval"] != 1:
            raise ValueError("This release uses unit-spaced abstract steps.")
        p, model = len(features), bundle["model"]
        ref = model["reference"]
        if ref["method"] not in {"regime_marginal", "regime_conditional"}:
            raise ValueError("Unsupported residual method.")
        if (type(model["alpha"]) not in (float, int) or model["alpha"] not in (.2, 1.) or
                model["quantile"] != .99 or model["initial_state"] != "zero" or
                model["reset_on_record_change"] is not True):
            raise ValueError("Release supports frozen point/EWMA reset parameters only.")
        _vector(ref["scaler_center"], p)
        _vector(ref["scaler_scale"], p, positive=True)
        if not ref["references"] or set(ref["references"]) != set(model["thresholds"]):
            raise ValueError("Reference/threshold regimes must match.")
        for regime, block in ref["references"].items():
            if not isinstance(regime, str) or not regime.strip():
                raise ValueError("Nonempty regime names required.")
            _vector(block["center"], p)
            _vector(model["thresholds"][regime], p, positive=True)
            matrices = [np.asarray(block[k], float) for k in ("covariance", "precision")]
            for a in matrices:
                if (a.shape != (p, p) or not np.isfinite(a).all() or
                        not np.allclose(a, a.T, rtol=1e-10, atol=1e-12) or
                        np.linalg.eigvalsh(a).min() <= 0):
                    raise ValueError("Require symmetric positive definite matrices.")
            if not np.allclose(matrices[0] @ matrices[1], np.eye(p), rtol=1e-7, atol=1e-7):
                raise ValueError("Covariance and precision disagree.")
    except (KeyError, TypeError, np.linalg.LinAlgError) as exc:
        raise ValueError("Malformed model bundle.") from exc


def make_bundle(model: dict[str, Any], features: list[str], units: list[str],
                provenance: dict[str, Any]) -> dict[str, Any]:
    """Wrap existing training-only parameters without refitting or changing them."""
    bundle = {"schema": SCHEMA, "data_source": "synthetic", "features": features,
              "units": units, "time_unit": "abstract_sampling_step", "sample_interval": 1,
              "score_meaning": "observation_consistency_not_probability_or_quality",
              "model": deepcopy(model), "provenance": deepcopy(provenance)}
    validate_bundle(bundle)
    return bundle


def load_bundle(path: str | Path) -> dict[str, Any]:
    """Read a JSON model without executable serialization and validate it."""
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_bundle(bundle)
    return bundle


def input_contract(bundle: dict[str, Any]) -> dict[str, Any]:
    """Return required sidecar declarations; units are never inferred or converted."""
    return {k: deepcopy(bundle[k]) for k in
            ("schema", "data_source", "features", "units", "time_unit", "sample_interval")}


def score_frame(frame: pd.DataFrame, bundle: dict[str, Any],
                contract: dict[str, Any]) -> pd.DataFrame:
    """Score complete ordered trajectories; return one row per input row/sensor.

    No sorting, imputation, unit conversion, quality labels or training occurs.
    All steps of each sequence must be present, including explicit missing rows.
    Calls are independent batches: splitting a sequence across calls resets state.
    """
    validate_bundle(bundle)
    if contract != input_contract(bundle):
        raise ValueError("Input contract must exactly match model features, units and sampling.")
    features = bundle["features"]
    if (frame.empty or frame.columns.duplicated().any() or
            set(frame.columns) != set(IDENTITY + features)):
        raise ValueError("Expected nonempty observations with exactly the documented columns.")
    data = frame.copy().reset_index(drop=True)
    for key in ("sequence_id", "recorded_regime"):
        if not data[key].map(lambda s: isinstance(s, str) and bool(s) and s == s.strip()).all():
            raise ValueError(f"Nonempty, unpadded strings required: {key}")
    for key in ["time_step", *features]:
        if data[key].map(lambda v: isinstance(v, (bool, np.bool_))).any():
            raise ValueError("Boolean measurements/time are not valid numeric observations.")
        data[key] = pd.to_numeric(data[key], errors="raise")
    steps = data.time_step.to_numpy(dtype=float)
    if (not np.isfinite(steps).all() or np.any(steps < 0) or
            np.any(steps > 2**53-1) or np.any(steps != np.floor(steps))):
        raise ValueError("Require nonnegative exact integer time steps.")
    data["time_step"] = steps.astype(np.int64)
    if np.isinf(data[features].to_numpy(float)).any():
        raise ValueError("Infinite measurements rejected; use NaN for missing.")
    output = []
    model = bundle["model"]
    for _, group in data.groupby("sequence_id", sort=False):
        if np.any(np.diff(group.time_step.to_numpy()) != 1):
            raise ValueError("Steps must be unique, increasing and consecutive within each sequence.")
        x = group[features].to_numpy(float)[None]
        records = group.recorded_regime.to_numpy()[None]
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                result = score_temporal(x, records, model)
            except FloatingPointError as exc:
                raise ValueError("Observations exceed supported numeric range.") from exc
        for t, (index, row) in enumerate(group.iterrows()):
            for j, feature in enumerate(features):
                available = bool(result["available"][0, t, j])
                status = "ok" if available else (
                    "unknown_regime" if row.recorded_regime not in model["thresholds"] else
                    "missing_sensor" if np.isnan(x[0, t, j]) else "missing_dependency")
                output.append({"input_row": int(index), "sequence_id": row.sequence_id,
                    "time_step": int(row.time_step), "recorded_regime": row.recorded_regime,
                    "sensor": feature, "unit": bundle["units"][j], "observed_value": x[0, t, j],
                    "consistency_score": result["reliability"][0, t, j],
                    "deviation_ratio": result["deviation_ratio"][0, t, j],
                    "available": available, "alarm": bool(result["alarm"][0, t, j]) if available else None,
                    "status": status, "_sensor_order": j})
    result_frame = pd.DataFrame(output).sort_values(["input_row", "_sensor_order"])
    result_frame["alarm"] = result_frame.alarm.astype("boolean")
    return result_frame.drop(columns="_sensor_order").reset_index(drop=True)


def read_observations(path: str | Path, features: list[str]) -> pd.DataFrame:
    """Read CSV while preserving literal identifiers and explicit sensor missingness."""
    import csv
    with Path(path).open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV column names.")
    return pd.read_csv(path, dtype=str, keep_default_na=False,
                       na_values={f: ["", "NaN", "nan"] for f in features})
