"""Independently recompute stored development metrics without fitting models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    """Return SHA-256 of saved artifact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(run: Path, data: Path) -> dict:
    """Cross-check predictions, original labels, tables, selections and summaries."""
    manifest = json.loads((run / "manifest.json").read_text())
    actual = {str(p.relative_to(run)) for p in run.rglob("*") if p.is_file()} - {"manifest.json"}
    assert actual == set(manifest)
    assert all(digest(run / p) == h for p, h in manifest.items())
    meta = json.loads((run / "metadata.json").read_text())
    cfg = json.loads((run / "config.json").read_text())
    assert digest(data / "manifest.json") == cfg["data_manifest_sha256"]
    assert all(digest(data / p) == h for p, h in json.loads((run / "input_manifest.json").read_text()).items())
    with (run / "metrics_by_seed.csv").open() as f:
        metric_rows = list(csv.DictReader(f))
    lookup = {(int(r["seed"]), r["view"], r["group"], r["target"], r["stratum"]): r for r in metric_rows}
    assert len(lookup) == len(metric_rows)
    checked = 0
    for seed in cfg["seeds"]:
        with (data / f"seed_{seed}/val/targets.csv").open() as f:
            labels = list(csv.DictReader(f))
        with (data / f"seed_{seed}/val/events.csv").open() as f:
            events = list(csv.DictReader(f))
        with np.load(data / f"seed_{seed}/val/inputs.npz", allow_pickle=False) as inputs:
            cutoff = inputs["recorded_regime"][:, -1]
        with np.load(run / f"seed_{seed}/validation_predictions.npz", allow_pickle=False) as stored:
            ids, y, pred = stored["sequence_id"], stored["targets"], stored["prediction"]
            assert list(ids) == [r["sequence_id"] for r in labels] == [r["sequence_id"] for r in events]
            assert np.array_equal(y, [[float(r[t]) for t in cfg["targets"]] for r in labels])
            assert np.isfinite(pred).all()
            for vi, view in enumerate(stored["views"]):
                masks = {"all": np.ones(len(ids), bool)}
                for ri, regime in enumerate(("warehouse", "transit_smooth", "transit_rough")):
                    masks["cutoff_regime:" + regime] = cutoff == ri
                if str(view).startswith(("bias", "drift", "missing")):
                    for ci, channel in enumerate(("temperature", "relative_humidity", "vibration")):
                        masks["fault_channel:" + channel] = np.asarray([int(r["channel"]) == ci for r in events])
                    for duration in (4, 12, 24):
                        masks[f"duration:{duration}"] = np.asarray([int(r["end_exclusive"]) - int(r["start"]) == duration for r in events])
                for gi, group in enumerate(stored["groups"]):
                    for ti, target in enumerate(stored["target_names"]):
                        for stratum, mask in masks.items():
                            if not mask.any():
                                continue
                            row = lookup[seed, str(view), str(group), str(target), stratum]
                            err = pred[vi, gi, mask, ti] - y[mask, ti]
                            assert int(row["batches"]) == mask.sum()
                            assert abs(float(row["mae"]) - np.mean(np.abs(err))) < 1e-10
                            assert abs(float(row["rmse"]) - np.sqrt(np.mean(err * err))) < 1e-10
                            checked += 1
    assert checked == len(metric_rows)
    with (run / "training_history.csv").open() as f:
        selection = list(csv.DictReader(f))
    for seed in cfg["seeds"]:
        for group in cfg["groups"][1:]:
            for target in cfg["targets"]:
                candidates = [r for r in selection if int(r["seed"]) == seed and r["group"] == group and r["target"] == target]
                chosen = [r for r in candidates if r["selected"] == "True"]
                assert len(candidates) == cfg["epochs"] and len(chosen) == 1
                assert float(chosen[0]["original_val_mae"]) == min(float(r["original_val_mae"]) for r in candidates)
                assert abs(float(chosen[0]["original_val_mae"]) - float(lookup[seed, "original", group, target, "all"]["mae"])) < 1e-10
    with (run / "paired_by_seed.csv").open() as f:
        paired = list(csv.DictReader(f))
    for r in paired:
        s, v, t = int(r["seed"]), r["view"], r["target"]
        better, base = r["comparison"].split("-")
        difference = float(lookup[s, v, better, t, "all"]["mae"]) - float(lookup[s, v, base, t, "all"]["mae"])
        baseline = float(lookup[s, "original", better, t, "all"]["mae"]) - float(lookup[s, "original", base, t, "all"]["mae"])
        assert abs(float(r["mae_difference"]) - difference) < 1e-10
        assert abs(float(r["degradation_difference"]) - difference + baseline) < 1e-10
    summary_checked = 0
    for name, raw, keys, values in (
        ("metrics_summary.csv", metric_rows, ["view", "group", "target", "stratum"], ["mae", "rmse"]),
        ("paired_summary.csv", paired, ["view", "comparison", "target"], ["mae_difference", "degradation_difference"]),
    ):
        with (run / name).open() as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            group = [r for r in raw if all(r[k] == row[k] for k in keys)]
            assert len(group) == int(row["data_repeats"]) == len(cfg["seeds"])
            for value in values:
                numbers = np.asarray([float(r[value]) for r in group])
                assert abs(float(row[value + "_mean"]) - numbers.mean()) < 1e-10
                assert abs(float(row[value + "_sd"]) - numbers.std(ddof=1)) < 1e-10
            summary_checked += 1
    assert meta["test_predictive_evaluation"] is False
    for seed in cfg["seeds"]:
        sm = json.loads((run / f"seed_{seed}/metadata.json").read_text())
        assert set(sm["prediction_splits_loaded"]) == {"reference_fit", "reference_cal", "train", "val"}
        assert sm["test_predictive_evaluation"] is False
        assert sm["model_replay_max_abs_difference"] <= 1e-7
        params = json.loads((run / f"seed_{seed}/parameters.json").read_text())
        for group in cfg["groups"][1:]:
            for target in cfg["targets"]:
                rows = [r for r in selection if int(r["seed"]) == seed and r["group"] == group and r["target"] == target]
                assert [int(r["epoch"]) for r in rows] == list(range(1, cfg["epochs"] + 1))
                best = min(rows, key=lambda r: float(r["original_val_mae"]))
                assert params["models"][group + "__" + target]["selected_epoch"] == int(best["epoch"])
                assert np.isfinite([float(r["train_online_standardized_mse"]) for r in rows]).all()
        with np.load(run / f"seed_{seed}/models.npz", allow_pickle=False) as models:
            assert all(np.isfinite(models[k]).all() for k in models.files)
    return {"status": "pass", "run_manifest_sha256": digest(run / "manifest.json"),
            "output_files_hashed": len(manifest), "input_files_hashed": len(json.loads((run / "input_manifest.json").read_text())),
            "metric_rows_recomputed": checked, "paired_rows_recomputed": len(paired),
            "summary_rows_recomputed": summary_checked, "selection_rows_checked": len(selection),
            "test_predictive_evaluation": False,
            "scope": "post-run artifact and arithmetic verification, no model refitting"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.run, args.data), indent=2))
