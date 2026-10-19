"""Semantic audit, numerical ridge, causal features and development-run checks."""
from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge
import yaml

from src.rq3_paired_data import write_dataset
from src.rq3_release_audit import sha256, strict_audit
from src.rq3_prediction_pilot import (
    GROUPS, development_inputs, features, fit_preprocessor, fit_score,
    predict, random_scores, ridge_path, run_pilot, score_view, select_model, targets, view_specs,
)


@pytest.fixture
def fixture_data(tmp_path):
    config = yaml.safe_load(Path("configs/rq3_paired_data.yaml").read_text())
    config["seeds"] = [42]
    config["counts"] = dict.fromkeys(config["counts"], 16)
    root = tmp_path / "fixture"
    write_dataset(config, root)
    cfg = yaml.safe_load(Path("configs/rq3_ridge_pilot_20261019.yaml").read_text())
    cfg.update(seeds=[42], dataset_id=config["dataset_id"],
               data_manifest_sha256=sha256(root / "manifest.json"), run_id="unit_test_only")
    return root, config, cfg


def refresh_manifest(root, path):
    manifest = json.loads((root / "manifest.json").read_text())
    manifest[str(path.relative_to(root))] = sha256(path)
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_strict_null_semantics_even_after_manifest_refresh(fixture_data):
    root, config, cfg = fixture_data
    path = root / "seed_42/train/targets.csv"
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        row["null_proxy"] = row["arrival_proxy"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    refresh_manifest(root, path)
    with pytest.raises(ValueError, match="null label"):
        strict_audit(root, expected_dataset_id=config["dataset_id"],
                     expected_source_sha256=cfg["generator_sha256"])


@pytest.mark.parametrize("key,value", [("dataset_id", "wrong"), ("source_sha256", "0" * 64), ("data_source", "real")])
def test_strict_metadata_semantics(fixture_data, key, value):
    root, config, cfg = fixture_data
    path = root / "metadata.json"
    meta = json.loads(path.read_text())
    meta[key] = value
    path.write_text(json.dumps(meta))
    refresh_manifest(root, path)
    with pytest.raises(ValueError, match="metadata"):
        strict_audit(root, expected_dataset_id=config["dataset_id"],
                     expected_source_sha256=cfg["generator_sha256"])


def test_external_manifest_pin(fixture_data):
    root, config, cfg = fixture_data
    with pytest.raises(ValueError, match="external manifest"):
        strict_audit(root, expected_dataset_id=config["dataset_id"],
                     expected_source_sha256=cfg["generator_sha256"], expected_manifest_sha256="0" * 64)


def test_ridge_path_matches_library_and_intercept():
    rng = np.random.default_rng(44)
    x = rng.normal(size=(20, 31))
    x[:, 0] = 1.
    y = rng.normal(size=(20, 2)) + [15., -30.]
    alphas = [1., 100., 10000.]
    fitted = ridge_path(x, y, alphas)
    z = (x - fitted["center"]) / fitted["scale"]
    for i, alpha in enumerate(alphas):
        ref = Ridge(alpha=alpha, solver="svd").fit(z, y)
        candidate = {**fitted, "weights": fitted["weights"][i]}
        np.testing.assert_allclose(predict(x, candidate), ref.predict(z), rtol=0, atol=1e-10)
    np.testing.assert_allclose(fitted["weights"][:, 0], 0, atol=1e-12)
    # One-dimensional closed form: standardized x has sum of squares 3.
    one_x = np.array([[-1.], [0.], [1.]])
    one_y = 2 * one_x + 3
    one = ridge_path(one_x, one_y, [3.])
    expected = (one_x + 3)
    np.testing.assert_allclose(predict(one_x, {**one, "weights": one["weights"][0]}), expected, atol=1e-12)


def test_selection_is_per_target_original_mae():
    path = {"center": np.zeros(1), "scale": np.ones(1), "intercept": np.zeros(2),
            "weights": np.array([[[1., 3.]], [[2., 2.]], [[3., 1.]]])}
    val_x = np.array([[1.], [2.]])
    val_y = np.array([[1., 1.], [2., 2.]])
    model, losses = select_model(path, val_x, val_y, [1., 100., 10000.])
    np.testing.assert_array_equal(model["alphas"], [1., 10000.])
    np.testing.assert_array_equal(predict(val_x, model), val_y)


def test_test_prediction_guards_before_filesystem_access(tmp_path):
    with pytest.raises(ValueError, match="excludes test"):
        development_inputs(tmp_path, 42, "test")
    with pytest.raises(ValueError, match="excludes test"):
        targets(tmp_path, 42, "test", np.array([]), ["arrival_proxy"])


def test_feature_isolation_causality_and_available_redundancy(fixture_data):
    root, data_cfg, cfg = fixture_data
    train = development_inputs(root, 42, "train")
    val = development_inputs(root, 42, "val")
    scorer = fit_score(root, 42, data_cfg["regimes"], cfg)
    prep = fit_preprocessor(train)
    first = score_view(val, data_cfg["regimes"], scorer)
    altered = deepcopy(val)
    altered["observations"][:, 60:] += 1e5
    later = score_view(altered, data_cfg["regimes"], scorer)
    np.testing.assert_array_equal(first["reliability"][:, :60], later["reliability"][:, :60])
    extra = {**val, "targets": np.ones(len(val["sequence_id"])), "hidden_process": np.ones(3)}
    np.testing.assert_array_equal(features(val, first, prep, "B3"), features(extra, first, prep, "B3"))
    missing = development_inputs(root, 42, "val", kind="missing")
    sc = score_view(missing, data_cfg["regimes"], scorer)
    np.testing.assert_array_equal(sc["available"], np.isfinite(missing["observations"]))
    train_score = score_view(train, data_cfg["regimes"], scorer)
    y = targets(root, 42, "train", train["sequence_id"], cfg["targets"])
    predictions = []
    for group in ("B1", "B2"):
        model = ridge_path(features(train, train_score, prep, group), y, [100.])
        predictions.append(predict(features(missing, sc, prep, group), {**model, "weights": model["weights"][0]}))
    np.testing.assert_allclose(*predictions, rtol=0, atol=1e-9)


def test_random_control_stable_across_order_and_view():
    bank = np.arange(5 * 4 * 3).reshape(5, 4, 3)
    ids = np.asarray(["a", "b", "c"])
    first = random_scores(ids, bank, 42, 901)
    np.testing.assert_array_equal(first[::-1], random_scores(ids[::-1], bank, 42, 901))


def test_amplitudes_are_paired_view_conditions(fixture_data):
    root, _, cfg = fixture_data
    original = development_inputs(root, 42, "val")["observations"]
    one = development_inputs(root, 42, "val", kind="bias", magnitude=1.)["observations"]
    four = development_inputs(root, 42, "val", kind="bias", magnitude=4.)["observations"]
    np.testing.assert_allclose(four - original, 4 * (one - original), rtol=0, atol=2e-13)
    assert len(view_specs(cfg)) == 10


def test_smoke_pipeline_preserves_inputs_and_archives_replay(fixture_data, tmp_path):
    root, _, cfg = fixture_data
    before = {str(p): sha256(p) for p in root.rglob("*") if p.is_file()}
    output = tmp_path / "unit_smoke"
    # Sentinel commit is confined to temporary unit-test output, never a release.
    metadata = run_pilot(root, cfg, output, "0" * 40)
    assert metadata["test_predictive_evaluation"] is False
    assert metadata["base_train_batches"] == metadata["base_val_batches"] == 16
    assert metadata["model_replay_max_abs_difference"] == 0
    assert metadata["B1_B2_max_abs_prediction_difference"] < 1e-8
    with np.load(output / "seed_42/validation_predictions.npz", allow_pickle=False) as p:
        assert p["prediction"].shape == (10, len(GROUPS), 16, 2)
    for name, expected in json.loads((output / "manifest.json").read_text()).items():
        assert sha256(output / name) == expected
    assert before == {str(p): sha256(p) for p in root.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        run_pilot(root, cfg, output, "0" * 40)
