"""Numerical recurrence, leakage boundaries, deterministic fitting and archival."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.rq3_lstm_pilot import (PlainLSTM, configure_runtime, fit_sequence_scaler,
    transform_sequence, train_model, infer, run_seed, summarize)
from src.rq3_paired_data import write_dataset
from src.rq3_prediction_pilot import run_pilot
from src.rq3_release_audit import sha256


def config():
    """Read the registered protocol with small training budgets only for tests."""
    c = yaml.safe_load(Path("configs/rq3_lstm_pilot_20261020.yaml").read_text())
    c.update(hidden_size=4, epochs=3, batch_size=8)
    return c


def test_lstm_recurrence_matches_hand_calculation():
    model = PlainLSTM(1, 1).double()
    with torch.no_grad():
        model.lstm.weight_ih_l0.copy_(torch.tensor([[.2], [-.3], [.4], [.5]], dtype=torch.float64))
        model.lstm.weight_hh_l0.copy_(torch.tensor([[.1], [.2], [-.1], [.3]], dtype=torch.float64))
        model.lstm.bias_ih_l0.copy_(torch.tensor([.1, -.1, .2, .3], dtype=torch.float64))
        model.lstm.bias_hh_l0.zero_()
        model.head.weight.fill_(1.7)
        model.head.bias.fill_(.2)
    seq = np.array([.4, -.7, .8])
    h, c, hs = 0., 0., []
    for x in seq:
        gate = np.array([.2, -.3, .4, .5]) * x + np.array([.1, .2, -.1, .3]) * h + [.1, -.1, .2, .3]
        i, f, o = 1 / (1 + np.exp(-gate[[0, 1, 3]]))
        c = f * c + i * np.tanh(gate[2])
        h = o * np.tanh(c)
        hs.append(h)
    actual = model(torch.tensor(seq[None, :, None], dtype=torch.float64)).detach().numpy()[0]
    assert abs(actual - (1.7 * np.mean(hs) + .2)) < 1e-12


def test_training_scaler_and_unseen_constant_columns():
    x = np.arange(24.).reshape(3, 4, 2)
    x[..., 1] = 0.
    scaler = fit_sequence_scaler(x)
    before = deepcopy(scaler)
    val = x.copy()
    val[..., 0] += 1000
    val[..., 1] = 5
    out = transform_sequence(val, scaler)
    assert scaler == before
    np.testing.assert_array_equal(out[..., 1], 0.)
    np.testing.assert_allclose(out[..., 0], (val[..., 0] - x[..., 0].mean()) / x[..., 0].std(), rtol=1e-6)
    assert out[..., 0].mean() > 100


def test_repeated_training_selection_and_gradient_updates():
    cfg = config()
    configure_runtime(cfg)
    rng = np.random.default_rng(70)
    x = rng.normal(size=(24, 5, 2)).astype(np.float32)
    y = x[:, :, 0].mean(axis=1).astype(float) + 10.
    a, pa, ha = train_model(x, y, x, y, cfg, 123, 321)
    b, pb, hb = train_model(x, y, x, y, cfg, 123, 321)
    assert pa == pb and ha == hb
    np.testing.assert_array_equal(infer(a, x, pa["target_center"], pa["target_scale"], 8), infer(b, x, pb["target_center"], pb["target_scale"], 8))
    torch.manual_seed(123)
    initial = PlainLSTM(2, 4)
    assert any(not torch.equal(a.state_dict()[k], initial.state_dict()[k]) for k in a.state_dict())
    best = min(ha, key=lambda r: r["original_val_mae"])
    assert pa["selected_epoch"] == best["epoch"]
    assert abs(pa["target_center"] - y.mean()) < 1e-12


def test_test_evaluation_rejected_before_data_access(tmp_path):
    cfg = config()
    cfg["evaluation_split"] = "test"
    with pytest.raises(ValueError, match="excludes test"):
        run_seed(tmp_path, tmp_path, cfg, tmp_path / "out", "0" * 40, 42)
    assert not (tmp_path / "out").exists()


def test_small_archival_integration_without_test_prediction(tmp_path, monkeypatch):
    dc = yaml.safe_load(Path("configs/rq3_paired_data.yaml").read_text())
    dc["seeds"] = [42]
    dc["counts"] = dict.fromkeys(dc["counts"], 16)
    dc["dataset_id"] = "lstm_unit_fixture_only"
    data = tmp_path / "data"
    write_dataset(dc, data)
    rc = yaml.safe_load(Path("configs/rq3_ridge_pilot_20261019.yaml").read_text())
    rc.update(seeds=[42], dataset_id=dc["dataset_id"], data_manifest_sha256=sha256(data / "manifest.json"))
    ridge = tmp_path / "ridge"
    run_pilot(data, rc, ridge, "0" * 40)
    cfg = config()
    cfg.update(seeds=[42], dataset_id=dc["dataset_id"], epochs=2,
               data_manifest_sha256=sha256(data / "manifest.json"), ridge_manifest_sha256=sha256(ridge / "manifest.json"))
    import src.rq3_lstm_pilot as module
    original = module.development_inputs
    loaded = []
    def guard(root, seed, split, **kw):
        assert split != "test"
        loaded.append(split)
        return original(root, seed, split, **kw)
    monkeypatch.setattr(module, "development_inputs", guard)
    first, second = tmp_path / "first", tmp_path / "second"
    before = sha256(data / "manifest.json")
    report = run_seed(data, ridge, cfg, first, "0" * 40, 42)
    assert report["model_replay_max_abs_difference"] == 0
    assert report["learned_models"] == 12
    with pytest.raises(FileExistsError):
        run_seed(data, ridge, cfg, first, "0" * 40, 42)
    run_seed(data, ridge, cfg, second, "0" * 40, 42)
    with np.load(first / "seed_42/validation_predictions.npz") as a, np.load(second / "seed_42/validation_predictions.npz") as b:
        np.testing.assert_array_equal(a["prediction"], b["prediction"])
    assert sha256(data / "manifest.json") == before
    assert set(loaded) == {"train", "val", "reference_fit", "reference_cal"}
    summarize(first, cfg)
    manifest = json.loads((first / "manifest.json").read_text())
    assert all(sha256(first / p) == h for p, h in manifest.items())
    with pytest.raises(FileExistsError):
        summarize(first, cfg)
