"""Causality, pairing and reproducibility checks for the new synthetic protocol."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.rq3_paired_data import (
    INPUT_KEYS, SPLITS, apply_view, audit_dataset, exposure_proxy, fit_fault_scales,
    generate_split, load_inputs, validate_config, write_dataset,
)


@pytest.fixture
def config():
    c = yaml.safe_load(Path("configs/rq3_paired_data.yaml").read_text())
    c["counts"] = dict.fromkeys(SPLITS, 16)
    c["seeds"] = [42]
    return c


def test_independent_streams_and_test_size(config):
    first = generate_split(config, 42, "train")
    changed = deepcopy(config)
    changed["counts"]["test"] += 10
    changed["faults"]["durations"] = [2]
    changed["record_delays"] = [0, 3]
    second = generate_split(changed, 42, "train")
    for section in ("inputs", "truth"):
        for key in first[section]:
            np.testing.assert_array_equal(first[section][key], second[section][key])
    np.testing.assert_array_equal(first["targets"], second["targets"])


def test_fault_plan_does_not_change_process_or_labels(config):
    first = generate_split(config, 42, "val")
    changed = deepcopy(config)
    changed["faults"]["durations"] = [2]
    changed["faults"]["magnitudes"] = [10]
    second = generate_split(changed, 42, "val")
    np.testing.assert_array_equal(first["inputs"]["observations"], second["inputs"]["observations"])
    np.testing.assert_array_equal(first["targets"], second["targets"])
    assert first["events"] != second["events"]


def test_proxy_formula_and_legal_temperature_change(config):
    d = generate_split(config, 42, "train")
    clean = d["truth"]["clean_process"]
    expected = 10 + .03 * np.maximum(clean[:, :, 0] - 298, 0).sum(axis=1)
    np.testing.assert_allclose(expected, exposure_proxy(clean, config), rtol=0, atol=1e-12)
    np.testing.assert_allclose(expected + d["truth"]["label_noise"].sum(axis=1), d["targets"][:, 0], rtol=0, atol=1e-12)
    warmer = clean.copy()
    warmer[:, :, 0] += 2
    assert np.all(exposure_proxy(warmer, config) > expected)
    changed = deepcopy(config)
    changed["means"][0][0] += 2
    changed["means"][1][0] += 2
    changed["means"][2][0] += 2
    warm_data = generate_split(changed, 42, "train")
    assert np.all(warm_data["targets"][:, 0] > d["targets"][:, 0])
    np.testing.assert_array_equal(warm_data["targets"][:, 1], d["targets"][:, 1])


def test_visible_prefix_and_normal_measurement_noise(config):
    d = generate_split(config, 42, "train")
    x = d["inputs"]
    assert set(x) == INPUT_KEYS
    assert x["observations"].shape == (16, 118, 3)
    assert x["available_time"].max() == 119
    difference = x["observations"] - d["truth"]["clean_process"][:, :118]
    assert np.count_nonzero(difference) == difference.size
    # Deterministic, broad numerical tolerance checks the intended scale, not a p-value.
    np.testing.assert_allclose(difference.std(axis=(0, 1)), config["measurement_noise_std"], rtol=.15)
    assert d["events"] == []


@pytest.mark.parametrize("kind", ["bias", "drift", "missing"])
def test_paired_changes_and_exact_amplitude(config, kind):
    reference = generate_split(config, 42, "reference_fit")["inputs"]
    scales = fit_fault_scales(reference)
    d = generate_split(config, 42, "val")
    original = d["inputs"]["observations"].copy()
    view, mask = apply_view(d["inputs"], d["events"], scales, kind, 2)
    changed = np.isnan(view["observations"]) | (view["observations"] != original)
    np.testing.assert_array_equal(changed, mask)
    np.testing.assert_array_equal(d["inputs"]["observations"], original)
    for i, event in enumerate(d["events"]):
        start, end, ch = event["start"], event["end_exclusive"], event["channel"]
        assert mask[i].sum() == end - start
        if kind != "missing":
            amplitude = 2 * scales[d["inputs"]["recorded_regime"][i, start], ch] * event["sign"]
            delta = view["observations"][i, start:end, ch] - original[i, start:end, ch]
            expected = amplitude * np.arange(1, end - start + 1) / (end - start) if kind == "drift" else amplitude
            np.testing.assert_allclose(delta, expected, rtol=1e-10, atol=1e-12)


def test_delay_is_causal_and_single_factor(config):
    d = generate_split(config, 42, "val")
    scales = np.ones((3, 3))
    delayed, mask = apply_view(d["inputs"], [], scales, record_delay=12)
    np.testing.assert_array_equal(delayed["observations"], d["inputs"]["observations"])
    np.testing.assert_array_equal(delayed["recorded_regime"][:, 12:], d["inputs"]["recorded_regime"][:, :-12])
    assert not mask.any()
    with pytest.raises(ValueError, match="combine"):
        apply_view(d["inputs"], d["events"], scales, "bias", record_delay=6)


def test_scale_fit_matches_visible_reference(config):
    reference = generate_split(config, 42, "reference_fit")
    inputs = reference["inputs"]
    scales = fit_fault_scales(inputs)
    for r in range(3):
        np.testing.assert_array_equal(scales[r], inputs["observations"][inputs["recorded_regime"] == r].std(axis=0, ddof=1))
    reference["truth"]["clean_process"][:, 118:] = 1e9
    np.testing.assert_array_equal(scales, fit_fault_scales(inputs))


def test_round_trip_audit_and_load_guards(config, tmp_path):
    root = tmp_path / "data"
    report = write_dataset(config, root)
    assert report["status"] == "pass"
    assert report["independent_batches"] == 80
    assert report["view_checks"] == 20
    assert report["test_predictive_evaluation"] is False
    assert audit_dataset(root) == report
    with pytest.raises(FileExistsError):
        write_dataset(config, root)
    with pytest.raises(ValueError, match="test requires"):
        load_inputs(root, 42, "test")
    with pytest.raises(ValueError, match="remain original"):
        load_inputs(root, 42, "train", "bias")
    with pytest.raises(ValueError, match="frozen config"):
        load_inputs(root, 42, "val", "bias", magnitude=10)
    view = load_inputs(root, 42, "val", "missing")
    assert set(view) == INPUT_KEYS
    assert np.isnan(view["observations"]).any()
    # Hidden truth cannot affect the prediction loader, even if corrupted on disk.
    truth_path = root / "evaluation_only" / "seed_42" / "val" / "truth.npz"
    truth_path.write_bytes(b"intentionally broken audit-only file")
    replay = load_inputs(root, 42, "val", "missing")
    np.testing.assert_array_equal(view["observations"], replay["observations"])
    with pytest.raises(ValueError, match="hash mismatch"):
        audit_dataset(root)


def test_disk_generation_is_reproducible(config, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    write_dataset(config, first)
    write_dataset(config, second)
    # NPZ/CSV byte hashes, not just sampled values, must reproduce in this environment.
    a = json.loads((first / "manifest.json").read_text())
    b = json.loads((second / "manifest.json").read_text())
    assert a == b


@pytest.mark.parametrize("change", [
    {"data_source": "real"}, {"observation_delay": -1}, {"sample_time": 119},
    {"report_available_time": 180}, {"seeds": [42, 42]}, {"ar_rho": 1.0},
    {"measurement_noise_std": [0, .2, .005]}, {"record_delays": [0, 118]},
])
def test_reject_invalid_protocol(config, change):
    config.update(change)
    with pytest.raises(ValueError):
        validate_config(config)


def test_event_ids_cannot_silently_misalign(config):
    d = generate_split(config, 42, "val")
    d["events"][0]["sequence_id"] = "unknown"
    with pytest.raises(ValueError, match="matched event"):
        apply_view(d["inputs"], d["events"], np.ones((3, 3)), "bias")
