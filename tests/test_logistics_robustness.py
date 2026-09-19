"""Independent checks for paired stress data, causal records and attribution."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.logistics_simulator import generate_logistics
from src.logistics_robustness import make_stress_scenario, training_regime_scales, validate_stress_config
from src.run_logistics_robustness import run_robustness, stress_metrics


@pytest.fixture
def stress_config() -> dict:
    """A compact protocol preserving every planned comparison."""
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/logistics_e1c.yaml").read_text())
    config["seeds"] = [42]
    config["sequence_length"] = 60
    config["sequences_per_regime"] = dict.fromkeys(config["sequences_per_regime"], 2)
    config["stress"].update(segment_length=6, transition_interval=20)
    return config


def test_mask_amplitude_events_and_multi_channel(stress_config: dict) -> None:
    base = generate_logistics(stress_config, 42)
    scales = training_regime_scales(base)
    for scenario in stress_config["stress"]["scenarios"]:
        data = make_stress_scenario(base, stress_config, scenario, 42)
        np.testing.assert_array_equal(data.clean != data.observed, data.mask)
        assert np.all(data.mask.any(axis=2).sum(axis=1) == round(scenario["ratio"] * 60))
        assert set(data.mask.sum(axis=2).ravel()) == {0, scenario["channels"]}
        expected = np.zeros_like(data.clean)
        for event in data.events:
            i = event["sequence_index"]
            for t in range(event["start"], event["end_exclusive"]):
                for channel, sign in zip(event["channels"], event["signs"]):
                    expected[i, t, channel] = sign * scenario["magnitude"] * scales[data.true_regimes[i, t]][channel]
        np.testing.assert_allclose(data.observed - data.clean, expected, atol=4e-14)


def test_paired_strength_channels_and_causal_delay(stress_config: dict) -> None:
    base = generate_logistics(stress_config, 42)
    scenarios = {s["name"]: make_stress_scenario(base, stress_config, s, 42)
                 for s in stress_config["stress"]["scenarios"]}
    reference = scenarios["steady_m2_r10_k1"]
    weak = scenarios["steady_m1"]
    np.testing.assert_array_equal(reference.mask, weak.mask)
    np.testing.assert_array_equal(reference.clean, weak.clean)
    np.testing.assert_allclose(reference.observed - reference.clean,
                               2 * (weak.observed - weak.clean), atol=6e-14)
    multi = scenarios["steady_k2"]
    assert np.all(~reference.mask | multi.mask)
    np.testing.assert_array_equal(reference.observed[reference.mask], multi.observed[reference.mask])
    immediate, delayed = scenarios["switch_d0"], scenarios["switch_d12"]
    for key in ("clean", "observed", "mask", "true_regimes"):
        np.testing.assert_array_equal(getattr(immediate, key), getattr(delayed, key))
    np.testing.assert_array_equal(delayed.recorded_regimes, immediate.true_regimes[:, np.maximum(np.arange(60) - 12, 0)])
    assert np.all(delayed.transition[:, 20:32]) and not delayed.transition[:, :20].any()
    assert np.all(delayed.recorded_regimes[:, 20:32] != delayed.true_regimes[:, 20:32])


def test_switching_preserves_standardized_innovations(stress_config: dict) -> None:
    base = generate_logistics(stress_config, 42)
    scenario = next(s for s in stress_config["stress"]["scenarios"] if s["name"] == "switch_d0")
    data = make_stress_scenario(base, stress_config, scenario, 42)
    profiles = stress_config["regimes"]
    for i, initial in enumerate(base.regimes[base.splits == "val"]):
        original = base.clean[base.splits == "val"][i]
        z = (original - profiles[initial]["mean"]) / profiles[initial]["std"]
        for state in profiles:
            selected = data.true_regimes[i] == state
            actual = (data.clean[i, selected] - profiles[state]["mean"]) / profiles[state]["std"]
            np.testing.assert_allclose(actual, z[selected], atol=2e-13)


def test_training_scales_and_test_independence(stress_config: dict) -> None:
    base = generate_logistics(stress_config, 42)
    scales = training_regime_scales(base)
    for regime, scale in scales.items():
        expected = base.clean[(base.splits == "train_fit") & (base.regimes == regime)].reshape(-1, 3).std(axis=0, ddof=1)
        np.testing.assert_array_equal(scale, expected)
    changed = deepcopy(stress_config)
    changed["sequences_per_regime"]["test"] += 2
    other = generate_logistics(changed, 42)
    s = stress_config["stress"]["scenarios"][0]
    a, b = (make_stress_scenario(d, stress_config, s, 42) for d in (base, other))
    np.testing.assert_array_equal(a.observed, b.observed)


def test_multilabel_metrics_hand_computed() -> None:
    truth = np.array([[True, True, False], [True, True, False], [False, False, False]])
    ratio = np.array([[3., 2., 0.], [4., 0., 3.], [.2, .1, .3]])
    result = {"available": np.ones_like(truth), "alarm": ratio > 1, "deviation_ratio": ratio}
    metrics = stress_metrics(truth, result)
    assert metrics["all_faults_row_recall"] == .5
    assert metrics["exact_fault_set_rate"] == .5
    assert metrics["top1_random_baseline"] == 2 / 3
    assert metrics["top1_localization"] == 1
    assert metrics["bystander_fpr"] == .5


def test_one_factor_validation(stress_config: dict) -> None:
    validate_stress_config(stress_config)
    stress_config["stress"]["scenarios"][1]["channels"] = 2
    with pytest.raises(ValueError, match="exactly one"):
        validate_stress_config(stress_config)


def test_replay_hashes_shared_support_and_pair_differences(stress_config: dict, tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(stress_config, sort_keys=False))
    a = run_robustness(config, output_root=tmp_path, run_id="a")
    b = run_robustness(config, output_root=tmp_path, run_id="b")
    for name in ("metrics_by_seed.csv", "paired_differences.csv", "seed_42/references.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    meta = json.loads((a / "metadata.json").read_text())
    assert meta["evaluation_split"] == "val" and meta["test_evaluated"] is False
    for name, sha in meta["output_sha256"].items():
        assert hashlib.sha256((a / name).read_bytes()).hexdigest() == sha
    membership = json.loads((a / "seed_42/membership.json").read_text())
    names = [name for subset in membership.values() for name in subset]
    assert len(names) == len(set(names))
    metrics = pd.read_csv(a / "metrics_by_seed.csv")
    assert (metrics.groupby(["scenario", "view", "regime", "window"]).n_cells.nunique() == 1).all()
    deltas = pd.read_csv(a / "paired_differences.csv")
    d = deltas[(deltas.scenario == "switch_d12") & (deltas.method == "pooled_marginal")]
    assert (d.f1.dropna() == 0).all()  # Pooled scores must ignore record delay.
    with pytest.raises(FileExistsError):
        run_robustness(config, output_root=tmp_path, run_id="a")
