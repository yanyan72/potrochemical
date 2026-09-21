"""Scientific checks for independently controlled events and CUSUM."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.controlled_events import (validate_controlled, validation_innovations,
                                   candidate_starts, controlled_scenario)
from src.cusum_reliability import causal_cusum, calibrate_cusum, score_cusum
from src.run_controlled_e1f import run_controlled

ROOT = Path(__file__).parents[1]


def configs() -> tuple[dict, dict, dict]:
    """Load protocol plus read-only training scales."""
    config = yaml.safe_load((ROOT/"configs/logistics_e1f.yaml").read_text())
    source = yaml.safe_load((ROOT/config["source_run"]/"config.yaml").read_text())
    scales = json.loads((ROOT/config["source_run"]/"seed_42/scales.json").read_text())
    return config, source, scales


def reference(kind: str = "marginal") -> dict:
    """Identity model for explicit recurrence checks."""
    state = {"center": [0., 0.], "covariance": [[1., 0.], [0., 1.]],
             "precision": [[1., 0.], [0., 1.]], "threshold": [2., 2.]}
    return {"method": f"regime_{kind}", "scaler_center": [0., 0.], "scaler_scale": [1., 1.],
            "references": {"a": state, "b": deepcopy(state)}}


def test_cusum_hand_calculation_sign_and_prefix() -> None:
    x = np.array([[[2.], [2.], [-2.], [-2.], [0.]]])
    records = np.full((1, 5), "a")
    h = causal_cusum(x, records, .5)
    np.testing.assert_array_equal(h.ravel(), [1.5, 3., 1.5, 3., 2.5])
    np.testing.assert_array_equal(causal_cusum(-x, records, .5), h)
    future = x.copy(); future[:, 3:] = 100
    np.testing.assert_array_equal(causal_cusum(future, records, .5)[:, :3], h[:, :3])
    repeat = causal_cusum(np.repeat(x, 2, axis=0), np.repeat(records, 2, axis=0), .5)
    np.testing.assert_array_equal(repeat[0], repeat[1])
    with pytest.raises(ValueError):
        causal_cusum(x, records, 0.)


def test_cusum_record_and_gap_reset_without_alarm_reset() -> None:
    x = np.full((1, 6, 2), 2.)
    x[0, 2, 0] = np.nan
    records = np.array([["a", "a", "a", "a", "b", "b"]])
    h = causal_cusum(x, records, .5)
    np.testing.assert_allclose(h[0, :, 0], [1.5, 3., np.nan, 1.5, 1.5, 3.], equal_nan=True)
    np.testing.assert_array_equal(h[0, :, 1], [1.5, 3., 4.5, 6., 1.5, 3.])


@pytest.mark.parametrize("kind", ["marginal", "conditional"])
def test_cusum_calibration_and_unavailable_inputs(kind: str) -> None:
    rng = np.random.default_rng(91)
    x = rng.normal(size=(4, 60, 2))
    states = np.repeat(np.array(["a", "a", "b", "b"])[:, None], 60, axis=1)
    original = x.copy()
    ref = reference(kind)
    model = calibrate_cusum(x, states, ref, k=.5, quantile=.99)
    np.testing.assert_array_equal(x, original)
    assert model["reference"] == ref and not model["reset_on_alarm"]
    # Calibration does not modify the frozen reference or require validation labels.
    other = calibrate_cusum(2*x, states, ref, k=.5, quantile=.99)
    assert other["reference"] == model["reference"] and other["thresholds"] != model["thresholds"]
    observed, visible = x[:1].copy(), states[:1].copy()
    observed[0, 2, 0] = np.nan
    visible[0, 4] = "?"
    scored = score_cusum(observed, visible, model)
    assert not scored["available"][0, 2, 0]
    assert bool(scored["available"][0, 2, 1]) == (kind == "marginal")
    assert not scored["available"][0, 4].any()
    assert scored["available"][0, 5].all()
    assert not scored["alarm"][~scored["available"]].any()


def test_candidate_starts_bounds_gaps_and_non_grid_phases() -> None:
    phases = set()
    for seed in range(30):
        starts = candidate_starts(180, max_duration=24, max_events=3, gap=12,
                                  post_steps=12, rng=np.random.default_rng(seed))
        assert min(starts) >= 0 and max(starts)+24+12 <= 180
        assert np.all(np.diff(np.sort(starts)) >= 36)
        phases.update((starts % 12).tolist())
    assert phases == set(range(12))


def test_paired_exact_duration_nested_count_and_amplitude() -> None:
    config, source, scales = configs()
    validate_controlled(config, source)
    x, ids, initial = validation_innovations(source, 42, config["validation_namespace"])
    scenarios = {s["name"]: s for s in config["scenarios"]}
    def make(name: str):
        return controlled_scenario(x, ids, initial, source, config, scenarios[name], scales, 42)
    base = make("steady_d12_n2_m1")
    for name in ("duration4", "duration24", "count1", "count3", "magnitude2"):
        data = make(name)
        np.testing.assert_array_equal(data.clean, base.clean)
        s = scenarios[name]
        assert len(data.events) == len(ids)*s["count"]
        assert data.mask.sum() == len(ids)*s["count"]*s["duration"]
        np.testing.assert_array_equal(data.mask, data.observed != data.clean)
        for event in data.events:
            assert event["end_exclusive"]-event["start"] == s["duration"]
    def identities(data):
        return {(e["sequence_index"], e["slot"], e["start"], tuple(e["channels"]), tuple(e["signs"])) for e in data.events}
    assert identities(make("count1")) < identities(base) < identities(make("count3"))
    assert identities(make("duration4")) == identities(base) == identities(make("duration24"))
    np.testing.assert_allclose(make("magnitude2").observed-base.clean, 2*(base.observed-base.clean), atol=1e-12)


def test_switch_delay_and_targeted_offsets_are_separate() -> None:
    config, source, scales = configs()
    x, ids, initial = validation_innovations(source, 42, config["validation_namespace"])
    scenario = {s["name"]: s for s in config["scenarios"]}
    def make(name: str):
        return controlled_scenario(x, ids, initial, source, config, scenario[name], scales, 42)
    base, delayed = make("switch_random"), make("switch_delay12")
    for attr in ("clean", "observed", "mask", "true_regimes"):
        np.testing.assert_array_equal(getattr(base, attr), getattr(delayed, attr))
    np.testing.assert_array_equal(delayed.recorded_regimes, base.true_regimes[:, np.maximum(np.arange(180)-12, 0)])
    for name, offset in (("switch_before6", -6), ("switch_at", 0), ("switch_after6", 6)):
        data = make(name)
        assert {e["offset_from_nearest_transition"] for e in data.events} == {offset}
        assert {e["start"] for e in data.events} == {60+offset, 120+offset}
        assert all(e["duration"] == 12 for e in data.events)


def test_new_validation_is_split_independent() -> None:
    config, source, _ = configs()
    a = validation_innovations(source, 42, config["validation_namespace"])
    changed = deepcopy(source)
    changed["sequences_per_regime"]["test"] = 100
    changed["sequences_per_regime"]["train_fit"] = 100
    b = validation_innovations(changed, 42, config["validation_namespace"])
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    other = validation_innovations(source, 42, config["validation_namespace"]+1)
    assert not np.array_equal(a[0], other[0])
    assert not set(a[1]) & set(other[1])
    assert all(str(i).startswith("valf_") for i in a[1])


def test_reject_mixed_factor_comparison() -> None:
    config, source, _ = configs()
    config["scenarios"][1]["count"] = 1
    with pytest.raises(ValueError, match="one factor"):
        validate_controlled(config, source)


def test_pipeline_repeatability_and_frozen_models(tmp_path: Path) -> None:
    config, _, _ = configs()
    config["seeds"] = [42]
    config["scenarios"] = config["scenarios"][:3]
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(config))
    a = run_controlled(path, output_root=tmp_path, run_id="a")
    b = run_controlled(path, output_root=tmp_path, run_id="b")
    for name in ("metrics_by_seed.csv", "events.csv", "event_design.csv", "seed_42/models.json"):
        assert (a/name).read_bytes() == (b/name).read_bytes()
    meta = json.loads((a/"metadata.json").read_text())
    assert not meta["test_evaluated"] and not meta["quality_evaluated"]
    for name, value in meta["output_sha256"].items():
        assert hashlib.sha256((a/name).read_bytes()).hexdigest() == value
    saved = json.loads((a/"seed_42/models.json").read_text())
    old = json.loads((ROOT/config["baseline_run"]/"seed_42/models.json").read_text())
    for method in config["methods"]:
        for temporal, suffix in (("point", "a1"), ("ewma", "a0.2")):
            assert saved["models"][f"{method}__{temporal}"] == old["models"][f"{method}__{suffix}"]
    metrics = pd.read_csv(a/"metrics_by_seed.csv")
    assert (metrics.groupby(["scenario", "view", "regime", "window"]).n_cells.nunique() == 1).all()
    assert (metrics.native_score_coverage == 1.).all()
    with pytest.raises(FileExistsError):
        run_controlled(path, output_root=tmp_path, run_id="a")
