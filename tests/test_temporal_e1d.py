"""Scientific regression checks for causal temporal scoring and event evaluation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.run_temporal_e1d import run_temporal, validate_config
from src.temporal_reliability import signed_residuals, causal_ewma, calibrate_temporal, score_temporal, event_evaluation

ROOT = Path(__file__).parents[1]


def reference(kind: str = "marginal") -> dict:
    """Small correlated reference with hand-computable conditional residuals."""
    covariance = np.array([[1., .8], [.8, 1.]])
    ref = {"center": [0., 0.], "covariance": covariance.tolist(),
           "precision": np.linalg.inv(covariance).tolist(), "threshold": [2., 2.]}
    return {"method": f"regime_{kind}", "scaler_center": [0., 0.], "scaler_scale": [1., 1.],
            "references": {"a": ref, "b": deepcopy(ref)}}


def test_signed_formula() -> None:
    x = np.array([[[2., 0.], [-2., 0.]]])
    states = np.array([["a", "a"]])
    result = signed_residuals(x, states, reference("conditional"))
    expected = np.array([[[2/.6, -1.6/.6], [-2/.6, 1.6/.6]]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(signed_residuals(x, states, reference()), x)


def test_ewma_causal_prefix_record_and_sequence_reset() -> None:
    r = np.array([[[2.], [2.], [-2.], [2.], [2.]]])
    records = np.array([["a", "a", "a", "b", "b"]])
    actual = causal_ewma(r, records, .5)
    np.testing.assert_array_equal(actual.ravel(), [1., 1.5, -.25, 1., 1.5])
    changed = r.copy(); changed[:, 3:] = 1000
    np.testing.assert_array_equal(actual[:, :3], causal_ewma(changed, records, .5)[:, :3])
    double = causal_ewma(np.repeat(r, 2, axis=0), np.repeat(records, 2, axis=0), .5)
    np.testing.assert_array_equal(double[0], double[1])
    np.testing.assert_array_equal(causal_ewma(r, records, 1.), r)


@pytest.mark.parametrize("kind", ["marginal", "conditional"])
def test_missing_unknown_and_recovery(kind: str) -> None:
    x = np.array([[[2., 2.], [np.nan, 2.], [2., 2.], [2., 2.], [2., 2.]]])
    records = np.array([["a", "a", "a", "unknown", "a"]])
    r = signed_residuals(x, records, reference(kind))
    h = causal_ewma(r, records, .2)
    assert np.isnan(h[0, 1, 0]) and np.isnan(h[0, 3]).all()
    assert np.isfinite(h[0, 1, 1]) == (kind == "marginal")
    np.testing.assert_allclose(h[0, 2, 0], .2*r[0, 2, 0])
    np.testing.assert_allclose(h[0, 4], .2*r[0, 4])


def test_calibration_independence_and_absolute_after_ewma() -> None:
    rng = np.random.default_rng(99)
    x = rng.normal(size=(4, 50, 2))
    states = np.repeat(np.array(["a", "a", "b", "b"])[:, None], 50, axis=1)
    ref = reference()
    a = calibrate_temporal(x, states, ref, alpha=.2, quantile=.99)
    b = calibrate_temporal(10*x, states, ref, alpha=.2, quantile=.99)
    assert a["reference"] == b["reference"] == ref
    for state in ref["references"]:
        np.testing.assert_allclose(b["thresholds"][state], 10*np.array(a["thresholds"][state]))
    # Opposite signed deviations cancel before absolute value.
    r = np.array([[[2., 2.], [-1.6, -1.6]]])
    h = causal_ewma(r, np.array([["a", "a"]]), .2)
    np.testing.assert_allclose(h[0, 1], 0., atol=1e-15)


def test_events_preactive_misses_late_alarms_and_tail() -> None:
    truth = np.zeros((1, 12, 2), bool)
    truth[0, 2:5, :] = True
    alarm = np.zeros_like(truth)
    alarm[0, 1:5, 0] = True  # Active before the event: no fresh onset.
    alarm[0, 6, 1] = True   # Late alarm after end: must not count as detection.
    events = [{"sequence_index": 0, "start": 2, "end_exclusive": 5, "channels": [0, 1]}]
    rows, tail = event_evaluation(alarm, truth, events, post_steps=3)
    assert rows[0]["any_active_hit"] and rows[0]["pre_active"]
    assert not rows[0]["fresh_onset_hit"] and rows[0]["fresh_delay_capped"] == 3
    assert not rows[1]["any_active_hit"] and np.isnan(rows[1]["fresh_delay_detected"])
    assert tail.sum() == 6 and not (tail & truth).any()
    alarm[0, 3, 1] = True
    rows, _ = event_evaluation(alarm, truth, events, post_steps=3)
    assert rows[1]["fresh_delay_detected"] == 1
    _, duplicate_tail = event_evaluation(alarm, truth, events*2, post_steps=3)
    np.testing.assert_array_equal(duplicate_tail, tail)


def test_pipeline_replay_and_source_immutability(tmp_path: Path) -> None:
    config = yaml.safe_load((ROOT / "configs/logistics_e1d.yaml").read_text())
    config["seeds"] = [42]
    config["scenarios"] = ["steady_m1", "switch_d12"]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    a = run_temporal(config_path, output_root=tmp_path, run_id="a")
    b = run_temporal(config_path, output_root=tmp_path, run_id="b")
    for name in ["metrics_by_seed.csv", "events.csv", "seed_42/models.json"]:
        assert (a/name).read_bytes() == (b/name).read_bytes()
    meta = json.loads((a/"metadata.json").read_text())
    assert meta["alpha_one_replay_verified"] and not meta["test_evaluated"]
    for name, value in meta["output_sha256"].items():
        assert hashlib.sha256((a/name).read_bytes()).hexdigest() == value
    audit = pd.read_csv(a/"calibration_audit.csv")
    # A linear empirical quantile has finite-sample count resolution (1/n).
    assert (audit.calibration_fpr <= .01 + 1/audit.n_calibration_cells + 1e-12).all()
    metrics = pd.read_csv(a/"metrics_by_seed.csv")
    assert (metrics.groupby(["scenario", "view", "regime", "window"]).n_cells.nunique() == 1).all()
    with pytest.raises(FileExistsError):
        run_temporal(config_path, output_root=tmp_path, run_id="a")


def test_reject_validation_recalibration() -> None:
    config = yaml.safe_load((ROOT/"configs/logistics_e1d.yaml").read_text())
    source = yaml.safe_load((ROOT/config["source_run"]/"config.yaml").read_text())
    config["threshold_quantile"] = .95
    with pytest.raises(ValueError, match="budget"):
        validate_config(config, source)
