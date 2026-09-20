"""Numerical and support-denominator checks for finite-memory scoring."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.finite_memory import causal_sma, calibrate_sma, score_sma, coverage_metrics, coverage_events
from src.temporal_reliability import calibrate_temporal, score_temporal
from src.run_finite_memory_e1e import run_finite_memory, validate_config

ROOT = Path(__file__).parents[1]


def reference(kind: str = "marginal") -> dict:
    """Identity reference so raw values equal signed residuals."""
    state = {"center": [0., 0.], "covariance": [[1., 0.], [0., 1.]],
             "precision": [[1., 0.], [0., 1.]], "threshold": [2., 2.]}
    return {"method": f"regime_{kind}", "scaler_center": [0., 0.], "scaler_scale": [1., 1.],
            "references": {"a": state, "b": deepcopy(state)}}


def test_window_formula_cancellation_and_causal_prefix() -> None:
    x = np.array([[[1.], [2.], [-3.], [7.], [8.], [9.]]])
    records = np.full((1, 6), "a")
    actual = causal_sma(x, records, 3)
    np.testing.assert_allclose(actual.ravel(), [np.nan, np.nan, 0., 2., 4., 8.], equal_nan=True)
    changed = x.copy()
    changed[:, 4:] = 10000
    np.testing.assert_allclose(actual[:, :4], causal_sma(changed, records, 3)[:, :4], equal_nan=True)
    np.testing.assert_array_equal(causal_sma(x, records, 1), x)
    with pytest.raises(ValueError):
        causal_sma(x, records, 2.5)


def test_record_gap_and_sequence_reset() -> None:
    x = np.ones((2, 10, 2))
    x[0, 3, 0] = np.nan
    records = np.full((2, 10), "a")
    records[:, 6:] = "b"
    h = causal_sma(x, records, 3)
    assert np.isnan(h[:, :2]).all()
    assert np.isnan(h[0, 3:6, 0]).tolist() == [True, True, True]
    assert np.isfinite(h[0, 3:6, 1]).all()
    assert np.isfinite(h[1, 2:6]).all()  # Other sequence unaffected.
    assert np.isnan(h[:, 6:8]).all()
    np.testing.assert_array_equal(h[:, 8:], 1.)


def test_fault_history_exits_exact_finite_window() -> None:
    clean = np.zeros((1, 20, 1))
    observed = clean.copy()
    observed[:, 5:8] = 10
    records = np.full((1, 20), "a")
    a, b = (causal_sma(v, records, 4) for v in (clean, observed))
    assert b[0, 10, 0] > 0
    np.testing.assert_array_equal(a[:, 11:], b[:, 11:])  # Last affected sample index7 exits at11.


@pytest.mark.parametrize("kind", ["marginal", "conditional"])
def test_calibration_and_missing_unknown_semantics(kind: str) -> None:
    rng = np.random.default_rng(87)
    x = rng.normal(size=(4, 30, 2))
    records = np.repeat(np.array(["a", "a", "b", "b"])[:, None], 30, axis=1)
    ref = reference(kind)
    model = calibrate_sma(x, records, ref, window_size=3, quantile=.99)
    bigger = calibrate_sma(x*10, records, ref, window_size=3, quantile=.99)
    for state in ("a", "b"):
        assert model["calibration_counts"][state] == [56, 56]
        np.testing.assert_allclose(bigger["thresholds"][state], 10*np.array(model["thresholds"][state]))
    assert model["reference"] == ref == bigger["reference"]
    point = calibrate_sma(x, records, ref, window_size=1, quantile=.99)
    old_point = calibrate_temporal(x, records, ref, alpha=1, quantile=.99)
    for k, v in score_sma(x, records, point).items():
        np.testing.assert_allclose(v, score_temporal(x, records, old_point)[k], equal_nan=True)
    observed = x[:1].copy()
    visible = records[:1].copy()
    observed[0, 4, 0] = np.nan
    visible[0, 10] = "?"
    result = score_sma(observed, visible, model)
    assert not result["available"][0, 4:7, 0].any()
    assert bool(result["available"][0, 4, 1]) == (kind == "marginal")
    assert not result["available"][0, 10:13].any()
    assert result["available"][0, 13].all()


def test_common_support_does_not_hide_operational_misses() -> None:
    truth = np.zeros((1, 5, 1), bool)
    truth[0, [0, 1, 3], 0] = True
    available = np.array([[[False], [False], [True], [True], [True]]])
    alarm = np.array([[[False], [False], [False], [True], [False]]])
    score = {"available": available, "alarm": alarm,
             "deviation_ratio": np.where(available, alarm*2., np.nan),
             "reliability": np.where(available, 1., np.nan)}
    selected = np.ones((1, 5), bool)
    common = available.all(axis=-1)
    result = coverage_metrics(truth, score, selected, common, np.zeros_like(truth), support="common")
    assert result["recall"] == 1.
    assert result["all_timeline_recall"] == pytest.approx(1/3)
    assert result["unavailable_positive_fraction"] == pytest.approx(2/3)
    assert result["native_score_coverage"] == .6
    events = [{"sequence_index": 0, "start": 0, "end_exclusive": 2, "channels": [0]}]
    rows, _ = coverage_events(score, truth, events, post_steps=2)
    assert rows[0]["fully_unavailable"]
    assert rows[0]["fresh_delay_capped"] == 2
    assert not rows[0]["fresh_onset_hit"]


def test_finite_pipeline_provenance_support_and_repeatability(tmp_path: Path) -> None:
    config = yaml.safe_load((ROOT/"configs/logistics_e1e.yaml").read_text())
    config["seeds"] = [42]
    config["scenarios"] = ["steady_m1", "switch_d12"]
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(config))
    a = run_finite_memory(path, output_root=tmp_path, run_id="a")
    b = run_finite_memory(path, output_root=tmp_path, run_id="b")
    for name in ("metrics_by_seed.csv", "events.csv", "seed_42/models.json"):
        assert (a/name).read_bytes() == (b/name).read_bytes()
    meta = json.loads((a/"metadata.json").read_text())
    assert not meta["test_evaluated"] and not meta["quality_evaluated"]
    for name, value in meta["output_sha256"].items():
        assert hashlib.sha256((a/name).read_bytes()).hexdigest() == value
    metrics = pd.read_csv(a/"metrics_by_seed.csv")
    common = metrics[metrics.support == "common"]
    assert (common.groupby(["scenario", "view", "regime", "window"]).n_cells.nunique() == 1).all()
    steady = metrics[(metrics.scenario == "steady_m1") & (metrics.temporal == "sma")]
    np.testing.assert_allclose(steady.native_score_coverage, 172/180)
    audit = pd.read_csv(a/"calibration_audit.csv")
    assert (audit.n_calibration_cells == 688).all()
    assert (audit.calibration_fpr <= .01+1/audit.n_calibration_cells).all()
    # Own-support baseline metric replay must reproduce E1d; only temporal labels changed.
    old = pd.read_csv(ROOT/config["baseline_run"]/"metrics_by_seed.csv")
    for temporal, alpha in (("point", 1.), ("ewma", .2)):
        keys = ["scenario", "method", "view", "regime", "window"]
        new = metrics[(metrics.temporal == temporal) & (metrics.support == "own")].set_index(keys).sort_index()
        previous = old[(old.seed == 42) & (old.alpha == alpha) & old.scenario.isin(config["scenarios"])].set_index(keys).sort_index()
        for field in ("f1", "recall", "clean_channel_fpr", "post_event_fpr"):
            np.testing.assert_allclose(new[field], previous[field], rtol=1e-12, equal_nan=True)
    with pytest.raises(FileExistsError):
        run_finite_memory(path, output_root=tmp_path, run_id="a")


def test_reject_changed_budget_and_source_coverage() -> None:
    config = yaml.safe_load((ROOT/"configs/logistics_e1e.yaml").read_text())
    source = yaml.safe_load((ROOT/config["source_run"]/"config.yaml").read_text())
    baseline = yaml.safe_load((ROOT/config["baseline_run"]/"config.yaml").read_text())
    config["threshold_quantile"] = .95
    with pytest.raises(ValueError, match="budget"):
        validate_config(config, source, baseline)
