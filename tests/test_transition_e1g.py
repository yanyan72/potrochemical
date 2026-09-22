"""Scientific checks for the single-factor record-reset ablation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.temporal_reliability import causal_ewma, calibrate_temporal, score_temporal, signed_residuals
from src.run_transition_e1g import run_transition, record_windows, validate_config, identical

ROOT = Path(__file__).parents[1]


def reference(kind: str = "marginal") -> dict:
    """Two identical identity references to isolate the effect of a record reset."""
    block = {"center": [0., 0.], "covariance": [[1., 0.], [0., 1.]], "precision": [[1., 0.], [0., 1.]]}
    return {"method": f"regime_{kind}", "scaler_center": [0., 0.], "scaler_scale": [1., 1.],
            "references": {"a": block, "b": deepcopy(block)}}


def test_reset_carry_hand_computation_and_causality() -> None:
    x = np.array([[[2.], [2.], [2.], [-2.], [2.]]])
    records = np.array([["a", "a", "b", "b", "b"]])
    reset = causal_ewma(x, records, .5)
    carry = causal_ewma(x, records, .5, reset_on_record_change=False)
    np.testing.assert_array_equal(reset.ravel(), [1., 1.5, 1., -.5, .75])
    np.testing.assert_array_equal(carry.ravel(), [1., 1.5, 1.75, -.125, .9375])
    changed = x.copy(); changed[:, 3:] = 999
    labels = records.copy(); labels[:, 3:] = "c"
    np.testing.assert_array_equal(carry[:, :3], causal_ewma(changed, labels, .5,
                                  reset_on_record_change=False)[:, :3])
    doubled = causal_ewma(np.repeat(x, 2, axis=0), np.repeat(records, 2, axis=0), .5,
                          reset_on_record_change=False)
    np.testing.assert_array_equal(doubled[0], doubled[1])
    np.testing.assert_array_equal(causal_ewma(x, records, 1, reset_on_record_change=False), x)
    with pytest.raises(ValueError):
        causal_ewma(x, records, .5, reset_on_record_change="false")


@pytest.mark.parametrize("kind", ["marginal", "conditional"])
def test_carry_gap_unknown_and_current_threshold(kind: str) -> None:
    model = {"reference": reference(kind), "alpha": .5, "reset_on_record_change": False,
             "thresholds": {"a": [1., 1.], "b": [2., 2.]}}
    x = np.full((1, 7, 2), 2.)
    x[0, 2, 0] = np.nan
    records = np.array([["a", "b", "b", "b", "unknown", "a", "a"]])
    scores = score_temporal(x, records, model)
    # Current b threshold is used despite carrying a history.
    np.testing.assert_array_equal(scores["deviation_ratio"][0, 1], [.75, .75])
    assert not scores["available"][0, 2, 0]
    assert scores["available"][0, 2, 1] == (kind == "marginal")
    assert not scores["available"][0, 4].any() and not scores["alarm"][0, 4].any()
    np.testing.assert_array_equal(scores["deviation_ratio"][0, 5], [1., 1.])
    # Gap state is cleared; first later value starts with alpha * residual.
    assert scores["deviation_ratio"][0, 3, 0] == .5
    assert np.isnan(signed_residuals(x, records, model["reference"])[0, 4]).all()


def test_stationary_calibration_identical_and_default_compatible() -> None:
    x = np.random.default_rng(4).normal(size=(4, 60, 2))
    records = np.repeat(np.array(["a", "a", "b", "b"])[:, None], 60, axis=1)
    a = calibrate_temporal(x, records, reference(), alpha=.2, quantile=.99)
    b = calibrate_temporal(x, records, reference(), alpha=.2, quantile=.99, reset_on_record_change=False)
    expected = deepcopy(a); expected["reset_on_record_change"] = False
    assert b == expected
    identical(score_temporal(x, records, a), score_temporal(x, records, b))
    missing_flag = deepcopy(a); del missing_flag["reset_on_record_change"]
    identical(score_temporal(x, records, a), score_temporal(x, records, missing_flag))


def test_record_window_boundaries_and_overlap() -> None:
    records = np.array([["a", "a", "b", "b", "c", "c", "c", "c"], ["a"]*8])
    np.testing.assert_array_equal(record_windows(records, 3)[0], [False, False, True, True, True, True, True, False])
    assert not record_windows(records, 3)[1].any()
    assert record_windows(records, 99)[0].sum() == 6
    with pytest.raises(ValueError):
        record_windows(records, 0)


def test_reject_retuning_and_bad_source_subset() -> None:
    config = yaml.safe_load((ROOT/"configs/logistics_e1g.yaml").read_text())
    source = yaml.safe_load((ROOT/config["source_run"]/"config.yaml").read_text())
    for field, value in [("alpha", .3), ("threshold_quantile", .95), ("scenarios", ["fake"]),
                         ("record_window_steps", 0), ("seeds", [42, 42])]:
        changed = deepcopy(config); changed[field] = value
        with pytest.raises(ValueError):
            validate_config(changed, source)


def test_pipeline_replay_invariance_hashes_and_reproducibility(tmp_path: Path) -> None:
    config = yaml.safe_load((ROOT/"configs/logistics_e1g.yaml").read_text())
    config["seeds"] = [42]
    config["scenarios"] = ["steady_d12_n2_m1", "switch_before6", "switch_delay12"]
    cfg = tmp_path/"config.yaml"; cfg.write_text(yaml.safe_dump(config))
    a = run_transition(cfg, output_root=tmp_path, run_id="a")
    b = run_transition(cfg, output_root=tmp_path, run_id="b")
    for file in ("metrics_by_seed.csv", "events.csv", "paired_by_seed.csv", "seed_42/models.json"):
        assert (a/file).read_bytes() == (b/file).read_bytes()
    meta = json.loads((a/"metadata.json").read_text())
    assert meta["ewma_replay_count"] == 12 and meta["steady_invariance_count"] == 4
    assert meta["calibration_reset_carry_exact"] and not meta["thresholds_refit"]
    assert not meta["test_evaluated"] and not meta["quality_evaluated"]
    for name, value in meta["output_sha256"].items():
        assert hashlib.sha256((a/name).read_bytes()).hexdigest() == value
    metrics = pd.read_csv(a/"metrics_by_seed.csv")
    assert (metrics.groupby(["scenario", "view", "regime", "window"]).n_cells.nunique() == 1).all()
    assert (metrics.native_score_coverage == 1).all()
    source = pd.read_csv(ROOT/config["source_run"]/"metrics_by_seed.csv")
    keys = ["seed", "scenario", "method", "temporal", "view", "regime", "window"]
    old = source[(source.seed == 42) & source.scenario.isin(config["scenarios"])].sort_values(keys).reset_index(drop=True)
    replay = metrics[(metrics.temporal != "carry") & metrics.window.isin(["all", "transition", "stable"])].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(replay[old.columns], old, check_exact=False, atol=1e-14, rtol=1e-14)
    with pytest.raises(FileExistsError):
        run_transition(cfg, output_root=tmp_path, run_id="a")
