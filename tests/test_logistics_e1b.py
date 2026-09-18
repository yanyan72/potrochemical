"""Scientific invariants for storage/transport references and their evaluation."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.logistics_simulator import generate_logistics, validate_logistics_config
from src.run_logistics_e1b import detection_metrics, run_experiment
from src.sensor_reliability import channel_deviation, fit_sensor_reference, score_sensors


@pytest.fixture
def logistics_config() -> dict:
    """Compact deterministic version of the frozen benchmark protocol."""
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/logistics_e1b.yaml").read_text())
    config["seeds"] = [42]
    config["sequence_length"] = 40
    config["sequences_per_regime"] = dict.fromkeys(config["sequences_per_regime"], 2)
    config["corruption"]["max_segment_length"] = 8
    return config


def test_simulation_reproducibility_and_split_independence(logistics_config: dict) -> None:
    data = generate_logistics(logistics_config, 42)
    again = generate_logistics(logistics_config, 42)
    np.testing.assert_array_equal(data.observed, again.observed)
    np.testing.assert_array_equal(data.clean != data.observed, data.mask)
    assert len(set(data.sequence_ids)) == len(data.sequence_ids)
    for split in set(data.splits):
        assert set(data.regimes[data.splits == split]) == set(logistics_config["regimes"])
    changed = deepcopy(logistics_config)
    changed["sequences_per_regime"]["test"] += 1
    other = generate_logistics(changed, 42)
    for split in ("train_fit", "train_cal", "val"):
        np.testing.assert_array_equal(data.observed[data.splits == split], other.observed[other.splits == split])
    assert all(e["replacement_source"] == "clean_train_empirical"
               for e in data.events if e["corruption_type"] == "random_replacement")


def test_conditional_formula_and_corruption_spread() -> None:
    covariance = np.array([[1.0, 0.8], [0.8, 1.0]])
    precision = np.linalg.inv(covariance)
    ref = {"center": [0., 0.], "covariance": covariance, "precision": precision}
    x = np.array([[2., 0.], [0., 0.]])
    result = channel_deviation(x, ref, "conditional")
    expected = np.abs(np.column_stack((x[:, 0] - .8 * x[:, 1], x[:, 1] - .8 * x[:, 0]))) / .6
    np.testing.assert_allclose(result, expected)
    assert result[0, 1] > 0  # A clean channel is implicated by the corrupted predictor.
    np.testing.assert_array_equal(channel_deviation(x, ref, "marginal"), np.abs(x))


def _fit_example(method: str) -> dict:
    rng = np.random.default_rng(42)
    x, cal = rng.normal(size=(200, 3)), rng.normal(size=(100, 3))
    return fit_sensor_reference(x, np.repeat("known", len(x)), cal, np.repeat("known", len(cal)),
                                method=method, single_regime="known", quantile=.99)


@pytest.mark.parametrize("kind", ["marginal", "conditional"])
def test_missing_unknown_and_monotone_scores(kind: str) -> None:
    model = _fit_example(f"regime_{kind}")
    center = np.asarray(model["scaler_center"]) + np.asarray(model["scaler_scale"]) * model["references"]["known"]["center"]
    values = np.vstack([center, center + np.array([10., 0., 0.]), center, center])
    values[2, 0] = np.nan
    scored = score_sensors(values, np.array(["known", "known", "known", "unseen"]), model)
    assert scored["reliability"][0, 0] > scored["reliability"][1, 0]
    assert not scored["available"][3].any()
    assert not scored["alarm"][3].any()
    assert np.isnan(scored["reliability"][3]).all()
    assert not scored["available"][2, 0]
    assert bool(scored["available"][2, 1]) == (kind == "marginal")


def test_calibration_not_fit_quantiles() -> None:
    rng = np.random.default_rng(2)
    x, cal = rng.normal(size=(200, 3)), rng.normal(size=(100, 3))
    kw = dict(method="pooled_conditional", single_regime="known", quantile=.99)
    a = fit_sensor_reference(x, np.repeat("known", 200), cal, np.repeat("known", 100), **kw)
    b = fit_sensor_reference(x, np.repeat("known", 200), cal * 10, np.repeat("known", 100), **kw)
    assert a["references"]["all"]["precision"] == b["references"]["all"]["precision"]
    assert a["references"]["all"]["threshold"] != b["references"]["all"]["threshold"]


def test_missing_excluded_and_bystander_metric() -> None:
    actual = np.array([[True, False], [False, False], [True, False]])
    ratio = np.array([[3., 2.], [.2, .3], [np.nan, np.nan]])
    scored = {"available": np.isfinite(ratio), "alarm": ratio > 1, "deviation_ratio": ratio}
    metric = detection_metrics(actual, scored, np.array([True, True, False]))
    assert metric["positive_cells"] == 1
    assert metric["tp"] == 1 and metric["fp"] == 1
    assert metric["bystander_fpr"] == 1
    assert metric["top1_localization"] == 1


def test_pipeline_and_replay(logistics_config: dict, tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(logistics_config, sort_keys=False))
    out = run_experiment(config, output_root=tmp_path, run_id="first")
    second = run_experiment(config, output_root=tmp_path, run_id="second")
    assert (out / "metrics_by_seed.csv").read_bytes() == (second / "metrics_by_seed.csv").read_bytes()
    assert (out / "seed_42/references.json").read_bytes() == (second / "seed_42/references.json").read_bytes()
    metadata = json.loads((out / "metadata.json").read_text())
    assert not metadata["test_evaluated"] and metadata["evaluation_split"] == "val"
    membership = json.loads((out / "seed_42/reference_sequences.json").read_text())
    assert not set(membership["fit"]) & set(membership["calibration"])
    assert not set(membership["fit"]) & set(membership["validation"])
    metrics = pd.read_csv(out / "metrics_by_seed.csv")
    assert set(metrics.method) == set(logistics_config["methods"])
    # Missingness changes the support by regime/scenario, never by method.
    assert (metrics.groupby(["scenario", "regime"]).n_cells.nunique() == 1).all()
    with pytest.raises(FileExistsError):
        run_experiment(config, output_root=tmp_path, run_id="first")


def test_invalid_config(logistics_config: dict) -> None:
    logistics_config["factor_loadings"][0] = float("nan")
    with pytest.raises(ValueError):
        validate_logistics_config(logistics_config)
