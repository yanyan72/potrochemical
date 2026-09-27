"""Release boundary tests: no production holdout performance is inspected here."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.scoring_component import make_bundle, input_contract, score_frame, validate_bundle, read_observations
from src.score_batch import run_batch
from src.temporal_reliability import score_temporal
from src.run_final_evaluation import retained_innovations, validate_release


def bundle(kind: str = "marginal") -> dict:
    """Small identity reference for hand-verifiable boundary checks."""
    block = {"center": [0., 0.], "covariance": [[1., 0.], [0., 1.]], "precision": [[1., 0.], [0., 1.]]}
    ref = {"method": f"regime_{kind}", "scaler_center": [0., 0.], "scaler_scale": [1., 1.],
           "references": {"a": block, "b": deepcopy(block)}}
    model = {"reference": ref, "alpha": .2, "quantile": .99, "initial_state": "zero",
             "reset_on_record_change": True, "thresholds": {"a": [1., 1.], "b": [1., 1.]}}
    return make_bundle(model, ["x", "y"], ["unit_x", "unit_y"], {"fixture": True})


def frame() -> pd.DataFrame:
    """Six observations including missing and unknown records."""
    return pd.DataFrame({"sequence_id": ["s"]*6, "time_step": range(6),
                         "recorded_regime": ["a", "a", "a", "unknown", "b", "b"],
                         "x": [2., 2., np.nan, 2., 2., 2.], "y": [2.]*6})


@pytest.mark.parametrize("kind", ["marginal", "conditional"])
def test_direct_api_parity_missing_unknown_and_nullable_alarm(kind: str) -> None:
    b, f = bundle(kind), frame()
    actual = score_frame(f, b, input_contract(b))
    expected = score_temporal(f[["x", "y"]].to_numpy()[None], f.recorded_regime.to_numpy()[None], b["model"])
    np.testing.assert_array_equal(actual.consistency_score.to_numpy(), expected["reliability"].ravel())
    assert actual.loc[4, "status"] == "missing_sensor"
    assert actual.loc[5, "status"] == ("ok" if kind == "marginal" else "missing_dependency")
    assert actual.loc[6, "status"] == "unknown_regime" and pd.isna(actual.loc[6, "alarm"])
    assert actual.loc[8, "deviation_ratio"] == .4  # independent reset after unknown


def test_prefix_causal_sequence_isolation_and_order_preserved() -> None:
    b, f = bundle(), frame()
    complete = score_frame(f, b, input_contract(b))
    prefix = score_frame(f.iloc[:2], b, input_contract(b))
    pd.testing.assert_frame_equal(complete.iloc[:4], prefix)
    second = f.copy(); second["sequence_id"] = "second"
    both = score_frame(pd.concat([f, second], ignore_index=True), b, input_contract(b))
    np.testing.assert_array_equal(both.consistency_score[:12], both.consistency_score[12:])
    interleaved = pd.concat([f.iloc[[i]] if j == 0 else second.iloc[[i]] for i in range(6) for j in range(2)])
    result = score_frame(interleaved, b, input_contract(b))
    assert result.sequence_id.tolist()[:4] == ["s", "s", "second", "second"]


@pytest.mark.parametrize("case", ["gap", "duplicate", "reverse", "infinite", "nonnumeric", "label", "emptyid", "boolean"])
def test_bad_input_rejected(case: str) -> None:
    b, f = bundle(), frame()
    if case == "gap": f.loc[5, "time_step"] = 7
    if case == "duplicate": f.loc[5, "time_step"] = 4
    if case == "reverse": f = f.iloc[::-1]
    if case == "infinite": f.loc[0, "x"] = np.inf
    if case == "nonnumeric": f["x"] = f.x.astype(object); f.loc[0, "x"] = "sensor_error"
    if case == "label": f["hidden_truth"] = 1
    if case == "emptyid": f.loc[0, "sequence_id"] = ""
    if case == "boolean": f["x"] = True
    with pytest.raises(ValueError): score_frame(f, b, input_contract(b))


@pytest.mark.parametrize("case", ["units", "zero_scale", "nan", "indefinite", "inverse", "threshold", "regime", "reset"])
def test_contract_or_model_rejected(case: str) -> None:
    b, contract = bundle(), input_contract(bundle())
    m = b["model"]
    if case == "units": contract["units"][0] = "wrong"
    if case == "zero_scale": m["reference"]["scaler_scale"][0] = 0
    if case == "nan": m["reference"]["scaler_center"][0] = float("nan")
    if case == "indefinite": m["reference"]["references"]["a"]["covariance"][0][0] = -1
    if case == "inverse": m["reference"]["references"]["a"]["precision"][0][0] = 2
    if case == "threshold": m["thresholds"]["a"][0] = 0
    if case == "regime": del m["thresholds"]["b"]
    if case == "reset": m["reset_on_record_change"] = False
    with pytest.raises(ValueError): score_frame(frame(), b, contract)


def test_csv_roundtrip_ids_output_no_overwrite_and_duplicates(tmp_path: Path) -> None:
    b, f = bundle(), frame()
    f["sequence_id"] = "NA"  # literal identifier, not a missing-value sentinel
    model, data, contract = (tmp_path/n for n in ("model.json", "data.csv", "contract.json"))
    model.write_text(json.dumps(b)); contract.write_text(json.dumps(input_contract(b)))
    f.to_csv(data, index=False)
    out = run_batch(model, data, contract, tmp_path/"run")
    assert json.loads((out/"metadata.json").read_text())["output_rows"] == 12
    actual = pd.read_csv(out/"scores.csv", keep_default_na=False)
    assert set(actual.sequence_id) == {"NA"} and actual.loc[6, "alarm"] == ""
    with pytest.raises(FileExistsError): run_batch(model, data, contract, out)
    data.write_text("sequence_id,time_step,recorded_regime,x,x\ns,0,a,1,1\n")
    with pytest.raises(ValueError): read_observations(data, b["features"])


def test_retained_membership_on_artificial_arrays_and_fixed_config() -> None:
    membership = {k: [k+"_0", k+"_1"] for k in ("train_fit", "train_cal", "val", "test")}
    ids = np.array([i for values in membership.values() for i in values])
    base = {"sequence_ids": ids, "splits": np.repeat(list(membership), 2),
            "regimes": np.array(["a"]*8), "clean": np.ones((8, 3, 2))*3}
    config = {"regimes": {"a": {"mean": [1., 1.], "std": [2., 2.]}},
              "sequences_per_regime": {"test": 2}, "sequence_length": 3, "features": ["x", "y"]}
    z, selected, _ = retained_innovations(base, config, membership)
    np.testing.assert_array_equal(z, np.ones((2, 3, 2)))
    assert selected.tolist() == membership["test"]
    bad = deepcopy(membership); bad["test"] = bad["val"]
    with pytest.raises(ValueError): retained_innovations(base, config, bad)
    with pytest.raises(ValueError): validate_release({"seeds": [42]})
