"""Export the preselected seed-42 model and a hand-constructed interface example."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import yaml
from .run_finite_memory_e1e import audited_source
from .run_logistics_e1b import ROOT, _json, _sha
from .scoring_component import make_bundle, input_contract


def prepare_demo(output: str | Path) -> Path:
    """Create reusable example assets without opening any retained-test array."""
    config = yaml.safe_load((ROOT/"configs/scoring_release.yaml").read_text())
    source = ROOT/config["model_run"]
    audited_source(source)
    training_config = yaml.safe_load((ROOT/config["training_run"]/"config.yaml").read_text())
    model_path = source/f"seed_{config['demo_seed']}"/"models.json"
    saved = json.loads(model_path.read_text())
    bundle = make_bundle(saved["models"][config["default_model"]], training_config["features"],
        training_config["units"], {"seed": config["demo_seed"], "source": str(model_path.relative_to(ROOT)),
        "source_sha256": _sha(model_path), "model_key": config["default_model"],
        "selection": "engineering_default_frozen_before_test"})
    rows = []
    for i in range(22):
        rows.append({"sequence_id": "demo_a", "time_step": i, "recorded_regime": "warehouse",
                     "temperature": 295. if i < 4 or i >= 15 else 303.,
                     "relative_humidity": 50., "vibration": .4})
    rows[16]["temperature"] = float("nan")
    rows[18]["recorded_regime"] = "unknown"
    for i in range(4):
        rows.append({"sequence_id": "demo_b", "time_step": i, "recorded_regime": "transit_smooth",
                     "temperature": 301., "relative_humidity": 52., "vibration": 1.5})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _json(output/"model.json", bundle)
    _json(output/"contract.json", input_contract(bundle))
    pd.DataFrame(rows).to_csv(output/"observations.csv", index=False)
    _json(output/"provenance.json", {"data_source": "synthetic_hand_constructed_interface_example",
        "not_an_experiment": True, "contains": ["bias", "missing_sensor", "unknown_regime", "sequence_reset"],
        "model_source_sha256": _sha(model_path)})
    return output


def main() -> None:
    """Export assets into a new directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(prepare_demo(parser.parse_args().output))


if __name__ == "__main__":
    main()
