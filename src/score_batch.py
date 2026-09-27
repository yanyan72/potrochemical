"""Batch CLI for complete synthetic sequences with explicit schema and units."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from .scoring_component import load_bundle, read_observations, score_frame


def run_batch(model: str | Path, observations: str | Path, contract: str | Path,
              output: str | Path) -> Path:
    """Validate and score a CSV; refuse to replace any existing output directory."""
    model, observations, contract, output = map(Path, (model, observations, contract, output))
    bundle = load_bundle(model)
    frame = read_observations(observations, bundle["features"])
    scores = score_frame(frame, bundle, json.loads(contract.read_text(encoding="utf-8")))
    output.mkdir(parents=True, exist_ok=False)
    scores.to_csv(output/"scores.csv", index=False)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    meta = {"schema": bundle["schema"], "data_source": "synthetic", "input_rows": len(frame),
            "output_rows": len(scores), "available_cells": int(scores.available.sum()),
            "alarm_cells": int(scores.alarm.sum()), "status_counts": scores.status.value_counts().to_dict(),
            "model_sha256": digest(model), "observations_sha256": digest(observations),
            "contract_sha256": digest(contract), "scores_sha256": digest(output/"scores.csv"),
            "state_scope": "complete_sequence_in_this_call", "quality_evaluated": False}
    (output/"metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return output


def main() -> None:
    """Execute model loading, strict validation and batch scoring from paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "observations", "contract", "output"):
        parser.add_argument(f"--{key}", required=True)
    args = parser.parse_args()
    print(run_batch(args.model, args.observations, args.contract, args.output))


if __name__ == "__main__":
    main()
