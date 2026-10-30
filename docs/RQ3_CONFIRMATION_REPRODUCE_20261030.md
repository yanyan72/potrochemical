# Reproduce selective-correction confirmation

All research data in this release are synthetic. Original data and development outputs remain unchanged. Only fresh test streams use the new namespace2026103001. Do not regenerate the original10000-batch dataset.

## Stored artifacts

Root: `results/rq3_prediction/rq3_selective_confirmation_20261030_v1`.

`archives/training_seed_*.zip` contains each seed's13 complete model state arrays, checkpoint choices, all40-epoch histories, validation predictions, fixed reference/scaler parameters, shared view schedule, random score indices, source hashes and manifest. `archives/evaluation_seed_*.zip` contains visible test inputs, separate evaluation-only latent process/targets, fixed events, all13-model/eight-view predictions, scoring and activation diagnostics, and manifest. Archives preserve every source byte; adjacent JSON records ZIP SHA256 and file counts. Root CSV tables are also provided directly for inspection. Published compressed archives are persistent copies, not scripts requiring data regeneration.

Restore into the same root with:

```sh
python src/restore_selective_confirmation.py results/rq3_prediction/rq3_selective_confirmation_20261030_v1
```

This refuses to replace any nonidentical existing file. Python3.11, NumPy, SciPy, PyTorch and PyYAML are used for experiments; matplotlib is used only for figures. Package versions are recorded in identity.json. All models ran on CPU with one PyTorch thread and deterministic algorithms.

## Training

Use a new empty result directory when reproducing; keep the original release intact. Run seeds42–46 separately:

```sh
PYTHONPATH=src python src/rq3_selective_confirmation.py train \
  --data data/processed/rq3_paired_20261019_rebuild_v1 \
  --legacy . \
  --a0-root results/rq3_prediction/rq3_uncertainty_a0_20261029_v1 \
  --scorer results/rq3_prediction/shared_mode_fault_development_20261029_v2 \
  --base-root results/rq3_prediction/rq3_joint_prediction_dev_20261029_v1 \
  --selective-root results/rq3_prediction/rq3_selective_prediction_dev_20261029_v1 \
  --config configs/rq3_selective_confirmation_20261030.json \
  --output results/reproduction_confirmation --seed 42 --commit <implementation-commit>
```

Repeat with43–46, then run `lock --output results/reproduction_confirmation`. The release records its actual implementation and training-lock commits. Identical software versions should reproduce checkpoints/predictions; exact cross-platform floating-point identity is not assumed. The first optimization repetition reproduces all15 applicable prior development models exactly on the execution machine.

## Held-out evaluation

Only after training is locked:

```sh
PYTHONPATH=src python src/rq3_selective_confirmation.py evaluate \
  --data data/processed/rq3_paired_20261019_rebuild_v1 --legacy . \
  --output results/reproduction_confirmation --seed 42 --lock-commit <training-lock-commit>
```

Repeat seeds43–46. This reconstructs the already-used frozen confirmation streams for reproduction; it is not a new independent experiment. Each per-seed output refuses overwrite. Then run `summarize --output results/reproduction_confirmation`.

## Independent verification

```sh
python src/verify_selective_confirmation.py \
  results/rq3_prediction/rq3_selective_confirmation_20261030_v1 --repository-root .
```

The verifier checks archived inputs and original source hashes,65 model budgets and checkpoint-selection arithmetic, data/label generation identities, every MAE/RMSE and paired difference, optimizer-averaged outer-seed intervals, and leave-one-seed sensitivity. Source paths recorded on the execution host are remapped to repository locations when unavailable. The approximate t intervals use five outer data seeds, not2500 batches or ten model instances.

The plotting/materials script uses this task's local staging paths; when running in a repository checkout, point ROOT and FIG to the corresponding `results/rq3_prediction/` paths. This changes only output locations, not the analysis.
