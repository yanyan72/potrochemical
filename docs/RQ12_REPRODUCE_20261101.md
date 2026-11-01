# Restore and reproduce the bounded RQ1/RQ2 studies

Source baseline: ce898f57. Protocol: ef3dafe6. Diagnostic/RQ1 implementation:60595e17. Fixed factorial extension:6bb17e32. Original RQ3 models/test and manuscript are unchanged.

## Restore saved results

The `results/rq12/20261101/archives/` directory contains 18 ZIPs and `archive_index.json`: one root archive and five seed archives for each of diagnostic, candidate validation, and RQ1 confirmation. Each seed archive contains actual observation arrays, masks, ratios, alarms, regime posteriors and relevant parameters/tables; these are not summary-only releases. Total compressed size is about310MB. Each ZIP is below GitHub's50MB advisory size.

From the repository root (Python3.11):

```bash
python -m src.restore_rq12 --archives results/rq12/20261101/archives --output /tmp/rq12-restored
python -m src.verify_rq12 /tmp/rq12-restored/diagnostic_v1 /tmp/rq12-restored/factorial_validation_v1 /tmp/rq12-restored/rq1_confirmation_v1 --repository-root . --output /tmp/rq12-verification.json
```

Restoration refuses overwrite and checks ZIP SHA256, CRC, extracted manifest hashes. The independent verifier recomputes batch counts, per-seed strata, posterior normalization, alarm/ratio identity, and primary RQ1 statistics. For original-source checks, provide the original dataset under `data/processed/rq3_paired_20261019_rebuild_v1` and restore the previously released RQ3 training archive into `results/rq3_prediction/rq3_selective_confirmation_20261030_v1/training` using its published restore instructions. No predictor is loaded or trained by this study. Source path remapping is supported by `--repository-root`; downloaded E1b text copies differ from original Git blobs only by one terminal LF, explicitly accepted for source identity checks.

## Re-execute scores in new output directories

```bash
python -m src.rq12_stress diagnostic --config configs/rq12_20261101.json --output /tmp/rq2-diagnostic-replay --commit 60595e17c2719eaa5871728684093fc800e3fd3d --data data/processed/rq3_paired_20261019_rebuild_v1 --scorers results/rq3_prediction/rq3_selective_confirmation_20261030_v1/training
python -m src.rq12_stress diagnostic --candidate --config configs/rq12_20261101.json --output /tmp/rq2-candidate-replay --commit 6bb17e32cf8aed8c2518b163fb594a7fb2006f31 --data data/processed/rq3_paired_20261019_rebuild_v1 --scorers results/rq3_prediction/rq3_selective_confirmation_20261030_v1/training
python -m src.rq12_stress rq1 --config configs/rq12_20261101.json --output /tmp/rq1-replay --commit 60595e17c2719eaa5871728684093fc800e3fd3d --references results/logistics_e1b/logistics_e1b_20260918_v1 --base-config configs/logistics_e1b.yaml
python -m pytest tests/test_rq12_stress.py -q
```

Diagnostic and candidate validation reuse distinct old-val slices0:64 and64:128. They are developmental. RQ1 uses fresh namespace2026110102,180 independent fixed-regime sequences and unchanged fitted references. All thresholds derive only from original reference calibration. Views, channels and time points do not become independent replicates; statistical outer n is five seeds.

`src/analyze_rq12.py`, `src/inspect_rq2_posterior.py` and `src/plot_rq12.py` expect restored stages under a local `results/` directory. They read stored arrays and produce descriptive summaries/figures; no algorithm changes. Python3.11.17/NumPy2.4.6 ran experiments and statistical analysis; the existing Python3.14 graphical environment rendered figures only.

The first test invocation used an incorrect working-directory-relative path, and the first plotting attempt lacked Matplotlib in the scientific environment; logs retained. One large GitHub blob upload returned ReadTimeout, but the subsequent tree creation proved the exact blob had been stored, and its expected Git SHA was verified. No experiments were rerun to select a favorable outcome.
