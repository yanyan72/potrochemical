# Selective correction confirmation — ExecPlan

Date: 2026-10-30. User authorized 50 main models plus 15 null-target controls.

## Purpose
Confirm the predictive value of selective use of the existing Trust Score, identify whether the fault hypothesis contributes independently, and deliver Methods/Results/Discussion and figures to the manuscript owner by 2026-11-28.

## Current state
Main a7660ed4 contains 21-model and 9-model development results. Selective correction improved three development seeds overall; SG-SA showed no overall additional benefit. Reuse src/rq3_joint_prediction_dev.py, rq3_selective_prediction_dev.py, rq3_shared_mode_fault_development.py, rq3_uncertainty_a0.py and the frozen rq3_paired_data generator. Existing 10000-batch dataset and all prior results remain unchanged.

## Scientific assumptions
Synthetic temperature-exposure proxy is the arrival target, not an enterprise assay law. 118 observed steps predict the 180-step endpoint. Six equally weighted scenarios do not represent industrial prevalence. Trust Score is consistency, not calibrated fault probability. New test uses unchanged generating parameters in namespace 2026103001.

## Scope
Five data seeds 42-46, two optimization repetitions 29001/39001 and 29002/39002, five arms N/NG/SG/SR/SA =50 models. Null target: first repetition, N/SG/SR=15 models. No threshold search or further gating mechanisms. Six primary views plus prespecified delay6 clean/bias secondary views. No main manuscript rewriting; SBS remains independent.

## Milestones
1. Freeze this plan and config; implement src/rq3_selective_confirmation.py and tests. Run pytest and a tiny existing-data smoke replay, never the fresh-test namespace. Validate feature parity with prior archives and both optimization streams. Upload protocol and implementation before fitting.
2. `python src/rq3_selective_confirmation.py train --seed <42..46> ...`: reuse train1000/val250 and reference_fit/cal per seed. For42-44 reuse exact saved score models/events/scaler/random draws;45-46 apply identical fixed fitting/calibration procedures to existing reference data. Train13 models per seed, persist checkpoints/predictions/history and source hashes. Upload and lock all65 models before test generation.
3. `... evaluate --seed <42..46> ...`: once generate500 new test batches per seed with original generator, save visible data separately from evaluation-only truth/targets. Use balanced 12-step events at first switch-6 or+18, fixed scales, unchanged scoring. Save predictions before computing metrics. No retraining after test.
4. `... summarize ...`: all seeds/views/channel/offset/regime strata; paired statistics, software/integrity verification, figures and bilingual manuscript materials. Publish complete results and send materials to the identified writing thread.

## Data leakage controls
Train/val remain original complete sequences. Reference calibration never uses quality labels. Random score trajectories drawn from training pool only, excluding recipient IDs in training. All arms share q inputs and other features. Same fault events/schedule/order/initialization within repetition; two optimization streams. New test namespace excluded from software fixtures and inaccessible until training lock. No existing test used for selection. No future hidden process in predictors. Original targets and data unchanged.

## Reproducibility
Config records all seeds, 40epochs/320updates, 18input/32hidden/6689parameters, inherited threshold exp(-1/2). Files sealed with SHA256 and checkpoints reloaded. GitHub protocol, implementation, training lock, results and receipt commits recorded. Output refuses overwrite; partial results retained. Estimate65 fits about11 minutes CPU from development timing, plus scoring, verification and upload; no GPU needed.

## Validation
Tests cover threshold reuse, arbitrary delay6 semantics, seed aggregation (not pooled batches), old development feature identities, train-only random scores, new namespace disjointness and tiny deterministic two-stream fit. Independently recompute metrics from saved predictions and targets; recheck all source hashes,65 budgets, checkpoint replay and remote blobs/archive manifests. Figures visually inspected.

## Statistical estimands
Primary: SG-N and SG-SR mean absolute error differences, equal weights across six scenarios, then optimization repeats averaged per data seed. Five outer seed differences, mean/SD and approximate95% paired t CI (df4). A joint positive claim needs both comparisons; no best-seed/model/scenario selection. SG-NG isolates selectivity; SG-SA tests scoring/correction fault hypothesis conditional on shared full-candidate q. Show per-optimization results and leave-one-seed means; secondary/channel strata descriptive. Null control only one optimizer repetition by budget design; also show train-mean predictor. No claim of exact equivalence from an interval containing zero.

## Progress
- [x] User accepted development and authorized fixed65-fit confirmation.
- [x] Inspected current sources and source-data availability; new seed45/46 reference models require fixed calibration, not method tuning.
- [x] Implementation and focused tests completed; implementation will be committed before fitting.
- [ ] 65 fits and training lock uploaded.
- [ ] New2500-batch held-out evaluation and verification.
- [ ] Complete evidence package and writing-thread handoff.

## Decision log
Keep the selected mapping and parameters unchanged. Reuse first optimization stream for continuity; add an independent second stream. Null controls use only first stream as previously proposed. Joint inference's anomaly-detection value and its conditional predictive increment are reported separately. SBS requires verified process-assay pairing and site execution; no new enterprise data or operations are invented.

## Surprises and discoveries
Pending execution; all outcomes, including failures, will be retained.

## Outcomes and retrospective
Pending. Once confirmation is complete, freeze results and prioritize manuscript integration over additional model search.
