# RQ2 frozen structural attribution and independent scoring holdout

## Purpose / current state
November1 fixed factorial24 improved compound-fault reference protection. Test whether allowing multiple faults contributes beyond small prior differences. Keep RQ3 and manuscript frozen; overall deadline November28.
## Assumptions / scope
One ablation only: restrict the factorial prior and transition to masks with at most one fault, renormalizing prior and each transition row. Identical emission parameters, memory and train-only calibration. This conditions allowed support; it cannot preserve transition entries exactly while deleting probability mass. Compare original joint12, full24 and restricted12. No tuning.
## Milestones / command
Implement src/rq2_confirmation.py and src/rq2_restricted_fault.py; tests compare unrestricted reproduction, restricted support, causality and no-inflation collapse. Freeze code before new inputs. Run python -m src.rq2_confirmation with explicit data/scorer/generator paths and implementation commit.
Generate32 fresh batches per seed42–46 using unchanged rq3_paired_data generator, namespace2026110201, test split. New160 batches; no quality labels used or prediction fitting. Existing26 stress views, signs/offsets balanced, schedule seed2026110202. Preserve all views and per-seed counts. Primary endpoints are equal-weight means over three strong dual-fault combinations and two record delays: full24 minus restricted12 for point recall (higher) and bystander FPR (lower). Paired n5 t95 intervals; Holm two-sided p across these two endpoints. Original joint12 and all single/clean/moderate/stratum outcomes descriptive. No changing endpoints after viewing.
## Leakage / reproducibility
Thresholds from original reference_cal only, references frozen. All inputs, event schedules, parameters, q and alarms archived. Exact scores can be regenerated from saved observations and parameters. Fresh generator outputs including evaluation-only arrays retained separately; they never enter scoring. Unique output folder, refusal to overwrite. Configuration/code/source hashes saved.
## Validation
Software tests, posterior normalization, count reaggregation, saved arrays and event masks; archived files hash checks. Numerical code Python3.11.
## Progress
- [x] Protocol defined before fresh scoring.
- [ ] Tests, frozen implementation, one fresh run, evidence and upload.
## Decision log
No alternative algorithm, no RQ3 fit, no manuscript edits. SBS unchanged pending site response.
## Discoveries / outcomes
Pending run. Stop after this fixed test and interpret all positive/negative effects; no parameter search.
