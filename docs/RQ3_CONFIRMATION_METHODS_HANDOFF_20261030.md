# Frozen Methods and experimental design — ready for manuscript integration

Date: 2026-10-30. The main manuscript remains owned by the writing task. This document freezes methods and experimental design; it contains no unobserved confirmation results.

## Methods

We use the existing regime-conditioned Trust Score to decide when to correct a sensor observation. For sensor j at time t, let r_tj be the absolute exponentially smoothed residual divided by its reference-calibrated threshold. The retained score is S_tj = exp(−r_tj²/2), with acceptance boundary S0 = exp(−1/2). Let q_t(z) denote the filtered regime probability and c_tj = sum_z q_t(z) μ_zj the corresponding reference mean in physical sensor units. Selective correction is defined by

\[
\lambda_{tj}=\left[1-S_{tj}/S_0\right]_+,
\qquad \widetilde{x}_{tj}=x_{tj}+\lambda_{tj}(c_{tj}-x_{tj}).
\]

Thus, r_tj ≤ 1 leaves the observation unchanged; larger residuals increase reference substitution continuously. The rule adds no quality-label-tuned threshold and retains the original Trust Score and predictor architecture. Continuous correction uses λ_tj = 1−S_tj. For paired uncorrupted observations x⁰, corruption e=x−x⁰ and reference error b=c−x⁰ give

\[
(\widetilde{x}-x^0)^2=(1-\lambda)^2e^2+\lambda^2b^2+2\lambda(1-\lambda)eb.
\]

This identity expresses the recovery–substitution trade-off. It motivates preservation of accepted measurements; it does not imply universally improved target prediction.

## Confirmation design

Five independently seeded synthetic datasets each provide 1000 training sequences and250 validation sequences from the existing version. The generation law and both targets are unchanged. An observed118-step prefix predicts an arrival temperature-exposure proxy at step180. We train five matched groups: no-score N, continuous actual-score correction NG, selective actual-score correction SG, selective randomized-score correction SR, and selective correction without the fault hypothesis SA. All groups receive the same full-candidate regime posterior as a common feature. Consequently, SG−SA isolates the fault hypothesis in scoring/correction conditional on this common posterior; it does not isolate the entire regime-inference pipeline.

Each main group has two optimization repetitions per data seed, giving50 main fits. The model has18 input channels,32 hidden units,6689 parameters, a single LSTM layer and mean-hidden-state readout. Training uses40 epochs,320 updates, AdamW with learning rate0.003 and weight decay0.0001, batch size128, and gradient clipping1. Model selection uses the earliest checkpoint minimizing six-view validation MAE. Initialization and shuffle streams differ between repetitions but match across arms. Each sequence's training-view schedule is shared across arms and repetitions. The reference model and calibration procedure are frozen; existing seed42–44 fitted references are reused, and the same procedure extends to45–46.

The six primary scenarios cross correct versus12-step-delayed regime records with clean observations, a one-scale weak drift, or a four-scale strong bias. Single-channel events last12 steps and start6 steps before or18 steps after the first regime transition, balancing channel and sign. The label and latent process remain paired across views. Random score trajectories come only from the training pool, excluding all views of the recipient sequence during training. Fifteen additional N/SG/SR fits predict the existing independent null target, using the first optimization repetition. A training-mean predictor is reported for this negative control without fitting an additional model.

After all65 checkpoints are locked,500 new independent test batches per data seed are generated once under namespace2026103001, retaining the original generating parameters. The2500 test sequences are evaluated on all six primary scenarios and two prespecified secondary views (six-step record delay with clean observations or strong bias). Hidden process and target files remain separate from predictor inputs. Test results do not change the method, checkpoint or training budget.

## Statistical analysis

The primary estimands are SG−N and SG−SR MAE differences, averaged equally over the six scenarios. Within each data seed, we average the two optimization repetitions before computing the five-seed paired mean, standard deviation and approximate95% t interval (four degrees of freedom). The primary predictive-value claim requires support against both comparators. SG−NG describes the selective-use contribution; SG−SA evaluates the additional fault hypothesis. Per-scenario, affected-channel, event-offset and terminal-regime analyses are descriptive and retain every prespecified stratum. We report individual optimization repetitions and leave-one-data-seed mean ranges as sensitivity analyses. Test batches, scenarios and models sharing a dataset are not treated as independent outer repetitions. The five-seed interval describes data-seed variability for the selected two optimization streams; two streams do not precisely characterize the population of all possible training initializations.

## Integration into the existing manuscript

RQ1: retain original single/pooled/regime reference evidence and definitions. RQ2: retain sensor consistency, fault localization and healthy-channel contamination evidence; the joint-inference module's detection value stands separately from its predictive increment. RQ3: introduce selective use as the lightweight bridge from consistency assessment to prediction. Preserve earlier negative results as evidence that naive score inclusion or unconditional correction is insufficient. Formal Results and figures will be appended from the locked confirmation, without rewriting the research direction.

## SBS parallel route

No new verified process–assay pairs or implemented field experiment have arrived in this task. The existing two-condition/two-duration/three-independent-container design remains a feasibility proposal. The outstanding needs are the prediction time/horizon, verified sample genealogy, actual process duration and implementable measurement conditions. No historical process values are inferred from endpoint assays. SBS remains outside the current main confirmation and does not delay manuscript integration.
