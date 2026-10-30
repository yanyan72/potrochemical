# Confirmed experimental materials for the original RQ1–RQ3 manuscript

2026-10-30. Replace the earlier development-only RQ3 results with the confirmation below, while retaining the development history where it explains the design. Integrate locally into Methods, Results and Discussion; retain the original research questions and core Trust Score. Detailed frozen design: [Methods handoff](RQ3_CONFIRMATION_METHODS_HANDOFF_20261030.md). Full results: [tables](RQ3_CONFIRMATION_TABLES_20261030.md).

## Methods: selective reference correction

For the absolute smoothed residual ratio \(r_{tj}\), retain the original consistency score \(S_{tj}=\exp(-r_{tj}^{2}/2)\). The calibrated acceptance boundary \(r_{tj}=1\) gives \(S_0=\exp(-1/2)\). With filtered regime probabilities \(q_t(z)\) and reference means in physical units, define

\[
c_{tj}=\sum_z q_t(z)\mu_{zj},\qquad
\lambda_{tj}=\left[1-\frac{S_{tj}}{S_0}\right]_+,\qquad
\widetilde{x}_{tj}=x_{tj}+\lambda_{tj}(c_{tj}-x_{tj}).
\]

The rule retains observations inside the acceptance region and increases reference substitution continuously outside it. It introduces no threshold tuned against quality labels and leaves the underlying score and predictor architecture unchanged. Continuous correction instead uses \(\lambda_{tj}=1-S_{tj}\). For a paired observation without injected corruption, \(x^0\), let \(e=x-x^0\) and \(b=c-x^0\). Then

\[
(\widetilde{x}-x^0)^2=(1-\lambda)^2e^2+\lambda^2b^2+2\lambda(1-\lambda)eb.
\]

This identity describes the trade-off between reducing corruption and introducing reference-substitution error. The paired observation includes ordinary measurement noise and is not the latent physical truth.

## Experimental design and inference

We froze five groups: no-score prediction (N), continuous actual-score correction (NG), selective actual-score correction (SG), selective randomized-score correction (SR), and selective correction without the fault hypothesis (SA). Each group used five data seeds and two optimization repetitions, yielding 50 main models. Fifteen additional N/SG/SR models predicted the original independent null target using the first optimization repetition. All groups shared the same full-candidate regime posterior and other common inputs, training sequences, faults, view schedule and matched initialization/order within each repetition. SG–SA therefore tests the fault hypothesis in the scoring/correction branch conditional on common regime information. SR draws score trajectories from training sequences only.

Each seed reused 1000 training and 250 validation batches. The fixed predictor was an 18-input, 32-hidden-unit, single-layer LSTM with mean-hidden readout and 6689 parameters, trained for 40 epochs and 320 updates. The earliest minimum six-view validation MAE selected the checkpoint. After all 65 models were locked, 500 new test batches per data seed were generated with the original synthetic law, giving 2500 independent test batches. The six primary paired views cross correct or 12-step-delayed records with clean observations, weak drift or strong bias. The same latent process and quality label are retained across views. Six-step delay with clean observations and strong bias provides two prespecified secondary views.

The primary contrasts are SG–N and SG–SR, with equal weights over the six scenarios. We average the two optimization repetitions within each data seed, then report the mean, standard deviation and approximate 95% paired t interval over five data seeds (four degrees of freedom). Individual optimization repetitions and leave-one-data-seed means assess sensitivity. Scenario and channel comparisons are descriptive; batches, correlated views and models sharing a dataset are not counted as independent outer repetitions. The joint claim of predictive value requires support against both primary comparators.

## Results: confirmed predictive value

On the independent holdout, SG achieved a six-scenario mean MAE of 1.607635, compared with 1.625892 for N and 1.625515 for SR. The paired differences were −0.018257 (95% interval [−0.031127, −0.005387]) and −0.017881 ([−0.033160, −0.002602]), corresponding to reductions of 1.12% and 1.10%. Both optimizer-averaged contrasts were negative in all five data seeds and remained negative after omitting any one data seed. These results support predictive value from selective use of the existing score in the frozen synthetic task.

The scenario analysis clarified the recovery–retention trade-off. SG reduced strong-bias MAE relative to N by 2.89% under correct records and 3.65% under delayed records, retaining 84.16% and 93.64% of the corresponding NG reductions. Continuous correction provided greater average recovery in these strong-bias scenarios. Selectivity reduced its cost elsewhere: the clean-scenario increase under correct records fell from 1.04% for NG to 0.18% for SG, while the delayed clean scenario changed from a 0.88% increase to a 0.05% reduction. The overall SG–NG difference was −0.006573, but its interval [−0.015216, 0.002070] crossed zero. The findings support a changed trade-off rather than uniform superiority to continuous correction.

Intervention diagnostics were consistent with this interpretation. For clean temperature observations under correct records, mean correction strength decreased from 6.36% to 0.117%, and only 0.95% of points triggered selective correction. Strong temperature-fault windows retained activation rates of 67.52% and 81.66% under correct and delayed records. Temperature faults accounted for most of the predictive benefit, with SG–N differences of −0.13597 and −0.16776. Humidity effects were smaller, and vibration effects were close to zero with intervals crossing zero.

SA achieved a mean MAE of 1.606999. SG–SA was +0.000636 (interval [−0.000713, 0.001985]), providing no evidence of an additional overall predictive benefit from the fault hypothesis in the correction branch. Its anomaly-detection value remained visible: strong-vibration event-added detection was 99.39%/99.63% for the joint branch, versus 93.29%/93.66% without the fault hypothesis (correct/delayed records). Strong-fault bystander false-positive rates were 0.92%/1.03%, versus 1.17%/1.31%. These scoring comparisons are descriptive and distinct from the primary predictive contrasts.

The optimizer sensitivity analysis showed different uncertainty across repetitions. Repetition 0 improved in four of five data seeds but both primary intervals crossed zero; repetition 1 improved in all five with both intervals below zero. The prespecified average over repetitions supported the primary contrasts, without establishing improvement under every initialization. For the independent null target, SG–N was −0.001540 (interval [−0.004029, 0.000948]) and SG–SR was −0.001482 ([−0.004085, 0.001121]). SG's MAE of 3.95990 did not outperform the training-mean predictor's 3.94938. Weak-drift detection also remained low (point recall 2.42%/2.93%), although its effect on downstream prediction in this task was small.

## Discussion: incremental contribution and the original research questions

The principal contribution is a coherent use of regime-conditioned consistency information: intervention is concentrated where the existing score indicates a departure from its reference, preserving normal observations that remain useful for prediction. This complements RQ1's reference construction and RQ2's localization assessment with a tested link to RQ3. The earlier negative results show why merely attaching scores or continuously substituting reference values is insufficient; the confirmation identifies a practical, interpretable way to use those scores.

The results also separate detection value from predictive necessity. Joint regime–fault inference improved localization and reduced healthy-channel contamination, but its extra fault hypothesis was not necessary for the observed overall prediction improvement conditional on the shared regime posterior. SA remained competitive. We therefore attribute the primary predictive result to selective score use, retaining joint inference for its independently measured scoring value rather than claiming that complexity itself generated the gain. A future simplification of the complete inference pipeline would be a separate question, since all present groups retain the common candidate posterior.

This study provides controlled confirmation for a synthetic temperature-exposure target, with five outer data seeds and two selected optimization streams. The t intervals are approximate, and the six-view mixture is an experimental weighting rather than industrial fault prevalence. These limits qualify the scope without changing the observed finding: selective correction recovers much of the strong-fault benefit while reducing the loss of normal predictive information. Enterprise SBS feasibility remains a separate line and is not represented as real-process validation here.

## Figures and tables

**Figure 1. Primary confirmation contrasts.** Large markers show five-seed means of optimizer-averaged paired MAE differences; error bars are approximate 95% t intervals, and small markers retain every data seed. Negative differences favor SG. Source: `paired_summary.csv` and `paired_by_seed.csv`; `01_primary_confirmation.pdf`.

**Figure 2. Six prespecified scenario contrasts.** Mean SG differences from N, SR, NG and SA. The figure shows reduced normal-observation costs and a partial strong-fault recovery trade-off against NG. Source: `paired_summary.csv`; `02_six_scenarios.pdf`.

**Figure 3. Recovery and retention by data seed.** NG, SG and SA differences from N, after averaging the two record conditions and optimization repetitions within each seed. All five seeds are shown. Source: `metrics_by_repeat.csv`; `03_tradeoff_by_seed.pdf`.

**Figure 4. Strong-bias channel attribution.** Descriptive SG–N and SG–SA differences by the affected channel, under correct and delayed records. Source: `paired_summary.csv`; `04_channel_attribution.pdf`.

Recommended main tables: five-group overall MAE and four paired contrasts; six-scenario comparison. Supplementary tables: every seed×scenario, individual optimization repetitions, channel/offset/terminal-regime strata, null controls and delay-six secondary views. Every PNG also has a vector PDF. The release includes all 65 checkpoints and all predictions, not just selected results.
