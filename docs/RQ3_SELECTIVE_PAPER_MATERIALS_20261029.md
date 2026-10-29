# Manuscript-ready experimental materials: selective correction

Version: 2026-10-29. These paragraphs describe development experiments, for integration into the existing RQ1–RQ3 manuscript. The main manuscript has not been edited.

## Methods — selective use of sensor trust

We retain the original regime-conditioned Trust Score and introduce a selective correction rule between the scoring module and the quality predictor. Let \(r_{tj}\) denote the absolute smoothed residual normalized by its calibrated threshold, and let \(S_{tj}=\exp(-r_{tj}^{2}/2)\). The acceptance boundary \(r_{tj}=1\) implies \(S_0=\exp(-1/2)\). Using the shared regime posterior \(q_t(z)\), we obtain the reference mean in the original sensor units as \(c_{tj}=\sum_z q_t(z)\mu_{zj}\). We define

\[
\lambda_{tj}=\left[1-\frac{S_{tj}}{S_0}\right]_+,
\qquad \widetilde{x}_{tj}=x_{tj}+\lambda_{tj}(c_{tj}-x_{tj}).
\]

The mapping leaves observations inside the calibrated acceptance region unchanged. Outside this region, it continuously increases the contribution of the regime reference. Since \(0\leq\lambda_{tj}\leq1\), the corrected observation remains a convex combination of the observed value and its reference. The threshold is inherited from score calibration and is not optimized against quality labels. The score is a consistency measure rather than a calibrated probability of sensor failure.

This rule can be interpreted through an error decomposition. For a paired observation without injected corruption, \(x^0\), define \(e=x-x^0\) and \(b=c-x^0\). Then

\[
(\widetilde{x}-x^0)^2=(1-\lambda)^2 e^2+\lambda^2 b^2+2\lambda(1-\lambda)eb.
\]

Correction can reduce the retained corruption term while introducing reference substitution error. Selectivity addresses this trade-off by suppressing intervention when the existing score does not provide sufficient anomaly evidence. The paired uncorrupted observation still contains ordinary measurement noise; it is not the latent physical truth.

We trained nine additional LSTMs: three data seeds crossed with selective correction using actual scores (SG), randomized training-pool score trajectories (SR), or scores and references computed without the fault hypothesis (SA). All models retained 18 input channels, 32 hidden units, 6689 parameters, 40 epochs and 320 updates. The previous no-score (N) and continuous-correction (NG/RG/AG) predictions were reused. Training sequences, targets, fault events, view schedules, initialization and minibatch order were matched. The complete candidate regime posterior was provided to every group as a common input; therefore, the SG–SA contrast evaluates the fault hypothesis in the scoring/correction branch conditional on this shared information.

## Results — development evidence for the correction trade-off

Across the six equally weighted scenarios, selective correction achieved a mean MAE of 1.59096, compared with 1.60405 for the no-score baseline, 1.60199 for randomized selective correction, and 1.61156 for continuous correction. The corresponding paired differences were −0.01309, −0.01103 and −0.02059, respectively. Each difference was negative for all three data seeds. Relative to the no-score baseline, the overall reduction was 0.82%.

The scenario analysis showed that selective correction preserved the strong-bias benefit while reducing normal-observation costs. Under correct and delayed regime records, the strong-bias MAE reductions relative to N were 2.66% and 2.99%. The clean-scenario increases were only 0.06% and 0.32%, compared with 2.01% and 1.69% for continuous correction. Weak-drift increases similarly decreased from 2.26% and 1.83% to 0.24% and 0.39%. Thus, the improvement did not require a new score or better weak-drift detection.

Temperature faults accounted for most of the predictive benefit. The SG–N differences under strong temperature bias were −0.15877 and −0.16562 for correct and delayed records, respectively, with improvements in every data seed. Humidity-fault differences were near zero or mildly negative, whereas vibration-fault differences remained positive. For clean temperature observations with correct records, approximately 0.80% of points activated correction and the average correction strength was 0.081%, compared with 6.25% for continuous correction.

The overall SG–SA difference was +0.00036, providing no overall advantage for the fault hypothesis in the correction branch. Nevertheless, the corresponding strong-temperature-fault differences were −0.00173 and −0.00266, both negative in all three seeds. This small subgroup trend motivates retaining the ablation in confirmation, while attributing the main present improvement to selective use of the score.

## Discussion — connection to RQ1–RQ3

The findings support treating reference construction, sensor consistency and downstream intervention as related but distinct decisions. RQ1 establishes the value of regime-conditioned references, and joint regime–fault inference extends reference selection when operating records and sensor observations are both uncertain. RQ2 evaluates whether the resulting scores detect corruption without contaminating healthy-channel references. RQ3 then asks whether using these scores improves prediction under a matched learning budget. Selective correction provides a direct and interpretable connection between the latter two questions without replacing the original score or predictor architecture.

An anomaly score may be useful even when unconditional application of that score is harmful. In these experiments, continuous correction attenuated informative normal variation; restricting correction to the existing anomaly region reduced this cost while retaining substantial recovery under strong temperature corruption. The effects also depended on the target: temperature exposure drives the synthetic quality proxy, so improved vibration detection need not yield comparable predictive gains. This motivates separating measurement consistency from task relevance rather than modifying the data-generating target to reward every sensor channel.

The evidence is encouraging but remains developmental: the study uses three data seeds and one shared optimization seed, and validation data select checkpoints. The six paired scenarios are not independent training repetitions. The next experiment should freeze the current method, evaluate independent test batches with crossed optimization repeats, and retain the randomized and no-fault-hypothesis controls. This will establish whether the observed trade-off improvement generalizes and whether the small joint-inference subgroup effect persists.

## Figure captions and source mapping

**Figure 1. Selective correction across six scenarios.** Mean paired MAE differences across three data seeds. Negative values indicate an advantage for SG. SR uses randomized score trajectories with the same selective mapping; RG uses the previous continuous mapping; NG uses actual scores continuously; SA removes the fault hypothesis from the scoring/correction branch. Source: `paired_summary.csv`, stratum `all`; figure `01_scenario_comparisons.pdf`.

**Figure 2. Recovery and normal-information loss by data seed.** MAE differences relative to the no-score model for continuous and selective correction. Correct and delayed record conditions are averaged within each corruption category and seed. The plot retains all seeds and shows the reduced clean/weak-drift cost alongside the retained strong-bias benefit. Source: `metrics_by_seed.csv`; figure `02_tradeoff_by_seed.pdf`.

**Figure 3. Strong-bias effects by affected sensor channel.** SG–N MAE differences under correct and delayed records. Markers show means and bars show standard deviations across three data seeds, not confidence intervals. Source: `paired_summary.csv`, channel strata; figure `03_channel_effects.pdf`.

Main tables: full six-scenario and per-seed comparisons are provided in `RQ3_SELECTIVE_FINDINGS_20261029.md`. All source tables and both PNG/PDF figure formats are archived with the experiment. No enterprise SBS validation is claimed by these materials.
