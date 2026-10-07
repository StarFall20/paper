# Round 29: counterfactual policy-path recoverability

## Core idea

The stronger candidate is a policy-path disagreement certificate. A DCE can
contain two models with similar observed-task likelihoods and different
counterfactual implications. The diagnostic evaluates the fitted models on a
predeclared feasible path of attribute interventions, so the model comparison
targets the use for which choice models are commonly built: demand and policy
prediction.

Let (m_0) be the additive MNL and (m_S) the declared structured utility
model. For a finite policy path (mathcal P={(q,p)}), where (q) and (p)
are quality and price shifts applied to product alternatives, define

\[
D_{mathcal P}(m_0,m_S)=
\frac{1}{|\mathcal P|}
\sum_{(q,p)\in\mathcal P}
\frac{1}{N}\sum_{i=1}^{N}
\left\|\hat P_{m_0}(\cdot\mid X_i^{q,p})-
\hat P_{m_S}(\cdot\mid X_i^{q,p})\right\|_1 .
\]

The path is fixed before outcomes are used. In the simulation it contains the
nine ((-0.6,0,0.6)^2) quality/price shifts. A second signal is the
respondent-level score-overdispersion ratio after the structured utility has
absorbed the declared observable terms. A policy alert indicates an
observable utility departure; a score alert indicates persistent individual
variation. Both signals together produce unresolved status.

This differs from comparing models on a test log loss. It measures whether
model choice changes the counterfactual demand surface. If two models agree on
the observed support but disagree along (mathcal P), the disagreement is
policy-relevant even when ordinary predictive scores are close.

## Simulation result

The policy-path audit uses 100 additive calibration replications and 24
evaluation replications per mechanism, with respondent-grouped development,
validation, and test samples. The policy path is calibrated at the 99th
percentile; the score gate uses the 97.5th percentile after structured-model
refitting.

| condition | policy alert | score alert | action accuracy | unresolved rate |
|---|---:|---:|---:|---:|
| additive | 0.000 | 0.000 | 1.000 | 0.000 |
| nonlinear | 1.000 | 0.000 | 1.000 | 0.000 |
| threshold | 0.750 | 0.000 | 0.750 | 0.000 |
| interaction | 1.000 | 0.042 | 0.958 | 0.042 |
| heterogeneity | 0.042 | 0.958 | 0.917 | 0.042 |
| nonlinear + threshold | 1.000 | 0.000 | 1.000 | 0.000 |
| nonlinear + interaction | 1.000 | 0.000 | 1.000 | 0.000 |
| combined | 1.000 | 0.625 | 0.625 | 0.625 |

The key separation is between observable structure and random taste
heterogeneity. The additive false-alert rate is zero in the 24-run evaluation;
the nonlinear and interaction conditions are detected without a score alert,
while the isolated heterogeneity condition produces score alerts with only one
policy-path false alert. The combined condition reaches unresolved status in
62.5% of runs. Threshold effects remain the weakest observable departure,
although the policy path detects 75.0% at the current signal strength.

The policy-path gap also tracks the direction of model regret across mechanism
conditions. The pooled correlation between (D_{mathcal P}) and the
base-minus-structured test decision regret is 0.95. This correlation is a
mechanism-level result; within a single condition the path gap is a diagnostic
signal, not a calibrated point forecast of regret.

Calibration must scale with the sample-size grid. A preliminary 50-replication
null calibration produced unstable high-quantile thresholds at (N=800). After
raising the independent additive calibration to 200 replications, the (N=800)
run gives 0/24 additive policy alerts, 24/24 alerts for nonlinear, threshold,
and interaction mechanisms, and 24/24 score alerts for isolated heterogeneity.
The power curve is therefore reported with its calibration resolution instead
of treating one threshold as transportable across sample sizes.

## Public Swissmetro check

The canonical purpose-1/3 Swissmetro DCE contains 6,768 observations from 752
respondents. An additive MNL and an alternative-specific time/cost MNL have
test log losses 0.79821 and 0.76579 under the same respondent split. Across a
predeclared multiplicative time/cost path (\{0.8,1.0,1.2\}^2), their mean
choice-probability L1 gap is 0.06468 and their mean predicted-share L1 gap is
0.01864. The public data have no mechanism labels, so this is evidence of
counterfactual sensitivity, not a causal explanation of the gap.

## Novelty and collision boundary

Existing work studies model uncertainty in welfare analysis, compares model
families, or develops flexible choice estimators. The present object is
narrower and operational: it defines a finite, feasible policy path before
outcomes are used, measures model disagreement on that path, calibrates a
policy-alert threshold under an additive DCE null, and combines it with a
respondent score gate to separate observed utility misspecification from
heterogeneity. It does not claim a new welfare bound or a new random-coefficients
estimator. The contribution is the recoverability certificate for the
counterfactual surface and its controlled mechanism benchmark.

The collision audit distinguishes four adjacent literatures. Welfare
misspecification studies quantify bias in welfare measures after choosing a
model; model-choice studies show that estimates such as loss aversion can
change with specification; ML/choice benchmarks compare out-of-sample scores;
and heterogeneity papers test parameter instability. The present certificate
uses a pre-outcome intervention path as the diagnostic object and reports an
action before policy simulation. It should be cited alongside, and not claimed
to replace, [robust welfare sensitivity](https://swlb2.aeaweb.org/articles?id=10.1257/aer.20220673),
[JOCM model-choice sensitivity](https://doi.org/10.1016/j.jocm.2024.100524),
[JOCM ML/DCM benchmarking](https://doi.org/10.1016/j.jocm.2024.100470), and
[JOCM structural heterogeneity testing](https://doi.org/10.1016/j.jocm.2025.100573).

The earlier fibre statistic remains outside this claim because its saturated
finite-support version has the same unrestricted alternative as a profile LR.

Raw outputs are `results/policy_path_triage.csv`,
`results/policy_path_triage_summary.csv`,
`results/policy_path_power_n800_cal200.csv`, and
`results/swissmetro_policy_path_audit.csv`. The executable entry points are
`analysis/policy_path_triage.py`, `analysis/policy_path_disagreement.py`,
`analysis/policy_path_power_curve.py`, and
`analysis/swissmetro_policy_path_audit.py`.
