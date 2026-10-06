# Journal of Choice Modelling revision plan

## Research question

**Independent pivot under review:** Can a utility basis be rejected by
randomized choice-task pairs that preserve the candidate model's utility
differences while changing the attribute decomposition? The paired-task test
is the only new object being considered for the main contribution. The
recoverability and process-triage analyses remain comparison baselines until
the paired-task data condition is verified.
An observational matched-task version may use frozen utility-difference
coordinates in Swissmetro, but it requires a separate clustered bootstrap and
is weaker than randomized pairs.

The implemented fallback uses cross-fitting and respondent-clustered multiplier
bootstrap. Its current boundary run controls the null at about 5% but has only
modest power against the engineered nonlinear and interaction conditions. It
is an audit of feasibility and does not replace the exact paired-task design.

The exact randomized benchmark now gives strong base power, and the damped-
Newton repair candidate returns rejection to the declared size for the
nonlinear, threshold, and interaction conditions. Interaction localization
remains incomplete because the joint shift changes multiple raw loci. The
four-cell probability-scale factorial audit was run and failed under a
combined interaction and nonlinear main effect, so it is excluded from the
contribution. The cluster sign-flip version is the current method object; it
remains high-risk until a purpose-built empirical supplement or a different
valid paired-task dataset is available.

Can a grouped-validation diagnostic decide whether a choice problem needs utility enrichment, discrete segmentation, continuous heterogeneity, process/sequence modelling, or candidate-library expansion? When the observed data cannot distinguish competing mechanisms, can an explicit unresolved branch reduce policy loss relative to forced model selection?

## Current baseline contribution

The paper develops a process- and recoverability-aware model-family triage rule for choice-model specification. A choice-set-aware diagnostic ranks candidate utility terms, a formal heterogeneity gate tests whether respondent-level structure is missing, a sequence diagnostic tests omitted state dependence, and the selected branch is refit in a behavioural model. When evidence cannot distinguish taste heterogeneity from attribute non-attendance, the rule abstains and identifies the missing measurement needed for resolution. The contribution is assessed through structure recovery, parameter and WTP recovery, calibration, external prediction, decision regret, computational cost, parsimony, and the cost of forced decisions. The paper presents an empirically tested decision rule for analysts, not a new generic search algorithm.

This baseline is below the preferred innovation threshold if presented as the
main paper claim. The standalone process branches overlap established JOCM
work and are retained as failure-boundary comparisons.

## Candidate main contribution

The model-equivalent choice-pair test constructs task pairs with identical
candidate-model utility-difference vectors and different attribute
decompositions. Under the candidate utility basis, the pair has identical
choice probabilities. A respondent-cluster sign-flip reference tests that
equality and can stratify violations by predeclared transformations. It does
not identify a unique omitted nonlinear, threshold, or interaction term. The
method is adopted only if a valid paired-task instrument is available.

## Evidence package

1. **Monte Carlo study:** at least 100 replications per condition. Conditions vary one mechanism at a time and include a combined condition. The true utility contains linear effects plus selected quadratic, interaction, and threshold terms, with optional random taste heterogeneity.
2. **LPMC:** large revealed-preference panel. Use the first two years for model construction and the final year for temporal external validation.
3. **Swissmetro:** panel stated-preference benchmark. Group all observations from each respondent in resampling and cross-validation.
4. **Small-sample stress test:** retain the existing pilot as a robustness analysis, with claims limited to finite-sample behaviour.

## Model set

- MNL baseline
- Mixed Logit for continuous taste heterogeneity
- Latent Class MNL for discrete heterogeneity
- Enriched interpretable utilities using prespecified quadratic, spline, hinge, and interaction terms
- Random Forest as the current diagnostic learner; XGBoost as the planned exact-runtime robustness learner
- ML-assisted specification: learner diagnostics, candidate-term screening, behavioural refit, nested/group validation, and parsimony selection

## Evaluation

### Behavioural recovery

- coefficient bias, RMSE, and empirical coverage
- WTP bias and RMSE
- recovery of interactions and thresholds
- false-positive term rate
- specification stability across resamples

### Predictive performance

- held-out log loss
- Brier score and calibration slope/intercept
- predicted choice shares
- individual-level choice prediction
- temporal external validation for LPMC

### Reporting discipline

Each paragraph has one job: motivate the question, define the method, explain identification, report evidence, or state the implication. Technical implementation details move to the methods subsections and appendix. The main text presents the decision logic before estimator settings.

## Required manuscript changes

- Replace the model-race framing with the ML-assisted specification question.
- Separate heterogeneity, nonlinear utility, threshold effects, and omitted interactions in the theory and simulation design.
- Remove claims based only on random train/test splits.
- Report validation design before reporting performance numbers.
- Add a limitations paragraph covering data licensing, transferability, and computational choices.
- Add a candidate-library coverage or omitted-term stress test so the mechanism-specific failure boundary is explicit.
- Add a recoverability ceiling that separates observable approximation error from latent-heterogeneity error.
- Add a permutation-robustness audit for the alternative-specific learner; if the current RF ranking changes after product-alternative permutation, replace it with a choice-set-aware diagnostic.
- Add a formal respondent-level heterogeneity gate before expanding to Latent Class or Mixed Logit.
- Add a process/sequence gate and an unresolved branch; do not claim that choice data alone identify attribute non-attendance versus random taste heterogeneity.
- Compare the triage rule with direct ANA and decision-rule models, and report the policy cost of forcing a model family when the mechanism remains unresolved.
- Position the contribution against Ortelli et al. (2021), Hernandez et al. (2023), Beeramoole et al. (2023), Delphos, and recent LLM-assisted specification work.
