# Journal of Choice Modelling revision plan

## Research question

Under which data-generating mechanisms do data-driven utility diagnostics improve a behavioural choice model, and when do they fail because the missing structure is latent or absent from the candidate library?

## Main contribution

The paper develops a mechanism-controlled validation benchmark for ML-assisted specification. A diagnostic learner ranks candidate utility terms; the selected terms are refit in a behavioural model. The contribution is assessed through structure recovery, parameter and WTP recovery, calibration, external prediction, decision regret, and parsimony. The paper presents a decision rule for analysts, not a new generic search algorithm.

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
- Position the contribution against Ortelli et al. (2021), Hernandez et al. (2023), Beeramoole et al. (2023), Delphos, and recent LLM-assisted specification work.
