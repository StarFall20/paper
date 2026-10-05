# Framework and model audit

## Resolved issues

1. **Design drift.** `simulation/design.yaml` described an older six-condition plan. It now records the eight active conditions, the 30-replication current benchmark, the 100-replication target, the respondent-level split, and planned external validation.
2. **Mixed Logit simulation design.** The targeted extension now keeps the respondent trajectory together, uses paired normal draws, records estimation and prediction draw counts, and includes a draw-stability script for 20, 40, and 80 draws.
3. **Dimension brittleness.** Alternative count, attribute count, opt-out index, and price index are explicit constants. The feature builder now raises an error when the alternative dimension does not match the declared design.
4. **Environment mismatch.** The requirements file now matches the NumPy version used by the local analysis environment. The XGBoost OpenMP limitation remains recorded as an execution constraint.

## Checks that passed

- Nested candidate-term selection uses only respondent-level training data for the inner choice.
- The corrected data-generating process separates observed covariate interactions from unobserved respondent-specific price sensitivity.
- The opt-out alternative has zero product attributes and an explicit utility indicator.
- The smoke test produces 32 simulation rows, 16 tree-model rows, and 8 targeted Mixed Logit rows.
- The draw-stability check shows small changes in the heterogeneity and combined conditions between 20, 40, and 80 paired draws.

## Open threats to validity

1. **Replication precision.** The corrected benchmark has 30 replications per condition; the target is 100. The current means are suitable for development and mechanism checks, not the final uncertainty statement.
2. **Behavioural recovery.** Coefficient bias, WTP recovery, calibration, term-selection stability, and coverage are not yet in the result tables.
3. **Model scope.** The Mixed Logit extension has one random price coefficient. The final model needs a documented random-coefficient vector and convergence diagnostics.
4. **Flexible learner scope.** HistGradientBoosting is a proxy. Exact XGBoost results require a working OpenMP runtime and a locked tuning budget.
5. **External validity.** LPMC temporal validation and Swissmetro respondent-grouped validation are still pending.

## Interpretation rule

The evidence supports a conditional workflow: term discovery helps when the missing utility structure is observable and represented by candidate terms. Continuous latent heterogeneity requires a model that represents random coefficients. The paper should keep these claims separate in the theory, tables, and conclusion.
