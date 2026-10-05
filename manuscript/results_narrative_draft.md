# Results narrative draft

## Simulation question

The simulation asks when diagnostic term discovery improves an interpretable choice model. The conditions vary one source of misspecification at a time and then combine nonlinear utility, a price threshold, an omitted interaction, and unobserved price sensitivity. Every replication uses a respondent-level split, so tasks from the same decision maker remain on one side of validation.

## Current benchmark pattern

The corrected 30-replication benchmark shows the clearest gain in the combined condition. Mean test accuracy is 0.587 for additive MNL, 0.649 for ML-assisted specification, and 0.650 for the fully structured MNL. Mean decision regret is 0.210 for additive MNL and 0.016 for both structured models. The result links the predictive gain to recovery of the simulated utility terms; it does not support a general claim that a flexible learner dominates behavioural models.

The heterogeneity condition provides the boundary case. The data contain respondent-specific random price sensitivity that is unavailable to the term-discovery stage. The three interpretable specifications have similar predictive performance in this condition. The result separates recoverable utility misspecification from preference variation that requires a heterogeneity model.

## Model-family extension

The two-class Latent Class MNL extension addresses discrete segmentation. In the combined condition its 10-replication mean accuracy is 0.582 with mean decision regret of 0.228. The ML-assisted specification reaches 0.651 accuracy with 0.017 regret in the same extension run. The class model does not recover the continuous price variation in the heterogeneity condition. This comparison motivates Mixed Logit as the remaining behavioural benchmark for continuous heterogeneity.

Random Forest and HistGradientBoosting provide a flexible-learner implementation check under the same grouped holdout. HistGradientBoosting is reported as a boosted-tree proxy because the current macOS environment lacks `libomp.dylib` for the XGBoost native library. These outputs remain supplementary until the exact XGBoost runtime, tuning budget, calibration procedure, and Mixed Logit comparison are fixed.

The targeted Mixed Logit extension estimates a single random price coefficient. Across three replications, the estimated random-price standard deviation is about 0.36 in the heterogeneity condition, with mean decision regret of 0.041. In the combined condition, regret remains 0.236. This separates continuous taste variation from the nonlinear and interaction terms that drive the combined condition. The extension is a diagnostic check; the final paper requires a full random-coefficient specification and WTP recovery.

## Reporting boundary

These results establish the mechanism pattern for the current scaffold. The manuscript should promote them to the main evidence section only after the replication count is expanded, coefficient and WTP recovery are added, and the locked workflow is run on LPMC and Swissmetro. The final discussion should preserve the boundary result: term discovery can expose observable utility structure, while continuous latent heterogeneity requires a model that represents that source of variation.
