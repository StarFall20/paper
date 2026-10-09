# Initial manuscript review

The supplied manuscript is titled “When Does XGBoost Improve Choice Prediction for Smart Medical Devices? A Controlled Simulation Benchmark Against Multinomial Logit”. It currently reports a synthetic benchmark with 400 consumers, 12 tasks per consumer, five replications per lambda level, and three model families.

## Priority revisions for the Journal of Choice Modelling version

1. Reframe the contribution around ML-assisted utility specification. The current framing is a direct predictive comparison between MNL, Random Forest, and XGBoost.
2. Expand the Monte Carlo design to at least 100 replications per condition and separate nonlinear utility, thresholds, omitted interactions, random taste heterogeneity, and a combined condition.
3. Add behavioural recovery outcomes: coefficient bias, WTP bias, confidence-interval coverage, interaction/threshold recovery, false-positive terms, stability, and parsimony.
4. Add Mixed Logit and Latent Class MNL so flexible prediction is compared with behavioural models that address heterogeneity.
5. Replace the single respondent-level 80/20 split as the main validation result with grouped nested validation. Add temporal external validation for LPMC and grouped panel validation for Swissmetro.
6. Add LPMC and Swissmetro as empirical demonstrations. Keep the present synthetic medical-device setting as the controlled simulation and retain the pilot only as a small-sample stress test if the data are available.
7. Treat the current lambda crossover and SHAP ranking as generator-specific evidence. They should not be presented as transferable thresholds or population-level attribute importance.
8. Move implementation settings and fixed seeds to an appendix or reproducibility supplement after the main validation logic is established.

## Repository status

The supplied public repository `StarFall20/paper` exists but is empty (`size: 0`, no files or default branch content). No code can be audited or extended from it yet. The local reproducibility scaffold in the parent workspace is ready to receive the first commit once code is added.
