# ML-assisted utility specification for choice modelling

This repository contains the reproducible materials for the revised Journal of Choice Modelling study. The central method uses ML diagnostics to propose candidate nonlinearities and interactions, then refits those candidates as interpretable random-utility models.

## Structure

```text
data/              Raw and processed data (raw files are not committed)
simulation/        Monte Carlo data-generating processes
analysis/           Reproducible analysis entry points
manuscript/         Revision plans, methods notes, and submission checklist
results/            Frozen summary results and selected-term frequencies
```

## Current benchmark

The corrected benchmark uses eight mechanism-specific conditions, 30 replications per condition, 400 consumers, and 12 tasks per consumer. It includes an explicit opt-out utility indicator and unobserved respondent-specific random price sensitivity in the heterogeneity condition. The assisted specification uses respondent-level nested forward selection over six candidate utility terms.

The corrected results are in `results/assisted_spec_results_corrected_30rep.csv`, `results/assisted_spec_summary_corrected_30rep.csv`, and `results/selected_terms_frequency_corrected_30rep.csv`. The model reports accuracy, log loss, Brier score, choice-share RMSE, and decision regret.

## Reproducibility plan

All splits are grouped by decision maker. The planned LPMC analysis uses a temporal holdout; Swissmetro uses respondent-grouped panel validation. Raw data remain excluded unless their licences permit redistribution. Mixed Logit, Latent Class MNL, XGBoost, LPMC, and Swissmetro are the next extensions in the locked submission plan.
