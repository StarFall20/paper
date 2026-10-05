# ML-assisted utility specification for choice modelling

This repository is organised around the revised Journal of Choice Modelling study.
The empirical design combines a large revealed-preference panel (LPMC), the Swissmetro stated-preference benchmark, and a small-sample stress test. The central method is an ML-assisted specification workflow: machine learning proposes candidate nonlinearities and interactions, while the final model remains a behavioural random-utility model.

## Planned structure

```text
data/              Raw and processed data (raw files are not committed)
src/               Data preparation, estimation, diagnostics, validation
simulation/        Monte Carlo data-generating processes and summaries
analysis/           Reproducible analysis entry points
manuscript/         Main text, appendix, tables, and figures
results/             Generated outputs (tables and figures)
```

## Reproducibility principles

- All splits are grouped by decision maker. The LPMC temporal validation uses the final observed year as an external holdout.
- Simulation conditions separate nonlinear utility, threshold effects, omitted interactions, random taste heterogeneity, and combined mechanisms.
- Every model is assessed with behavioural recovery metrics and predictive metrics.
- Candidate terms selected by ML are refit in interpretable utility models and evaluated with nested validation.
- Random seeds, software versions, data provenance, and estimation settings are recorded for every run.

## Data access

The repository will contain scripts and metadata for downloading or locating LPMC and Swissmetro. Raw data remain excluded from version control unless their licence permits redistribution.
Data provenance and the empirical validation protocol are recorded in `data/README.md`.

## Current status

The first reproducible Monte Carlo run is complete. It uses eight mechanism-specific conditions, 24 replications per condition (192 replications total), 400 consumers, and 12 tasks per consumer. The current minimal implementation compares additive MNL with structured MNL and reports accuracy, log loss, Brier score, choice-share RMSE, and decision regret. The results are in `results/simulation_results.csv` and `results/summary_results.csv`.

This run is a validated analysis scaffold. A targeted random-price Mixed Logit extension is now available as a heterogeneity check. The full random-coefficient specification, exact XGBoost runtime, LPMC, and Swissmetro remain to be added when their estimation dependencies and data files are available.

The corrected assisted-specification experiment is expanded to 30 replications per condition. It includes an explicit opt-out utility indicator and unobserved random price sensitivity in the heterogeneity condition. Nested respondent-level forward selection over six candidate utility terms is compared with additive and fully structured MNL. Results are in `results/assisted_spec_results_corrected_30rep.csv`, `results/assisted_spec_summary_corrected_30rep.csv`, and `results/selected_terms_frequency_corrected_30rep.csv`.

A pure-NumPy two-class Latent Class MNL extension is included in `analysis/run_simulation.py`. Its 10-replication extension results are in `results/latent_class_extension_10rep.csv` and `results/latent_class_extension_summary_10rep.csv`. The latent-class run is an extension check; the 30-replication corrected benchmark remains the locked primary simulation until the remaining model families are added.

Random Forest and HistGradientBoosting are evaluated in `analysis/run_ml_extension.py` with the same grouped holdout. HistGradientBoosting is labelled as a boosted-tree proxy because XGBoost cannot load its native macOS library without `libomp.dylib`. The five-replication tree extension is in `results/tree_extension_5rep.csv` and `results/tree_extension_summary_5rep.csv`.

Run `analysis/smoke_test.sh` after installing `requirements-analysis.txt` to verify the simulation, tree, and targeted Mixed Logit entry points. The frozen summary outputs are checksummed in `results/MANIFEST.sha256`.

The current framework and model audit is in `manuscript/model_audit.md`. Draw-count sensitivity for the targeted Mixed Logit check is in `results/mixed_logit_draw_stability.csv`.
