# ML-assisted utility specification for choice modelling

This repository is organised around the revised Journal of Choice Modelling study.
The empirical design combines a large revealed-preference panel (LPMC), the Swissmetro stated-preference benchmark, and a small-sample stress test. The central method is a mechanism-controlled validation benchmark: a Random Forest diagnostic ranks candidate nonlinearities and interactions, while the final model remains a behavioural random-utility model.

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
- Candidate terms ranked by the Random Forest diagnostic are refit in interpretable utility models and evaluated with nested validation.
- Random seeds, software versions, data provenance, and estimation settings are recorded for every run.

## Data access

The repository will contain scripts and metadata for downloading or locating LPMC and Swissmetro. Raw data remain excluded from version control unless their licence permits redistribution.
Data provenance and the empirical validation protocol are recorded in `data/README.md`.

## Current status

The first reproducible Monte Carlo run is complete. It uses eight mechanism-specific conditions, 24 replications per condition (192 replications total), 400 consumers, and 12 tasks per consumer. The current minimal implementation compares additive MNL with structured MNL and reports accuracy, log loss, Brier score, choice-share RMSE, and decision regret. The results are in `results/simulation_results.csv` and `results/summary_results.csv`.

This run is a validated analysis scaffold. A targeted random-price Mixed Logit extension is now available as a heterogeneity check. The full random-coefficient specification, exact XGBoost runtime, LPMC, and Swissmetro remain to be added when their estimation dependencies and data files are available.

The corrected assisted-specification experiment is expanded to 30 replications per condition. It includes an explicit opt-out utility indicator and unobserved random price sensitivity in the heterogeneity condition. Random Forest ranking followed by nested respondent-level behavioural refitting over a five-term shortlist from six candidate utility terms is compared with additive and fully structured MNL. Results are in `results/assisted_spec_results_corrected_30rep.csv`, `results/assisted_spec_summary_corrected_30rep.csv`, `results/selected_terms_frequency_corrected_30rep.csv`, and `results/assisted_spec_recovery_corrected_30rep.csv`.

The data generator restores the opt-out row after applying the product-price floor, so opt-out carries no product attributes. All downstream extension results were regenerated after this audit fix.

A pure-NumPy two-class Latent Class MNL extension is included in `analysis/run_simulation.py`. Its 10-replication extension results are in `results/latent_class_extension_10rep.csv` and `results/latent_class_extension_summary_10rep.csv`. The latent-class run is an extension check; the 30-replication corrected benchmark remains the locked primary simulation until the remaining model families are added.

Random Forest and HistGradientBoosting are evaluated in `analysis/run_ml_extension.py` with the same grouped holdout. HistGradientBoosting is labelled as a boosted-tree proxy because XGBoost cannot load its native macOS library without `libomp.dylib`. The five-replication tree extension is in `results/tree_extension_5rep.csv` and `results/tree_extension_summary_5rep.csv`.

Run `analysis/smoke_test.sh` after installing `requirements-analysis.txt` to verify the simulation, tree, and targeted Mixed Logit entry points. The frozen summary outputs are checksummed in `results/MANIFEST.sha256`.

The current framework and model audit is in `manuscript/model_audit.md`. Draw-count sensitivity for the targeted Mixed Logit check is in `results/mixed_logit_draw_stability.csv`. The novelty and journal-fit audit is in `manuscript/novelty_positioning.md`. The deeper innovation gate and ranked contribution options are in `manuscript/innovation_gate.md`. An exploratory respondent-score heterogeneity gate is implemented in `analysis/heterogeneity_gate.py` with diagnostics in `results/heterogeneity_gate_30rep.csv` and `results/heterogeneity_gate_summary_30rep.csv`.

The process-aware innovation audit is documented in `manuscript/process_gate_note.md`. Its separate prototype (`analysis/process_gate.py`) tests attribute non-attendance and sequence dependence and includes an unresolved branch when respondent-level score dispersion cannot distinguish random taste sensitivity from non-attendance. The exploratory outputs are in `results/process_gate_30rep.csv` and `results/process_gate_summary_30rep.csv`; they are not calibrated manuscript thresholds.

The independent innovation search is documented in `manuscript/independent_idea_search.md`. The current high-risk pivot is a model-equivalent choice-pair test: paired tasks hold candidate utility differences fixed while changing the attribute decomposition, so a systematic choice-share difference is a direct specification failure. Its 500-replication proof-of-concept is in `analysis/equivalent_pair_test.py` and `results/equivalent_pair_test.csv`. A direct audit of the public Swissmetro file found too few clean repeated profiles for this test, so the pivot requires a paired-task supplement or a different dataset; it is not an empirical submission claim on the current data.

The exact randomized three-alternative benchmark is in `analysis/exact_paired_task_benchmark.py` and `results/exact_paired_task_benchmark.csv`. In the current 50-replication boundary run, the additive candidate has rejection rates 0.06 under the null, 0.84 against the nonlinear condition, and 1.00 against threshold and interaction conditions. Adding the corresponding term reduces rejection to 0.04, 0.02, and 0.02. Interaction localization remains incomplete because the joint shift changes multiple raw loci. A four-cell probability-scale repair was audited and failed under a combined interaction plus nonlinear main effect; it is excluded from the innovation claim. See `manuscript/exact_paired_task_note.md` and `manuscript/factorial_contrast_note.md`.

The sharper candidate-preserving randomization test is in `analysis/randomization_paired_task_test.py` and `results/randomization_paired_task_test.csv`. It uses held-out tasks with identical candidate utility differences and a respondent-cluster sign-flip reference. The 50-replication run gives 0.06 null rejection, 0.82/1.00/1.00 rejection against nonlinear/threshold/interaction conditions, 0.08/0.02/0.04 after repair terms, and retains 0.74/1.00 power when nonlinear or interaction terms are combined with random cost sensitivity. A cubic out-of-library condition is rejected at 1.00. The frozen CSV now includes all-pair and predeclared shift-specific rows. Its implementation-level novelty remains high-risk because observable invariance theory and pair-based MNL tests already exist; see `manuscript/randomization_paired_task_note.md`.

The pre-outcome transformation design audit is in `analysis/metamorphic_design_score.py` and `analysis/metamorphic_design_benchmark.py`. The constrained maximin shifts `(5,45), (20,0), (35,45)` raise the weakest normalized probe score from 1.142 for the fixed design to 1.632. In the 50-replication benchmark, null rejection remains 0.06 while nonlinear power rises from 0.82 to 1.00 and nonlinear-plus-random-cost power rises from 0.74 to 1.00. An unconstrained search repeats `(40,60)` three times and is retained as a failure boundary. Outputs are `results/metamorphic_design_score.csv` and `results/metamorphic_design_benchmark.csv`.

The leading independent innovation candidate is the Behavioral Metamorphic Specification Test (BMST), whose formal utility-fibre relation is the Utility-Fibre Invariance Test (UFIT). Held-out tasks keep candidate utility differences fixed while changing their raw attribute decomposition, and a respondent-cluster randomization test measures cross-task exchangeability. Round 4 narrows the claim to a candidate-basis sufficiency audit and requires a formal proposition plus a valid paired-task instrument before submission-level novelty is claimed. The Mechanism-Fingerprint Choice Audit in `analysis/mechanism_fingerprint_benchmark.py` is a bounded localization extension. The simulation output is `results/mechanism_fingerprint_benchmark.csv`; the overlap boundary and decision are documented in `manuscript/innovation_gate_round3.md` and `manuscript/innovation_gate_round4.md`.

The BMST negative-control benchmark is in `analysis/bmst_negative_controls.py` and `results/bmst_negative_controls.csv`. Across 100 replications, rejection is 0.04 under the additive null, 0.68 for omitted nonlinearity, 0.03 for random linear taste, 0.23 for task-specific scale drift, and 1.00 for an order effect. These results define a relation-violation boundary; they do not identify the mechanism from a p-value.

The required instrument for a defensible empirical test is specified in `manuscript/paired_task_supplement_design.md`. It defines the common-shift construction, respondent-fold estimation, cluster randomization reference, counterbalanced order, and the opt-out boundary.

The weaker cross-fitted observational fallback is implemented in `analysis/observational_equivalence_test.py`. Its boundary simulation keeps null size near 5% but gives modest power against deliberately visible nonlinear and interaction alternatives. The public Swissmetro run produces 126, 236, 1,371, and 3,430 approximate pairs at tolerances 0.01, 0.02, 0.05, and 0.10. Its p-values change from about 0.43–0.48 under row-order orientation to 0.003–0.033 at the larger tolerances under time orientation, while cost orientation gives 0.847–0.977. This orientation sensitivity is a failure boundary, not evidence of misspecification. See `manuscript/observational_equivalence_note.md`.

The current distance to a defensible submission claim is recorded in `manuscript/submission_gap_audit.md`.
