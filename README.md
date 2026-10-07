# Fibre-conditional sufficiency audits for choice modelling

This repository is organised around the revised Journal of Choice Modelling study.
The current contribution is a fibre-conditional sufficiency audit for a
coarsened candidate menu summary \(\phi\). The null is
\(Y\perp X\mid\phi(X)\); the conditional response function is left
unrestricted. Feasible reflected DCE tasks preserve the complete candidate
menu vector and randomize raw profiles within each fibre. A finite exposure
matrix certifies which residual directions are structurally visible. The
nuisance-orthogonal fibre quotient calculation is an extension for declared
scale and framing tangents. Earlier BMST/UFIT labels are historical names and
are not separate contributions.

The conceptual core is in
`manuscript/fibre_sufficiency_exposure_round22.md` and its proof note. The
multinomial full-menu gate is implemented in
`analysis/no_fqc_multinomial_check.py`; the finite support counterexample is in
`analysis/fibre_exposure_counterexample.py`. The current status and remaining
empirical gates are recorded in `manuscript/progress_status.md`.

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

The earlier BMST/UFIT experiments are retained as development history. Their
current interpretation is the fibre-conditional sufficiency audit described at
the top of this README. The Mechanism-Fingerprint Choice Audit in
`analysis/mechanism_fingerprint_benchmark.py` is a bounded localization
extension; it does not carry a separate innovation claim.

The BMST negative-control benchmark is in `analysis/bmst_negative_controls.py` and `results/bmst_negative_controls.csv`. Across 100 replications, rejection is 0.04 under the additive null, 0.68 for omitted nonlinearity, 0.03 for random linear taste, 0.23 for task-specific scale drift, and 1.00 for an order effect. These results define a relation-violation boundary; they do not identify the mechanism from a p-value.

The preferred one-member-per-pair supplement is benchmarked in `analysis/bmst_assignment_benchmark.py` and `results/bmst_assignment_benchmark.csv`. With 600 respondents and 100 replications, rejection is 0.05 under the additive null, 0.79 for omitted nonlinearity, 0.05 for random linear taste, and 1.00 for an omitted interaction. The design removes carryover from the primary estimand.

The required instrument for a defensible empirical test is specified in `manuscript/paired_task_supplement_design.md`. It defines the common-shift construction, respondent-fold estimation, cluster randomization reference, counterbalanced order, and the opt-out boundary.

The public-data audit in `manuscript/public_data_pair_audit.md` records the search for an existing empirical BMST instrument. The open TUDelft computer-vision choice dataset has repeated observations, but its documented schema and random task construction do not establish candidate-preserving pair assignment. It is therefore reserved for descriptive or predictive transfer; a purpose-built paired-task supplement remains required for the primary empirical claim.

The weaker cross-fitted observational fallback is implemented in `analysis/observational_equivalence_test.py`. Its boundary simulation keeps null size near 5% but gives modest power against deliberately visible nonlinear and interaction alternatives. The public Swissmetro run produces 126, 236, 1,371, and 3,430 approximate pairs at tolerances 0.01, 0.02, 0.05, and 0.10. Its p-values change from about 0.43–0.48 under row-order orientation to 0.003–0.033 at the larger tolerances under time orientation, while cost orientation gives 0.847–0.977. This orientation sensitivity is a failure boundary, not evidence of misspecification. See `manuscript/observational_equivalence_note.md`.

The current distance to a defensible submission claim is recorded in `manuscript/submission_gap_audit.md`.

The fifth-round cross-literature novelty boundary is recorded in `manuscript/novelty_boundary_round5.md`. It incorporates Daly's utility-difference and scale-identification warning and limits BMST to a declared choice-probability relation on a tested task fibre.

The sixth-round independent innovation audit is in
`manuscript/innovation_search_round6.md`. It records the development path
that led to the current sufficiency formulation and demotes the original
XGBoost-versus-MNL crossover to a controlled benchmark.

The seventh-round audit in `manuscript/innovation_audit_round7.md` addresses
the direct comparison-complexity objection. The primary test is restricted to
nontrivial fibres, preserves the full menu difference vector, and compares an
observational LR test and residual learner with the designed fibre probe. The
anchored benchmark is in `analysis/anchored_fibre_benchmark.py` and
`results/anchored_fibre_benchmark.csv`; the sample-size planning run is in
`analysis/bmst_power_curve.py` and `results/bmst_power_curve.csv`.

The eighth-round innovation search asks whether the scale, geometry, framing,
and order controls can be made one pre-outcome criterion. The leading upgrade
is nuisance-orthogonal utility-fibre testing: task-pair contrasts are selected
after projecting the omitted-utility signal away from a declared nuisance
tangent space. The proof-of-concept is in
`analysis/nuisance_orthogonal_fibre_benchmark.py`, with the method note in
`manuscript/nuisance_orthogonal_fibre_note.md` and the search audit in
`manuscript/innovation_search_round8.md`. It is a planning result until the
nuisance library, out-of-library controls, and paired-task instrument are
completed.

The ninth-round audit introduces a sharper instrument: parity-separated
utility-fibre testing. A candidate-preserving trajectory uses reflected
coordinates (z_A(t)=c+t) and (z_B(t)=c-t). Symmetric geometry and scale
changes are even in (t); a quadratic omitted decomposition term is odd. The
central (+t/-t) contrast therefore cancels the declared even processing
mechanism before inference. The binary benchmark is in
`analysis/parity_fibre_benchmark.py`; the full-menu multinomial check is in
`analysis/parity_multinomial_check.py`; the formal result is in
`manuscript/parity_fibre_proposition.md` and the overlap audit is in
`manuscript/innovation_search_round9.md`. This is now the preferred primary
instrument when the substantive fibre admits the reflection symmetry; the
round-eight nuisance-orthogonal design remains the general extension.

The parity proposition has also been generalized to the moderate-utility form
(F(Delta V / D)), so the odd/even construction is not tied to multinomial
logit. The logistic/probit link check is in
`analysis/parity_link_robustness.py` and `results/parity_link_robustness.csv`.

The round-ten collision audit is in `manuscript/innovation_search_round10.md`.
It positions the parity test against moderate-utility theory, tradeoff
complexity, choice-set specification tests, symmetric DCE designs, and
procedural-invariance studies, and records the exact remaining novelty and
empirical gates.

The application-specific smart-device parity audit is in
`manuscript/smart_device_parity_design.md`, with exact profile equalities and an
A/B/opt-out planning benchmark in `analysis/smart_device_fibre_design.py` and
`results/smart_device_fibre_benchmark.csv`.

The five-task smart-device candidate grid is in
`analysis/smart_device_fibre_candidate_grid.py` and
`results/smart_device_fibre_candidates.csv`; it enumerates 677 feasible
reflections and selects gaps from 0.20 to 1.00.

A fibre-optimal smart-device task rule is implemented in
`analysis/smart_device_fibre_optimal_design.py` and
`results/smart_device_fibre_optimal_design.csv`. It ranks 648 pure-odd
candidate reflections by pre-outcome local odd information.

The calibration-error gate is audited in
`analysis/smart_device_calibration_robustness.py` and
`results/smart_device_calibration_robustness.csv`; it motivates an independent
pilot and a predeclared candidate-gap tolerance for empirical deployment.

The eleventh-round collision audit is in
`manuscript/innovation_search_round11.md`. It incorporates the 2024 latent
utility and permutation invariance paper and narrows the claim away from a
general transformation-group theorem.

The round-twelve JOCM positioning audit is in
`manuscript/jocm_positioning_round12.md`. It aligns the parity-fibre design
with recent JOCM work on Bayesian design, ordering effects, attribute
attendance, interpretable ML, and model-building workflows.

The matched ordinary-grid versus fibre-optimal five-task benchmark is in
`analysis/smart_device_multitask_benchmark.py` and
`results/smart_device_multitask_benchmark.csv`; at weak signal strength .3,
power is .49 versus .87 at 150 respondents; the combined omitted-plus-even-
scale rates are .335 versus .785. The corrected scripts use one orientation
coin per respondent block and respondent-cluster sign flips.

The smart-device preregistration protocol is in
`manuscript/smart_device_preregistration_spec.md`; invariant checks are in
`analysis/verify_smart_device_preregistration.py`.

The manuscript-ready innovation section is in
`manuscript/innovation_section_draft.md` and is written around the candidate
fibre, parity estimand, information criterion, smart-device implementation, and
explicit failure boundaries.

The round-thirteen manuscript revision package is in
`manuscript/revision_package_round13.md`; it provides the title, abstract,
section map, result hierarchy, and demotion rules for the original ML-first
narrative.

The round-fourteen anti-stitch audit is in
`manuscript/novelty_audit_round14.md`. It narrows the contribution to one
nuisance-balanced candidate-fibre design and records the collision audit with
Healy and Leo (2026), Shubatt and Yang, McGranaghan et al. (2024), and latent
permutation-invariance work. The corrected design is implemented in
`analysis/smart_device_nuisance_balanced_design.py` and its five-task output is
`results/smart_device_nuisance_balanced_design.csv`. The comparison-complexity
negative-control benchmark is in
`analysis/smart_device_complexity_balance_binary_benchmark.py` and
`results/smart_device_complexity_balance_binary_benchmark.csv`; at 600
respondents it rejects .185 of loose-grid complexity-only samples and .050 of
balanced-grid samples, while omitted-direction power is 1.000 for both.

The fibre-violation profile is implemented in
`analysis/smart_device_fibre_violation_profile.py` and
`results/smart_device_fibre_violation_profile.csv`. It adds a max-T profile to
the same balanced task block. Under a localized omitted interaction, global
power is .330, profile power is .885, and localization accuracy is .990; null
rejection is .035 and .050. The random-taste boundary benchmark is in
`analysis/smart_device_random_taste_balance_benchmark.py`; rejection is .085,
.095, and .085 at coefficient standard deviations 0, .1, and .2.

The round-sixteen independence and assignment audit is in
`manuscript/novelty_audit_round16.md`. It records which literature ingredients
are used as bounded inputs, which claims are excluded, and why the corrected
respondent-level assignment is part of the estimand.

The round-eighteen JOCM topic audit is in
`manuscript/jocm_topic_gap_audit_round18.md`, and the structural-fibre design
and simulation record is in `manuscript/falsification_design_round18.md`.
The round-nineteen overlap audit is in
`manuscript/literature_overlap_audit_round19.md`; it records the material
collision with the 2013 maximin model-robust DCE abstract and narrows the
claim accordingly. The independent theoretical core is the fibre quotient
completeness operator in `manuscript/fibre_quotient_completeness_round19.md`.
The writing-ready replacement for the earlier innovation section is
`manuscript/innovation_section_round19.md`.
The reproducible structural search is
`analysis/structural_fibre_design.py`, with task, calibration, and power
outputs under `results/structural_fibre_*.csv`.
