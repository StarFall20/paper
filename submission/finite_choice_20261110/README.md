# Finite solutions in prediction assisted choice logit

## Current manuscript and verification

The 14 November 2026 held-out-record and literature integration is current. The full Word is 61 pages with 1186 native mathematical objects, 131 display paragraphs and 13 tables; the main reading Word is 29 pages. Section 4.4 and Appendix C6 add prediction checks on 3405 released human choice records. Table C1 retains all seven estimators, both prediction routines and both label budgets. The preceding 12 tables and every display body are preserved. All 83 analysis files parse; 80 of the preceding 81 are byte-identical, and quick_check.py now reconciles the added evidence. There are no new human observations or AI answers.

The [current source/layout pointer](reports/current_manuscript.json), [integration record](reports/holdout_manuscript_integration_20261114.json), [complete visual review](reports/holdout_manuscript_visual_qa_20261114.json) and [mathematical check](reports/holdout_integration_mathematical_check_20261114.json) identify the current artifacts. The [isolated validation](reports/holdout_integration_clean_validation_20261114.json) records the independent extraction, all 83 matching scientific sources and all 13 table checks. Earlier dated reports describe their historical stages. The [institutional literature audit](reports/institutional_literature_audit_20261113/README.md) retains 9 actual WOS searches/chains, 537 exports and 512 unique indexed records. The [supplementary literature record](results/holdout/literature_followup.md) records four actual Google Scholar queries, bounded title screening and selected primary readings. Scopus returned Preview after the authorized institutional exchange; no Scopus literature query ran. PPI++ formal-version full-text and supplement reconciliation remains incomplete. These records do not prove originality.

Separate publication figures are provided as [font-embedded vector PDFs](manuscript/figures/publication_pdf/). Regenerate them with `python authoring/export_publication_figures.py`; no model draws or fits run. The PNGs embedded in Word remain reading assets. An AI-assistance declaration is included before references. Final author review and publisher portal declarations remain pending. This package is not an acceptance certificate.


This archive accompanies **When prediction assisted logit estimates exist**. It contains one compensated choice objective, its completed results and the code needed to regenerate them. It excludes the historical research notebook, raw questionnaires, human participant response vectors and downloaded numeric pickle inputs. Complete generated menu records are included for numerical verification. The separate questionnaire application with 300 records supplies no observations to this paper.

The stable package directory name is retained. Current empirical results distinguish finite estimation, held-out probability quality and known-truth error/coverage. At 50 labels, human-ridge-initialized power tuning is finite in all 500 allocations but has mean AI-aided held-out log loss 0.896512, versus direct human ridge 0.762353. This is not evidence for a superior new estimator. The [holdout supplement](results/holdout/README.md) retains all 14000 records, eight unavailable fits, coefficients, plans and independent metric checks. Respondent IDs are unavailable; the fixed-row evaluation does not certify independent people or population transport.

The binary results concern independent Gaussian remaining attributes,an estimated intercept,one utility signal,a frozen predictor and a finite auxiliary ratio. The menu result uses alternative-specific Gaussian profiles,common slopes and fitted alternative constants. Theorem3 allows simplex-face predictions with positive mean squared error;AppendixF treats perfect hard labels by an exact weighted likelihood reduction. The positive power upper envelope also bounds every power simultaneously. The binary strict-gap condition and the larger-menu condition differ. Clustered respondent choices and arbitrary categorical designs are not covered by this Gaussian phase law. The finite-design support LP,projection routines and symmetric zonotope special case are classical. Numerical agreement does not establish priority.

## Start here

- [Complete manuscript and proofs](manuscript/paper.md).
- [Editable Word manuscript](manuscript/paper.docx), with native Word equations.
- [Table 1 boundary values](manuscript/table1.csv), [Table 2 estimation comparisons](manuscript/table2.csv), [Table 3 fixed power replay](manuscript/table3.csv) and [Table 4 training-cost comparison](manuscript/table4.csv).
- [Menu endpoint table F1](manuscript/tableF1.csv),[matched menu table F2](manuscript/tableF2.csv) and [complete menu records](results/menu/hard_menu_matched_phase_trials.csv).
- [Source attribution](ATTRIBUTION.md) and [data download manifest](reports/public_conjoint_audit_20261109/source_manifest.json).
- [Full aggregate simulation grid](results/finite_pool_summary.csv), [imperfect predictions](results/imperfect_pool_summary.csv) and [stronger utility](results/strong_signal_pool_summary.csv).

Run commands from this directory with Python 3.12. The exact recorded package versions are in [requirements.txt](requirements.txt); create your own environment and install them before running. Set the linear algebra thread limits for the recorded convention:

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python analysis/quick_check.py --out regenerated/checks --full-math
python analysis/audit_binary_choice.py examples/intercept.csv --features intercept --out regenerated/example.json
python analysis/build_publication_assets.py --results results --manuscript manuscript
```

The quick check requires no external data. It checks a closed form logistic root, boundary/exterior/rank cases, alternative moment and interval LPs, normal integration, proper augmentation weight invariance and intercept covariance formulas. It reconciles all thirteen manuscript tables, the training-cost status counts and the completed aggregate grids. It preserves the single original unknown LP status and checks its independent positive-gap resolution. `--full-math` adds signal, imperfect prediction and positive power numerical checks. It does not rerun the full Monte Carlo grids.

Optional Word authoring uses the packages in `authoring/requirements.txt`:

```sh
python authoring/build_manuscript_docx.py --source manuscript/paper.md --out manuscript/paper.docx --report regenerated/docx_structure.json
```

The strict TeX subset converter emits native OMML equations. The full Word has 61 pages, 1186 native mathematical objects, thirteen tables, six main figures and seven appendices. The 29-page [main reading Word](manuscript/main_text.docx) contains the main text, data availability and references, with six tables and 238 native mathematical objects. The current records linked above identify exact sources and every reviewed page. Fraction arguments must be braced in this strict subset. Source changes require a fresh visual review. Rebuilding a DOCX can change ZIP metadata even when text and equations agree.

The audit CSV has `choice`, `prediction`, `labelled` and explicitly named feature columns. Labelled is zero or one. Auxiliary choices may be blank. Predictions lie in the unit interval. Include an explicit intercept column if wanted. The output distinguishes support status from fit status and reports the selected power. It changes no target and issues no confidence interval.

## Fetch and reproduce public allocations

```sh
python analysis/fetch_public_conjoint_audit.py --out local_inputs
python analysis/public_conjoint_optimizer_audit.py --input-dir local_inputs --out regenerated/public
python analysis/public_conjoint_matched_audit.py --input-dir local_inputs --out regenerated/public --allocations 500
python analysis/power_tuned_choice_audit.py --input-dir local_inputs --out regenerated/public --allocations 500
python analysis/verify_analysis_decision.py --input-dir local_inputs --out regenerated/public
```

The fetcher downloads four pinned files, retains their original license/README and verifies each size and Git blob hash. The numeric loader verifies the same hashes and restricts permitted pickle globals. Downloaded research scripts are never executed. Public data remain under `local_inputs`; do not add them to a public backup automatically.

The two matched reanalyses use different stated objectives. The pooled imputation unit correction has five AI and two attributes only exterior cases; auxiliary only PPI has six and two. `verify_analysis_decision.py` expects the eight recorded auxiliary only cases produced by the fixed 500 allocation experiment. Changing that experiment requires a new plan and verification; the script deliberately rejects unrecognized case counts. Five row source blocks are not verified respondent identifiers. Point targets are converged empirical projections.

## Reproduce the probability boundaries

```sh
python analysis/finite_ratio_choice_boundary.py --out regenerated/theory
python analysis/finite_ratio_signal_boundary.py --out regenerated/theory
python analysis/imperfect_predictor_choice_boundary.py --out regenerated/theory
python analysis/verify_matched_moment_proposition.py --out regenerated/theory
```

The balanced grid has 80 cells with 500 draws per cell; the signal and imperfect prediction grids each have 80 cells with 300 draws per cell. Every curve includes its unresolved count. The fixed dimension variance reference weight is an established comparator, held fixed for these grids. It is distinct from the data selected tuning method in the public and support experiments. Zero power uses ordinary logit separation.

## Reproduce error and coverage

```sh
python analysis/finite_pool_choice_validation.py --input-dir local_inputs --out regenerated/pool --replications 500
python analysis/imperfect_finite_pool_validation.py --input-dir local_inputs --out regenerated/pool --replications 500
python analysis/imperfect_strong_signal_validation.py --input-dir local_inputs --out regenerated/pool --replications 500
python analysis/imperfect_population_targets.py --input-dir local_inputs --out regenerated/pool
python analysis/imperfect_population_targets.py --input-dir local_inputs --out regenerated/pool --truth-scale 3 --prefix strong_signal_pool
python analysis/verify_finite_pool_covariance.py --out regenerated/pool
```

These are generated choices on released attribute support. They do not add respondents or LLM samples. The original oracle grid has 12 conditions; the original signal and threefold signal prediction grids add six conditions each. Every condition has 500 repetitions and seven methods. This gives 84000 method-condition-repetition outcomes across the three grids. All methods share samples within their comparison. Raw generated trial files are recreated locally; this archive contains aggregates, statuses, coordinate coverage, error tails and paired comparisons.

The original completed oracle run predates a one dimensional covariance shape compatibility fix. [The source compatibility record](results/finite_pool_source_compatibility.json) identifies both hashes and explains why the original 11 dimensional calculations are unchanged. Do not resume an old checkpoint with a changed source hash. A clean rerun uses a new output directory. Scripts with `--resume` accept only matching checkpoints. Do not point exploratory shorter runs at the shipped aggregate directory.

MSE and paired loss comparisons condition on available points. Coverage given issuance and issued and covers have different denominators. Eleven coordinate intervals are marginal, not simultaneous. Monte Carlo errors use repetition level fractions, not 5500 independent coordinates. No finite sample interval guarantee follows from the existence theorem. Ridge sandwich intervals do not remove penalty bias.

## Reproduce the positive power endpoint and original draw replay

```sh
python analysis/vanishing_power_choice.py --out regenerated/power --replications 500
python analysis/verify_vanishing_power_phase.py --out regenerated/power
python analysis/verify_proper_augmentation_invariance.py --out regenerated/power
python analysis/replay_choice_power_path.py --input-dir local_inputs --reference regenerated/pool/finite_pool_trial_diagnostics.csv --out regenerated/power
python analysis/summarize_replayed_power_pairs.py --out regenerated/power
python analysis/write_vanishing_power_report.py --results regenerated/power --manuscript manuscript
```

Run the original 500 repetition finite pool experiment first: its locally regenerated trial file supplies the replay reference. The replay reconstructs the exact 500 draws for 20 labels, 600 auxiliary rows and proxy error 0.25. All 1000 original human/unit support statuses agree. It adds no independent response sample. The separate Gaussian phase experiment has 12 conditions and 500 paired draws each; every draw compares zero, four positive powers and limiting positive feasibility. The completed source hashes are checked without altering those sources after execution.

The raw phase table retains one unknown solver result. The verifier reconstructs its draw and records an independently verified exterior direction in `vanishing_power_resolution.json`; `vanishing_power_phase_verified.csv` is separate from the raw table. Keep both. A new solver may resolve the original draw directly, so regenerated unresolved counts can differ without changing the mathematical event.

Table 3 retains all availability counts. Figure 5 compares errors on the same 412 available unit power fits; the complete paired table and human separation strata accompany it. The original positive power right limit takes the proportional sample limit first and the power limit second. Its recorded derivation and status flags predate the stronger balanced joint result below; they are retained unchanged. No new optimal inference weight or confidence interval guarantee follows. The [original derivation](reports/public_conjoint_audit_20261109/vanishing_power_derivation_20261110.md) and [regularization prior comparison](reports/public_conjoint_audit_20261109/synthetic_regularization_prior_comparison_20261110.md) state that earlier step.

## Reproduce the simultaneous power bound and its direct prior check

```sh
python analysis/joint_power_choice.py --out regenerated/power --reference regenerated/power/vanishing_power_phase_verified.csv
python analysis/write_joint_power_report.py --results regenerated/power
python analysis/verify_zonotope_reduction.py --out regenerated/power
python analysis/summarize_replayed_choice_loss.py --out regenerated/power
```

The joint experiment reconstructs the same 6000 Gaussian samples and evaluates two additional powers, 1/sqrt(m) and 1/m. All 12 original limiting support aggregates match. Its 36 report cells retain 500 draws each; there are no unresolved classifications. The independent geometry check compares 48 limiting-cone projections with a full constrained quadratic program and checks explicit polar inequalities and orthogonality. A nonzero zero-scale face is retained in the closed cone.

The [joint derivation](reports/public_conjoint_audit_20261109/joint_power_derivation_20261110.md) bounds all powers simultaneously and gives a strictly positive,data dependent vanishing-power result for the balanced fixed-zero-intercept model. It extends the earlier iterated endpoint for that model. This was the original balanced stage;the general extension below supersedes that scope restriction. The exact-zero control is an analytic mixture of completed empirical frequencies added after the simulation,not a new random experiment. None of these statements is an error distribution or interval guarantee.

The [direct-prior comparison](reports/public_conjoint_audit_20261109/joint_power_prior_scope_20261110.md) identifies the unit-power constant-one-half case with the established Gaussian zonotope absorption problem. A separate 200-case signed-cube LP verifies that algebra; the special case is not an innovation claim. The choice-loss summary reuses the 500 original support fits. It evaluates known-truth expected Bernoulli KL on the fixed 1200-row support and supplies no independent human predictive validation.

## Reproduce the general simultaneous power boundary

```sh
python analysis/general_joint_power_boundary.py --out regenerated/general
python analysis/verify_general_joint_geometry.py --out regenerated/general
python analysis/verify_general_joint_phase.py --out regenerated/general
```

The same signal/free-intercept/frozen-predictor model now has an all-power upper bound and a strictly positive,data-dependent vanishing-power existence boundary. Its limiting value strictly exceeds ordinary logit for any fixed predictor strictly inside(0,1). This concerns finite-root existence,without coefficient-error or confidence-interval guarantees. A changing trained predictor or hard prediction is outside these conditions. The [proof record](reports/public_conjoint_audit_20261109/general_joint_power_derivation_20261110.md) and [completed result](reports/public_conjoint_audit_20261109/一般模型联合权重上界与进度_20261110.md) state the scope and prior attribution.

Population checks include30conditions,120finite-power comparisons and200/400node quadrature. The initial100/200node failure remains in the archive;its follow-up increased integration order without loosening tolerance. Independent geometry uses120original/rescaled LP matches,24full constrained projections,56original-moment positive-power witnesses and64negative-cost weak separators.

The complete new phase run has6000independent Gaussian draws,180aggregate cells with500repetitions and90000feasibility calls. Original unknown solver statuses are retained in the raw table and reconstructed by the separate verifier. Its certificate record and verified table accompany the unchanged raw table. One positive-power boundary/proximity call remains a separate status;the verifier does not force it into an interior or exterior class. A regenerated solver can resolve those cases directly;the verifier accepts the newly observed unknown count while retaining the full500draw denominator. No new participant or language-model responses are collected.

## Reproduce the established DY-prior comparator

```sh
python analysis/replay_dypl_comparator.py --input-dir local_inputs --references regenerated/pool --out regenerated/DY
python analysis/write_dypl_comparator_report.py --results regenerated/DY
```

Run the original500repetition finite-pool and strong-signal studies first;their locally generated trial files supply both references. This additional comparison replays the same1000original draws for the two Table2conditions,without changing the seven original strategy grids. It adds raw DY pseudo-response fits using alpha=m/(m+11) and alpha=0.5,a zero prior mode and zero initialization. The comparator uses labelled rows alone. All1000human status reconstructions agree and their available MSEs agree to relative tolerance1.8e-16. Both settings retain500fits in each condition,2000fits total.

The ordinary score sandwich is a diagnostic for raw DY estimates. It does not implement the cited papers' high-dimensional Gaussian bias/variance rescaling on this discrete attribute support. Shrinkage bias,population targets,paired MSE/choice-loss differences and repetition-level Monte Carlo precision remain reported. The [primary-source review](reports/public_conjoint_audit_20261109/dypl_noisy_label_prior_audit_20261110.md) and [recorded plan](reports/public_conjoint_audit_20261109/dypl_comparator_plan_20261110.json) distinguish established methods from the candidate compensated-loss result. External R code was inspected at a pinned commit and independently reimplemented from the estimator formula;none is executed by this package.

## Inventory and output map

The manuscript's Section 3.8 and Appendix E retain a fixed number of prediction indices among fitted Gaussian attributes. A conditional corollary covers independently trained linear logistic scores when their signal overlap,orthogonal norm and intercept converge to finite deterministic limits. It does not supply arbitrary cross-fitting or black-box training guarantees. Identical calibrated response/prediction laws can have different existence boundaries when their relationship to fitted attributes changes. The generic alignment principle and Gaussian rotation are established tools;the paper calculates their role in its compensated objective.

```sh
python analysis/attribute_aligned_prediction.py --out regenerated/alignment
python analysis/calibrated_aligned_prediction.py --out regenerated/alignment
python analysis/verify_trained_prediction_rotation.py --out regenerated/alignment
python analysis/aligned_prediction_phase.py --out regenerated/alignment
python analysis/write_aligned_prediction_report.py --results regenerated/alignment
```

These commands require no public input download. The two population grids retain18and12conditions. The finite-size illustration was selected AFTER inspecting the calibrated grid and retains all16cells,500draws per cell and8000support calculations. Its [complete paired table](results/aligned_phase_paired_summary.csv) includes a reversed finite-sample ordering;rows share draws. The manuscript reports it in [TableE1](manuscript/tableE1.csv). The initial rank/feasibility classification assertion and exact original source are retained in the results and report folders. Six rank-deficient diagnostic cases remain separate. The60training fits check only exact rotations,not convergence of training quantities. [Primary-source scope](reports/public_conjoint_audit_20261109/aligned_prediction_prior_scope_20261110.md) states the nearest inspected literature and actual reading units.

The [complete finite-design power-domain supplement](reports/public_conjoint_audit_20261109/完整权重区间审计_20261110.md) replays all 500 original allocations at 50 labels with both original prediction routines. It computes a closed feasible endpoint and a separate strict interior witness. The eight original unit-power exterior cases each have an independently checked recession direction and fixed-power brackets. Existing selected powers remain unchanged and admissible. Exact zero selections are checked as ordinary logit endpoints. These classical parametric LP calculations supplement the finite-design audit and do not change the three main comparison tables.

```sh
python analysis/power_domain_choice.py --out regenerated/domain
python analysis/audit_public_power_domain.py --input-dir local_inputs --reference regenerated/public/power_trial_diagnostics.csv --out regenerated/domain
python analysis/write_power_domain_report.py --results regenerated/domain
```

Run the original public power audit above before this replay. The package supplies [aggregate results](results/power_domain_summary.csv), [eight cutoff cases](results/power_domain_exterior_table.csv), [certificates](results/power_domain_exterior_certificates.json) and [complete-case checks](results/power_domain_replay_checks.json). The 1000-row individual diagnostics are recreated locally. A closed feasible endpoint alone does not prove a finite root; a zero strict-witness optimum can occur even when the closed endpoint is one. No largest-admissible-power estimator, optimal tuning rule or inference guarantee is proposed.

`analysis/` holds the independent estimator, theory, verification and figure code with their local import closure. `results/` holds the completed aggregate files. `manuscript/figures/` holds standard PNG and editable SVG plots. Tables are generated from those same aggregates. `MANIFEST.sha256` covers every shipped file except itself; `verify_package.py` verifies it and all local Markdown links without downloading inputs.

```sh
python verify_package.py
```

The paper and this archive do not assert journal acceptance or an exhaustive Scopus/Scholar search. The research archive retains the closest prior applicability review and the submission assessment separately.

## Independent training label costs

```sh
python analysis/training_budget_diagnostic_20261111.py --out regenerated/training_budget
```

The plan fixes eight conditions,100 draws per condition,six methods and a total label union shared by the direct human baselines. It generates 800 statistical model draws and 4800 retained method rows,without private questionnaire records or LLM answers. Training and direct ridge use a fixed mean-loss penalty of 0.1 on all parameters,including the intercept. Fixed correction powers are diagnostic choices. The complete summaries,pairing counts and all 397 independently certified ordinary-logit separation cases are in `results/training_budget_*`.

The original full simulated trial table and its independent verifier are preserved in the separate training-budget supplement in the repository. This main package ships aggregates and regenerates all trials with the command above. The packaged source only adapts its import path; its full800-design replay reproduces the original trial and aggregate CSVs byte for byte,as recorded in `results/training_budget_source_adaptation.json`. This replay adds no statistical replications.

## Whole menu reproduction

Theorem 3 and Appendix F concern independent Gaussian alternative profiles, common attribute coefficients, fitted alternative constants and a fixed prediction law. The probability vector can lie on simplex faces. Positive mean squared prediction error supplies the scale bound; perfect hard predictions have their separate exact ordinary likelihood reduction. The strict gap for always wrong binary predictions is zero, and the larger-menu gap is strictly positive under the stated assumptions. This result does not validate clustered fixed-task DCE data or provide an accuracy guarantee.

The [complete menu records](results/menu/hard_menu_matched_phase_trials.csv) contain 1000 generated model draws and 9000 classifications, with 18 cells of 500 repetitions. The [saved ordinary separators](results/menu/hard_menu_ordinary_separation_certificates.json) certify 1090 boundary cases. The matched experiment uses the actual existing variance reference of the hard predictor for one orthogonal coefficient; all methods share each menu sample and dimension. It was selected after the population calculation as a mechanism illustration. The [reading scope](reports/menu/reading_scope.json) records the inspected multiPPI++, common coefficient MNL and categorical theory sources.

Verify the retained records in a separate directory:

```sh
python - <<'PY'
import shutil
shutil.copytree('results/menu', 'regenerated/menu_checks', dirs_exist_ok=True)
PY
python analysis/verify_hard_menu_phase.py --out regenerated/menu_checks --plan analysis/hard_menu_phase_plan_20261111.json
python analysis/verify_hard_menu_prediction.py --out regenerated/menu_diagnostics
```

Recompute the population and finite experiment using the fixed plans:

```sh
python analysis/menu_hard_prediction_boundary.py --out regenerated/menu --plan analysis/menu_hard_prediction_plan_20261111.json
python analysis/hard_menu_variance_reference.py --out regenerated/menu --plan analysis/hard_menu_variance_plan_20261111.json
python analysis/hard_menu_matched_phase.py --out regenerated/menu --plan analysis/hard_menu_phase_plan_20261111.json
python analysis/menu_population_grid.py --out regenerated/menu_reduction --plan analysis/menu_population_grid_plan_20261111.json
```

The last command recreates the separately recorded 12 binary reductions as part of the posterior-prediction integration diagnostic. It uses the signal convention stated in Appendix F. The hard-prediction population and phase files are a different predictor rule and remain distinct. Regeneration on another solver version can change diagnostic margins and fallback metadata; the supplied historical checkpoint remains the original reference. No interrupted draw is counted twice. The exact failed sources and case inputs are preserved for audit.

Rebuild the publication figure and tables from completed aggregates:

```sh
python analysis/build_menu_publication_assets.py --results results/menu --manuscript manuscript
python authoring/build_manuscript_docx.py --source manuscript/paper.md --out regenerated/paper.docx --report regenerated/docx_structure.json
```

Figure 3 uses `figure_menu_phase.png` and its editable SVG. The retained legacy assets `figure2_estimates_intervals.png` and `figure3_power_endpoint.png` appear as Figures 4 and 5. Figure4 now includes the documented Firth comparison. Tables F1 and F2 come from the same complete menu CSVs. The menu scientific source hashes remain checked in [source hashes](results/menu/source_hashes.json). The earlier manuscript and numerical failure records retain their dated scope.

## Oracle local score equivalence

Section 3.5 and Appendix B hold the finite sample truth score and the full fixed dimensional asymptotic covariance equal along a prediction and power path. The same corrected objective has different finite solution boundaries. The strict null example uses rational bounds; the nonzero signal illustration uses matched Gaussian draws. The construction uses the true attribute-only probability as a theoretical reference. It introduces no fitted calibration or new weight selection algorithm.

The package retains the [completed population calculation](results/score/population_clean/variance_reference_population_primary.csv), [complete matched records](results/score/phase/variance_reference_phase_trials.csv), [independent verification](results/score/phase/variance_reference_complete_verification.json) and [rational bounds](results/score/certified_bounds/rational_counterexample_certificate.json). Figure 2 uses `identical_score_phase.png` and its editable SVG. Table B1 contains every displayed frequency. The [scientific source closure](results/score/source_hashes.json) has 13 unchanged files and identifies both frozen plans. The [derivation record](reports/score/score_identity_derivation.md) and [prior reading scope](reports/score/source_scope.json) distinguish the oracle comparison from existing PPI and GMM results.

The [clean-directory validation](reports/score_clean_validation_20261112.json) records the completed full mathematical quick check. The standard quick check verifies all eight manuscript tables, the exact rational inequalities used in Proposition 2, the full covariance and objective identities, and the 16000 retained score-path records. It regenerates all 4000 matched truth scores and 2355 ordinary separators. It does not rerun all original Monte Carlo classification programs.

Regenerate the new analysis in fresh output directories:

```sh
python analysis/variance_reference_existence_audit.py --plan analysis/variance_reference_population_plan_20261112.json --out regenerated/score_population
python analysis/verify_variance_score_identity.py --out regenerated/score_identity
python analysis/certify_variance_reference_counterexample.py --out regenerated/score_bounds
python analysis/variance_reference_matched_phase.py --plan analysis/variance_reference_phase_plan_20261112.json --out regenerated/score_phase
python analysis/verify_variance_reference_phase.py --plan analysis/variance_reference_phase_plan_20261112.json --out regenerated/score_phase --population regenerated/score_population --certificate regenerated/score_bounds/rational_counterexample_certificate.json
python analysis/plot_variance_reference_audit.py --phase regenerated/score_phase --population regenerated/score_population --out regenerated/score_publication
```

The last command writes the figure, editable SVG and full table to the temporary output directory. The complete simulator supports resuming only with the same source and plan hashes. No questionnaire inputs enter this analysis.

## Reproduce the observed reference comparison

Section4.3and AppendixB5 extend the within-objective comparison to a frozen observed reference. The source closure is byte-identical to the completed public-design runs. The observed-reference stage reconciled all nine manuscript tables and the10000retained path targets; its exact scalar illustration uses fraction arithmetic. It does not rerun every public path classifier.

```sh
python analysis/observable_score_path_audit.py --plan analysis/observable_score_plan_20261112.json --input-dir local_inputs --reference results/observable/reference_power_weights.csv --out regenerated/observable_high
python analysis/observable_low_label_path.py --plan analysis/observable_low_label_plan_20261112.json --input-dir local_inputs --out regenerated/observable_low
python analysis/certify_observable_score_path.py --out regenerated/observable_high/exact_example
python analysis/verify_observable_score_path.py --input-dir local_inputs --high regenerated/observable_high --low regenerated/observable_low --out regenerated/observable_high/verification
```

The already documented input fetcher prepares local_inputs. To verify retained records without recomputing all10000targets, pass results/observable/high and results/observable/low as --high and --low and write verification to a fresh directory. It reconstructs870support states,checks them against independent residual-interior programs,and verifies all490exterior directions and391ordinary weak separators. The original scalar-interface failure is preserved in reports/observable/diagnostics;the original11-coordinate analysis sources are unchanged.

The same500block permutations supply both label budgets and prediction routines. Strictly interior base powers have201and176allocations at50labels,80and65at20labels. Nine prediction-by-allocation comparisons change finite availability in six distinct permutations,with seven exterior-to-finite changes and two in reverse. Zero-power cases are separate because recomputed prediction-score variance can vanish. The ridge reference remains finite on all500low-budget allocations;these comparisons assert no prediction-error or inference advantage. The [closer-prior scope](reports/observable/closest_prior_scope.json) retains current reading limits.

The complete-GMM stage validation is recorded in [its clean check](reports/complete_gmm_clean_validation_20261112.json). It passes the full mathematical quick check and a separate public-record verification. The previous validation records describe their dated stages. Publication sufficiency and exhaustive literature priority remain open.


## Complete moments and objective-specific existence

Appendix B6 compares the existing complete GMM criterion with the compensated loss on the same retained scalar design. The complete criterion attains a finite global minimum although the human MLE is separated and unit-power compensation is unbounded below. A fraction witness below both coefficient-tail limits proves attainment; seven numerical starts remain diagnostics. The specified reference is zero, and the weight is the uncalibrated centered covariance inverse with covariance ridge. CC-PPI additionally specifies its supervised pilot, protected covariance update and final fitting procedure. No comparative risk or new-estimator claim is made.

```sh
python analysis/certify_complete_gmm_comparison.py --out regenerated/complete_gmm
python analysis/verify_complete_gmm_comparison.py --certificate regenerated/complete_gmm/complete_gmm_exact_comparison.json --out regenerated/complete_gmm/verification
```

The [derivation](reports/complete_gmm/derivation.md) proves the exact comparison and the human-MLE implication of the inspected prior sufficient certificate. All nine manuscript tables and original simulated/public-data records are preserved. The current layout and clean-package records supersede the dated 45-page integration stage without changing those historical files.

The earlier [complete-GMM clean validation](reports/complete_gmm_clean_validation_20261112.json) reruns the full mathematical quick check and exact comparison from a fresh extraction. All 64 analysis files match that tested package. It preserves all nine table blocks and distinguishes complete-moment existence from the compensated-loss boundary.


## Probability loss and the current reading versions

The current full manuscript integrates Proposition 6 and its proof in Appendix D6, public paired loss Table5 and null-Gaussian TableD1. That integration stage contained eleven tables, preserving the previous nine blocks and all preexisting displayed mathematical bodies. Complete menu definitions and theorems are in AppendixF0; main Section3.9 retains their assumptions, consequences and matched evidence. The [main text](manuscript/main_text.md) and native main_text.docx provide a reading version with data availability and references. The [technical appendix text](manuscript/technical_appendices.md) and the full paper retain the complete argument.

The [closest-theorem comparison](reports/probability_integration/closest_theorem_comparison.json) specifies exactly which source versions and theorem units were read. The [journal reading record](reports/probability_integration/journal_readership_and_writing.json) distinguishes methodological relevance from publication prediction. The original [standalone probability-loss report](results/probability_risk/README.md) and its clean validation are frozen records from the immediately preceding, separate-supplement stage. The current integration records supersede their historical statement that the main manuscript had not yet been changed. Their numerical source closure and retained coefficients are unchanged.

```sh
python analysis/reconcile_probability_risk_tables.py --out regenerated/probability_risk_tables
python analysis/verify_vanishing_power_risk.py --root results/probability_risk --input-dir local_inputs --out regenerated/probability_risk_verification
python analysis/summarize_vanishing_power_risk.py --root results/probability_risk --out regenerated/probability_risk_summary
```

The first command also runs inside quick_check.py and reconciles both new tables against all1920verified records, checking common public-fit cohorts and each simulation denominator. The retained numerical fits and endpoint LPs were already replayed from an independent standalone extraction; that record remains in results/probability_risk/clean_validation.json. Full raw path optimization and download instructions remain available in the preceding standalone package; no private questionnaire or downloaded pickle input is included here.


## Known positive scales and nested auxiliary budgets

Section 3.10 and Appendix G extend the same objective to a fixed bounded known positive multiplier of the entire utility, including its fitted intercept. The comparison matches full regular information and keeps ordinary human separation identical on each paired sample. Along an established variance rule, a growing auxiliary sample can improve regular covariance while removing the unrestricted finite estimate. The later partial recovery and both transition directions remain in Table 6 and the retained paired records. At held power the balanced population boundary is nondecreasing with the auxiliary ratio. No arbitrary finite-sample monotonicity or general estimator ranking follows.

The [evidence snapshot](results/radial_budget/README.md) contains all 12000 criterion records, representing 10000 distinct objectives; 2000 smallest-ratio duplicate records are labelled. All distinct objectives already received independent original-loss verification. [Scientific source hashes](results/radial_budget/source_hashes.json) identify the eleven unchanged source files, plans and numerical input. The publication builder separately reconciles every cell and transition, then generates Table 6 and the manuscript-sized Figure 6.

```sh
python analysis/build_radial_publication_assets.py --out regenerated/radial_tables
python analysis/verify_radial_auxiliary_budget.py --root results/radial_budget --plan analysis/radial_auxiliary_budget_plan_20261112.json --reference inputs/radial_budget/previous_radial_verified.csv --out regenerated/radial_verification
python analysis/radial_auxiliary_budget_audit.py --plan analysis/radial_auxiliary_budget_plan_20261112.json --reference inputs/radial_budget/previous_radial_verified.csv --out regenerated/radial_generation
```

Use a fresh output directory for each generation. These are generated model data; no questionnaire inputs are required. The [nearest-prior mapping](reports/closest_prior_equivalence_20261113.md) identifies the Gaussian absorption step already covered by general prior theory. The proposed increment is its explicit compensated profile and the matched-information and auxiliary-budget consequences. Internal checks establish consistency of this evidence; originality, sufficient choice-analysis value and current submission rules remain separate assessment questions.


## Categorical separation faces and the existing coefficient paths

Appendix D3 extends the same signed-recession margin selector to complete and quasi-complete separation. Maximally separable human rows have margin at least one; neutral rows have zero margin. The human separation-face construction and generic margin limit retain their classical attribution. This deterministic implication does not assume Gaussian categorical support or claim a new general theory.

The [evidence](results/categorical_face/README.md) retains all60previous public block allocations,120predictor/allocation targets,and720old path records. There are574retained finite roots and146unavailable records. Independent checks cover1200row exposure programs,120original-objective support programs,96uncompressed selector programs,and574root/distance calculations. No new choice coefficient was fitted. Six initial row-solver errors remain in their own columns with simplex resolution; all numerical diagnostics are preserved.

The package contains the previous public coefficient records as a numeric generated input. Source research pickle files remain external and are fetched by their pinned manifest. If already downloaded, the fetcher verifies and reuses them.

```sh
python analysis/fetch_public_conjoint_audit.py --out local_inputs
python analysis/categorical_face_path.py --input-dir local_inputs --coefficients inputs/categorical_face/existing_public_coefficients.json --plan results/categorical_face/plan.json --out regenerated/categorical_face/results
python analysis/verify_categorical_face_path.py --input-dir local_inputs --coefficients inputs/categorical_face/existing_public_coefficients.json --plan results/categorical_face/plan.json --results regenerated/categorical_face/results --out regenerated/categorical_face/verification
python analysis/summarize_categorical_face_path.py --results regenerated/categorical_face/results --verification regenerated/categorical_face/verification --out regenerated/categorical_face/summary
```

Use fresh output directories. Distance is measured against the entire optimal selector set with objective slack1e−8. Figure shading is the descriptive interquartile range on common six-power allocation cohorts. The finite grid supplies no convergence-rate or population-confidence claim. The new result strengthens the categorical application; the main contribution remains the compensated attainment boundary and its matched-information consequences.


## Calibrated predictors and the focused current manuscript

Appendix B4 specializes the existing convex-order proposition to calibrated signals and gives a nonzero-signal construction with matched conditional prediction variance. The latter matches the full regular covariance function for every fixed power. The [evidence snapshot](results/calibrated_pair/README.md) retains all 36 exploratory pairs and 2400 finite-size classifications,including both upper-ratio adverse comparisons. Its historical statement that the manuscript was frozen describes the preceding standalone stage; the current integration supersedes that editorial status. No new estimator or calibration algorithm is introduced.

The two new scientific entry scripts are included with the existing 79 analysis sources. The source closure and all eight retained scientific outputs match the independent standalone clean run. Re-run its independent verifier with:

```sh
python analysis/verify_calibrated_choice_pair.py --input results/calibrated_pair/results --out regenerated/calibrated_pair_verification
```

The snapshot also has its own `run_reproduction.py` command for regenerating the full primary and verification workflow. Current dependencies pin the tested threadpoolctl 3.7.0; historical version records remain unchanged. The [editable highlights](submission_materials/Highlights.docx) have five items of 75–76 characters. The subsequently read current JCM guide confirms the separate editable highlights requirement. The [13 November core integration](reports/core_revision_integration_20261113.json) and [its visual review](reports/core_revision_visual_qa_20261113.json) describe that dated stage; the current pointer at the start of this README identifies the present manuscript bytes.


The current [runtime record](reports/core_revision_runtime_versions_20261113.json) covers scientific checks and Word authoring. requirements.txt now includes the tested python-docx and lxml versions required by both authoring entry points. LibreOffice and Poppler are needed for rendering review, separately from the scientific calculations.


## Author information update on 13 November 2026

The [author facts](submission_materials/author_facts_20261113.json) record LU JIAHANG, Shandong University of Finance and Economics, and no research funding. The [separate author page](submission_materials/title_page.docx), [funding statement](submission_materials/funding_statement.md) and cover letter use those facts. The sole author and correspondence are now confirmed in the author record; final manuscript approval and portal declarations remain for author review. The author page is reproducible with `python authoring/build_title_page_docx.py --facts submission_materials/author_facts_20261113.json --out regenerated/title_page.docx` after creating the regenerated directory. Its [one-page visual review](reports/author_page_visual_qa_20261113.json) is complete.

The [13 November proof scope follow-up](reports/proof_scope_followup_20261113.md) verifies the latest version of the closest general M-estimator theorem and records the checks then outstanding. The [14 November internal review](reports/proof_contribution_review_20261114.md) explains the subsequently resolved citation and guide access, the added DIAL attribution and the remaining contribution questions. Neither internal review establishes a submission threshold.

## Reproduce the held out record diagnostic

The input fetcher above supplies the pinned numeric files. Run the full primary fit in a new directory:

```sh
python analysis/public_holdout_diagnostic.py --input-dir local_inputs --plan results/holdout/plan.json --out regenerated/holdout --canonical analysis
python analysis/verify_public_holdout.py --input-dir local_inputs --plan results/holdout/plan.json --results regenerated/holdout --out regenerated/holdout_checks --canonical analysis
```

The exact command options and compressed-coefficient replay instructions are in [the standalone README](results/holdout/README.md). The published run uses 500 fixed block permutations and nested 50/150-label budgets. The final 3405 rows enter no model fitting or tuning. Allocation MCSEs condition on the released file. Verification reconstructs original objectives and metrics; it is distinct from a second complete primary refit. The package's quick check reconciles all 28 Table C1 cells, availability denominators, source hashes and retained verification records without downloading data.
