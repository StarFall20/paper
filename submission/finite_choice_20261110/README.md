# Finite solutions in prediction assisted binary logit

This compact archive accompanies **When prediction assisted logit estimates exist**. It contains one scientific argument, its completed aggregate results and the code needed to regenerate them. It excludes the historical research notebook, raw questionnaires, participant level response vectors and downloaded numeric pickle inputs. The independent 300 record questionnaire application is retained in the research archive and supplies no observations to this paper.

The main result concerns independent Gaussian remaining attributes, an estimated intercept, one utility signal, a frozen predictor, fixed positive correction power and a finite auxiliary to label ratio. Under the same signal and estimated intercept assumptions, a second result gives the simultaneous upper existence boundary over every correction power and the joint limit for strictly positive powers tending to zero. Its boundary strictly exceeds ordinary logit for each fixed strictly interior predictor; one excluded hard-prediction counterexample removes that strict gain. The finite design LP and the symmetric Gaussian zonotope special case are classical. Numerical agreement does not establish priority or a general categorical DCE phase theorem.

## Start here

- [Complete manuscript and proofs](manuscript/paper.md).
- [Editable Word manuscript](manuscript/paper.docx), with native Word equations.
- [Table 1 boundary values](manuscript/table1.csv), [Table 2 estimation comparisons](manuscript/table2.csv) and [Table 3 fixed power replay](manuscript/table3.csv).
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

The quick check requires no external data. It checks a closed form logistic root, boundary/exterior/rank cases, alternative moment and interval LPs, normal integration, proper augmentation weight invariance and intercept covariance formulas. It reconciles all three main tables and Appendix Table E1 and the completed aggregate grids. It preserves the single original unknown LP status and checks its independent positive-gap resolution. `--full-math` adds signal, imperfect prediction and positive power numerical checks. It does not rerun the full Monte Carlo grids.

Optional Word authoring uses the packages in `authoring/requirements.txt`:

```sh
python authoring/build_manuscript_docx.py --source manuscript/paper.md --out manuscript/paper.docx --report regenerated/docx_structure.json
```

The strict TeX subset converter is written for this manuscript and emits native OMML equations. It is not a general TeX conversion service. The shipped Word document was rendered and all 30 pages inspected; equation structure and page checks are recorded in `results/manuscript_render_review.json`. Changes to the source require a fresh visual review. Rebuilding a DOCX can change ZIP metadata even when its text and equations agree.

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

Table 3 retains all availability counts. Figure 3 compares errors on the same 412 available unit power fits; the complete paired table and human separation strata accompany it. The original positive power right limit takes the proportional sample limit first and the power limit second. Its recorded derivation and status flags predate the stronger balanced joint result below; they are retained unchanged. No new optimal inference weight or confidence interval guarantee follows. The [original derivation](reports/public_conjoint_audit_20261109/vanishing_power_derivation_20261110.md) and [regularization prior comparison](reports/public_conjoint_audit_20261109/synthetic_regularization_prior_comparison_20261110.md) state that earlier step.

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

The manuscript's Section3.7 and AppendixE retain a fixed number of prediction indices among fitted Gaussian attributes. A conditional corollary covers independently trained linear logistic scores when their signal overlap,orthogonal norm and intercept converge to finite deterministic limits. It does not supply arbitrary cross-fitting or black-box training guarantees. Identical calibrated response/prediction laws can have different existence boundaries when their relationship to fitted attributes changes. The generic alignment principle and Gaussian rotation are established tools;the paper calculates their role in its compensated objective.

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
