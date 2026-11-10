# Finite solutions in prediction assisted binary logit

This compact archive accompanies **When prediction assisted logit estimates exist**. It contains one scientific argument, its completed aggregate results and the code needed to regenerate them. It excludes the historical research notebook, raw questionnaires, participant level response vectors and downloaded numeric pickle inputs. The independent 300 record questionnaire application is retained in the research archive and supplies no observations to this paper.

The main result concerns independent Gaussian remaining attributes, an estimated intercept, one utility signal, a frozen predictor, fixed positive correction power and a finite auxiliary to label ratio. The finite design LP is classical and works directly with a supplied binary loss. Numerical agreement does not establish priority or a general categorical DCE phase theorem.

## Start here

- [Complete manuscript and proofs](manuscript/paper.md).
- [Editable Word manuscript](manuscript/paper.docx), with native Word equations.
- [Table 1 boundary values](manuscript/table1.csv) and [Table 2 estimation comparisons](manuscript/table2.csv).
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

The quick check requires no external data. It checks a closed form logistic root, boundary/exterior/rank cases, alternative moment and interval LPs, normal integration, independent cone projections and intercept covariance formulas. It also reconciles the completed aggregate grids and displayed table rounding. `--full-math` adds the signal and imperfect prediction numerical checks. It does not rerun the full Monte Carlo grids.

Optional Word authoring uses the packages in `authoring/requirements.txt`:

```sh
python authoring/build_manuscript_docx.py --source manuscript/paper.md --out manuscript/paper.docx --report regenerated/docx_structure.json
```

The strict TeX subset converter is written for this manuscript and emits native OMML equations. It is not a general TeX conversion service. The shipped Word document was rendered and all 17 pages inspected; equation structure and page checks are recorded in `results/manuscript_render_review.json`. Changes to the source require a fresh visual review. Rebuilding a DOCX can change ZIP metadata even when its text and equations agree.

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

## Inventory and output map

`analysis/` holds the independent estimator, theory, verification and figure code with their local import closure. `results/` holds the completed aggregate files. `manuscript/figures/` holds standard PNG and editable SVG plots. Tables are generated from those same aggregates. `MANIFEST.sha256` covers every shipped file except itself; `verify_package.py` verifies it and all local Markdown links without downloading inputs.

```sh
python verify_package.py
```

The paper and this archive do not assert journal acceptance or an exhaustive Scopus/Scholar search. The research archive retains the closest prior applicability review and the submission assessment separately.
