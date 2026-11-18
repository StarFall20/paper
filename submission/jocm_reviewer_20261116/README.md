# Reproduce When prediction assisted logit estimates exist

Section 4.6 and Appendix C13 add the choice-probability and price comparisons near positive power limits. The endpoint analysis uses 107 positive uncapped allocations and 835 same-budget unit-interior controls. Appendix C12 retains the original feasible-power range and comparisons. The complete and reading copies have 77 and 30 reviewed pages, with 20 tables and 151 displays in the complete copy. Start with [the main text](manuscript/main_text.docx), [the complete paper and proofs](manuscript/paper.docx), and the [evidence map](EVIDENCE_MAP.md). The [current source and layout pointer](reports/current_manuscript.json) identifies the release. Old editorial integration reports have been removed from this copy; retained dated files serve source provenance or executable checks.

Use Python 3.12 with [the recorded versions](requirements.txt). Run these commands from this directory, using a new output directory:

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python verify_package.py
python analysis/quick_check.py --out regenerated/checks --full-math
```

The first command verifies inventory, source provenance, local links, manuscript hashes and nested manifests. The second reruns analytic checks and reconciles all 20 manuscript tables. It does not refit every historical simulation or public allocation. [REPRODUCTION_COMMANDS.md](REPRODUCTION_COMMANDS.md) lists the original analysis commands and their computational scope.

Three recent simulation studies have self-contained entry points. Their default workflows reconstruct saved records and independently verify them; `--full` repeats their primary computations:

```sh
python results/learned_training/run_reproduction.py --out regenerated/learned_training
python results/menu_quality/run_reproduction.py --out regenerated/menu_quality
python results/discrete_menu_boundary/run_reproduction.py --out regenerated/discrete_menu_boundary
```

The two recent public-data studies require the pinned released numeric inputs. Download and verify their retained records with:

```sh
python analysis/fetch_public_conjoint_audit.py --out local_inputs
python results/holdout/analysis/verify_public_holdout.py --input-dir local_inputs --plan results/holdout/plan.json --results results/holdout/results --out regenerated/holdout --canonical results/holdout/analysis
python results/public_auxiliary_budget/run_reproduction.py --input-dir local_inputs --out regenerated/public_auxiliary_budget
```

Consult each study's README before full runs. The held-out study has separate fitting and verification entries. It uses choice rows whose respondent IDs are unavailable. Generated coupled variants, allocation permutations and additional solver calls are not independent people or new model cohorts.

The [curation record](CURATION.json) lists every retained and omitted parent file. All 83 root analysis sources are retained and unchanged in the current integration; quick_check.py checks 20 tables. The new nested one-step supplement contains nine source files and all 60000 method records. All preceding scientific result files are unchanged. The [attribution record](ATTRIBUTION.md) identifies established methods and the owners of public inputs. No questionnaire observations, participant identifiers, authentication records or downloaded article full texts enter this package.

This is a reproduction release for author and reviewer inspection. Final author approval and publisher declarations remain separate. The manuscript studies an existence boundary; it does not establish a generally superior new estimator or certify global priority.

The one-step comparison uses the original 24 conditions with 500 repetitions and the same ridge pilots. Its full primary replay has already reproduced the saved outputs in an isolated directory. For a new replay, fetch the pinned public inputs first, then run:

```sh
python results/one_step_source_audit/run_reproduction.py --input-dir local_inputs --out regenerated/one_step --full
```

The [source scope](results/one_step_source_audit/source_scope.json) distinguishes static upstream code reading, the preprint formula and the common-pilot experiment. This comparison does not assert literal upstream API reproduction or identity with formal PPI++ Appendices A–E.

The [EPI scope supplement](results/epi_scope/README.md) specializes an established empirical-likelihood supervised-score constraint to the existing scalar example. Recheck it without any downloaded inputs or new simulations:

```sh
python results/epi_scope/check_existing_design.py --out regenerated/epi/exact
python results/epi_scope/verify_existing_design.py --certificate regenerated/epi/exact/exact_existing_design.json --out regenerated/epi/verification
```

This is a classical separation-scope clarification. It supplies no new estimator, performance ranking or global priority certificate. The [current validation](reports/epi_validation_20261117.json) distinguishes inherited unchanged mathematics from the new exact-sign check.

The [matched-reference-information supplement](results/matched_local_information/README.md) is integrated in the current manuscript. Reproduce its deterministic calculations without generating participants or model cohorts:

```sh
python results/matched_local_information/check_exact_information.py --out regenerated/matched/exact
python results/matched_local_information/verify_original_loss.py --certificate regenerated/matched/exact/exact_information.json --out regenerated/matched/verification
```

The [current validation](reports/matched_information_validation_20261117.json) separates inherited unchanged science from the new exact and independent 90-digit original-loss checks. Reference-point identities imply no actual sampling-variance equality, variance-optimal selection or universal priority claim. Earlier dated reports retain their historical scope.

The [released food DCE audit](results/public_panel_dce/README.md) is integrated in the current manuscript. Its existing original-row verifier checked all 16000 records, 15418 available coefficients, 253 exterior directions and 2000 cluster weights. Check manuscript Tables C4/C5 directly from all saved method records:

```sh
python results/public_panel_dce/analysis/reconcile_tables.py --package-root . --out regenerated/public_panel_table_check.json
```

Full source download, primary fit and independent reconstruction commands are in the nested README. They use public observed choices and create no new respondent or AI answer. Small-budget failures, larger-budget absence of failure, conditional-risk denominators and the stronger all-human training-cost comparator remain reported. The [current validation](reports/public_panel_validation_20261117.json) distinguishes the new empirical checks from inherited unchanged mathematical evidence. Earlier dated reports retain historical scope.


The [official API runtime snapshot](results/official_ppi_runtime/README.md) and [secondary price-equivalent snapshot](results/public_panel_price_ratios/README.md) are now integrated in the article. Each retains its dated standalone scope and pre-integration proposal, manifest, independent verifier and completed separate-copy replay. Follow their own environment instructions: official API execution uses a separate dependency environment. The current package check validates their inventories without refitting models. Price ratios are descriptive, with no true-WTP accuracy or population-confidence claim. Selected API failures establish diagnostic examples, not failure prevalence. The complete user-provided March 2024 PPI++ arXiv v2 body and appendices and pinned official code are available and used. Identity with the 2026 formal version is unverified. Specific-result priority and sufficient choice-analysis value remain under assessment.


The current nested [bounded-scale](results/bounded_scale_scope/README.md) and [direct-theorem comparison](results/direct_theorem_comparison/README.md) packages include their plans, independent checks and deterministic replays. Their internal prose records the pre-integration stage; the current integrated manuscript and [pointer](reports/current_manuscript.json) supersede statements about proposed edits. Specific-result priority and sufficient choice-analysis value remain unresolved.


The [independent-workflow source audit](results/independent_released_workflow/README.md) is a dated pre-integration record; its proposed-edit and unchanged-Word statements are superseded by the [current pointer](reports/current_manuscript.json). It preserves the input limits, saved-output checks and original-power versus changed-power distinction. Only its local navigation link and nested manifest were adapted. No source PDF, raw workbook or individual prediction table is redistributed.
