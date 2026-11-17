# Reproduce When prediction assisted logit estimates exist

The reviewed manuscript includes the 17 November 2026 one-step comparator integration in Sections 5.1–5.2 and Appendix C8. All 17 preceding tables and 140 display bodies remain unchanged; Table C3 adds the full 24-condition loss sensitivity. Start with [the main text](manuscript/main_text.docx), [the complete paper and proofs](manuscript/paper.docx), and the [evidence map](EVIDENCE_MAP.md). The [current source and layout pointer](reports/current_manuscript.json) identifies the release. Old editorial integration reports have been removed from this copy; retained dated files serve source provenance or executable checks.

Use Python 3.12 with [the recorded versions](requirements.txt). Run these commands from this directory, using a new output directory:

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python verify_package.py
python analysis/quick_check.py --out regenerated/checks --full-math
```

The first command verifies inventory, source provenance, local links, manuscript hashes and nested manifests. The second reruns analytic checks and reconciles all 18 manuscript tables. It does not refit every historical simulation or public allocation. [REPRODUCTION_COMMANDS.md](REPRODUCTION_COMMANDS.md) lists the original analysis commands and their computational scope.

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

The [curation record](CURATION.json) lists every retained and omitted parent file. All 83 root analysis sources are retained: 82 analysis algorithms are unchanged and quick_check.py now checks 18 tables. The new nested one-step supplement contains nine source files and all 60000 method records. All preceding scientific result files are unchanged. The [attribution record](ATTRIBUTION.md) identifies established methods and the owners of public inputs. No questionnaire observations, participant identifiers, authentication records or downloaded article full texts enter this package.

This is a reproduction release for author and reviewer inspection. Final author approval and publisher declarations remain separate. The manuscript studies an existence boundary; it does not establish a generally superior new estimator or certify global priority.

The one-step comparison uses the original 24 conditions with 500 repetitions and the same ridge pilots. Its full primary replay has already reproduced the saved outputs in an isolated directory. For a new replay, fetch the pinned public inputs first, then run:

```sh
python results/one_step_source_audit/run_reproduction.py --input-dir local_inputs --out regenerated/one_step --full
```

The [source scope](results/one_step_source_audit/source_scope.json) distinguishes static upstream code reading, the preprint formula and the common-pilot experiment. This comparison does not assert literal upstream API reproduction or identity with formal PPI++ Appendices A–E.
