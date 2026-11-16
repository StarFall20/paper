# Reproduce When prediction assisted logit estimates exist

The reviewed manuscript and its scientific results are preserved byte-for-byte from the 16 November 2026 discrete-profile release. Start with [the main text](manuscript/main_text.docx), [the complete paper and proofs](manuscript/paper.docx), and the [evidence map](EVIDENCE_MAP.md). The [current source and layout pointer](reports/current_manuscript.json) identifies the release. Old editorial integration reports have been removed from this copy; retained dated files serve source provenance or executable checks.

Use Python 3.12 with [the recorded versions](requirements.txt). Run these commands from this directory, using a new output directory:

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python verify_package.py
python analysis/quick_check.py --out regenerated/checks --full-math
```

The first command verifies inventory, source provenance, local links, manuscript hashes and nested manifests. The second reruns analytic checks and reconciles all 17 manuscript tables. It does not refit every historical simulation or public allocation. [REPRODUCTION_COMMANDS.md](REPRODUCTION_COMMANDS.md) lists the original analysis commands and their computational scope.

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

The [curation record](CURATION.json) lists every retained and omitted parent file. All 83 root analysis sources and all released scientific result files are unchanged. The [attribution record](ATTRIBUTION.md) identifies established methods and the owners of public inputs. No questionnaire observations, participant identifiers, authentication records or downloaded article full texts enter this package.

This is a reproduction release for author and reviewer inspection. Final author approval and publisher declarations remain separate. The manuscript studies an existence boundary; it does not establish a generally superior new estimator or certify global priority.
