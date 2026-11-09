# Public binary-choice audit and finite-solution candidate

Read [the Chinese progress report](公开数据重分析与创新进展.md), [clipping identity](clipping_identity.md), [candidate proof](existence_candidate_derivation.md) and [closest-prior boundary](literature_boundary.md). Independent novelty and submission readiness are unproven.

The original source is Lu, Wang, Zhang and Zhang's [generative-augmented-inference](https://github.com/ChengLu1999/generative-augmented-inference), pinned at d18717206867c7a12ff470879a0dd5d488b300c2. Their README states MIT for code and CC-BY-4.0 for data. Source attribution and hashes are retained in the manifests. Numeric files remain outside this repository; no downloaded script is executed. Restricted numeric unpickling is used after hash verification.

Run from the repository root with Python and the versions recorded in `runtime_versions.json` available. No torch is needed. The reference Adam path uses an independent NumPy implementation of the inspected float32-buffer update; no claim to reproduce all PyTorch execution paths is made.

```sh
python analysis/fetch_public_conjoint_audit.py --out ../work/gai_data
python analysis/public_conjoint_optimizer_audit.py --input-dir ../work/gai_data --out ../outputs/public_conjoint_audit
python analysis/public_conjoint_matched_audit.py --input-dir ../work/gai_data --out ../outputs/public_conjoint_audit --allocations 500
python analysis/verify_choice_nonexistence.py --input-dir ../work/gai_data --out ../outputs/public_conjoint_audit
python analysis/rectified_choice_existence_checks.py --out ../outputs/public_conjoint_audit --replications 2000
python analysis/write_public_conjoint_audit.py --out ../outputs/public_conjoint_audit
```

Limit BLAS threads to one for the recorded numerical convention. The outputs include every trial and all fit statuses. Repository tables publish aggregates; the trial file is generated locally. The verification script expects the fixed seed,500 allocations and seven recorded exterior cases; it deliberately fails on a different configuration rather than certifying an unrecognized experiment.

The six matched methods share the1200-row prefix design and one empirical human-projection target. Label budgets count choice rows. Five-row blocks follow the source convention without claiming independently verified participant IDs. The nuisance implementation uses source-style five row folds and separate ridge logistic fits within hard-AI-label strata. A second corrected estimator omits the AI label. No fallback was needed in the500 allocations. This is not an independent-respondent cross-validation claim.

`MAPE_*` fields mean100×mean(abs(estimate−reference)/(1+abs(reference))). They are stabilized parameter errors, not standard MAPE. `point_estimation_summary.csv` includes finite-fit counts; `paired_point_comparisons.csv` conditions on shared finite fits. `status_counts.csv` and `nonexistence_ray_checks.json` must accompany those comparisons. No confidence-interval coverage, PPI++ superiority or general AI replacement conclusion has been tested.

The Gaussian figure uses simulated attributes and frozen predictors. Panel(a) plots exact probabilities with simulation points and approximate95% Monte Carlo error bars; dotted lines show proportional-dimension limits. Panel(b) plots simulated frequencies with the corresponding candidate limits. The two panels describe a theoretical benchmark, not the actual vaccine or DH36 design. The160 exact-probability cells include rare-event conditions; `exact_probability_consistency_checks.csv` records exact-binomial diagnostics, with multiple-cell qualification in the report.

## Power tuning and informative proxies

The extension is specified in `power_extension_plan.json` and `surrogate_signal_plan.json`. Run these after fetching the same numeric data:

```sh
python analysis/power_tuned_choice_audit.py --input-dir ../work/gai_data --out ../outputs/public_conjoint_audit --allocations 500
python analysis/informative_surrogate_existence.py --out ../outputs/public_conjoint_audit --replications 2000
python analysis/informative_surrogate_signal_checks.py --out ../outputs/public_conjoint_audit
python analysis/write_power_extension_report.py --out ../outputs/public_conjoint_audit
```

`power_tuned_choice_audit.py --resume` resumes a matching500-allocation checkpoint and verifies its script hash. The auxiliary-only PPI objective is explicit; it differs from the previous pooled-imputation objective. Human-root and source-style PPI-root initializers are separate methods. A missing verified PPI root is retained as `initialization_unavailable`; we do not infer what an unchecked optimizer's last iterate would do. Every exterior weighted moment receives a direct separating-ray check. The score and analytical derivative checks are in `power_numeric_checks.json`.

The pinned reference implementation is identified in `ppi_power_source_manifest.json`. Its published functions were inspected and independently implemented; no downloaded script runs. `power_reference_checks.json` records a source-hash and independent covariance-identity spot check, created during this analysis; it is not required by the report writer to rerun the numerical study. For a clean reproduction without that spot-check file, the report writer recomputes the independent tuning identity from its own estimator code.

The informative-proxy DGP is a frozen conditional-mean oracle, not newly generated LLM answers. Its exact balanced-null formula, nonzero-signal boundary and variance-optimal-limit argument are in Section8 of the derivation. The135 exact-probability cells and162 signal cells are separate from the earlier Gaussian simulations. This research does not establish priority merely by matching a simulation to a formula.
