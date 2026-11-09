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
