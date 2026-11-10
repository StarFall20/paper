The current integrated manuscript and isolated reproduction entry are [here](../../submission/finite_choice_20261110/README.md). The [10November progress report](完整论文与复现包进展_20261110.md) states the remaining contribution assessment; artifact completion is not a submission-readiness finding.

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

`MAPE_*` fields mean100×mean(abs(estimate−reference)/(1+abs(reference))). They are stabilized parameter errors, not standard MAPE. `point_estimation_summary.csv` includes finite-fit counts; `paired_point_comparisons.csv` conditions on shared finite fits. `status_counts.csv` and `nonexistence_ray_checks.json` must accompany those comparisons. That public-data audit evaluates point estimates. The separate known-truth simulation below evaluates interval coverage; neither study establishes general PPI++ superiority or AI replacement.

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

## Finite auxiliary samples and known-truth coverage

Read [the finite-pool report](有限样本验证与理论扩展.md), [the finite-ratio derivation](finite_ratio_boundary_candidate.md) and the two plans `finite_pool_validation_plan.json` and `finite_ratio_check_plan.json`.

```sh
python analysis/finite_pool_choice_validation.py --input-dir ../work/gai_data --out ../outputs/public_conjoint_audit --replications 500
python analysis/verify_finite_pool_covariance.py --out ../outputs/public_conjoint_audit
python analysis/finite_ratio_choice_boundary.py --out ../outputs/public_conjoint_audit
python analysis/write_finite_pool_report.py --out ../outputs/public_conjoint_audit
```

Use a fresh output directory for a clean reproduction. The first command draws independent choice rows from the1200-row empirical attribute support, fixes the converged11-dimensional fit as a simulation parameter, and generates new Bernoulli choices and informative flipped-label proxies. No new respondent or LLM responses are collected. The12 conditions have500 repetitions each and seven prespecified estimators. The generated `finite_pool_trial_diagnostics.csv` and per-repetition `finite_pool_unavailable_cases.json` stay local; the repository retains aggregate coverage, every status count, availability, tails, coordinate results and paired comparisons. The covariance verifier checks numerical identities, without asserting interval coverage from those identities.

The original completed benchmark recorded script SHA256 `b859719384c90623fd22937d290f76ce327d9240c06d29c12f239612b9f95601`. A subsequent one-coordinate check required wrapping scalar covariance output in `np.atleast_2d` and removed a redundant scalar assertion. This is the identity operation for the original11-coordinate experiment; its outputs were retained without a rerun. `finite_pool_source_compatibility.json` records both source hashes and this bounded compatibility check. The old completed checkpoint intentionally retains its original hash. `--resume` accepts only a checkpoint produced by the identical current source; it does not silently accept the older hash.

The finite-ratio experiment separately uses Gaussian attributes with a balanced null, a fixed zero intercept, finite N/m and fixed positive power. It completes80 conditions with500 LP draws each. These are feasibility experiments, not confidence-interval experiments or actual respondent counts. The100 original-moment versus interval LP comparisons use positive powers, including unit power. Zero power is the separate classical ordinary-logit endpoint; the box LP cannot diagnose that endpoint by checking zero feasibility, because zero is automatically feasible there.

## Utility signal and estimated intercept

Read [the signal report](非零效用信号与有限样本边界.md) and [candidate proof](finite_ratio_signal_candidate.md). The latter now supplies the multiplier/scale tightness and uniform-convergence argument missing from the original outline. Independent proof and priority review remain open.

```sh
python analysis/finite_ratio_signal_boundary.py --out ../outputs/public_conjoint_audit
python analysis/write_finite_ratio_signal_report.py --out ../outputs/public_conjoint_audit
```

The signal experiment is separate:80 conditions with300 draws each, utility signals1 and2, a free intercept, N/m=1,4 and two fixed positive powers. All300 draws enter each feasibility denominator; no numerical-unresolved case occurred. The script also compares120 original-moment/interval LPs,12 direct cone projections versus polar saddle calculations,100/200-node population integration and the separate classical ordinary-logit endpoint. `--checks-only` runs those identities without the phase experiment. The source hash in `finite_ratio_signal_status.json` identifies the completed run. This oracle experiment does not validate data-dependent weights or fitted predictors. The next extension below evaluates imperfect frozen predictions.

## Imperfect predictions and stronger utility

Read [the new report](预测偏差检验与创新边界深化.md), [general-M-estimator applicability audit](general_M_estimator_prior_audit.md) and [matched-risk/covariance proposition](matched_moment_existence_proposition.md). This remains one corrected-choice question; generic conic tools, variance tuning and finite-sample PPI costs are attributed prior work.

```sh
python analysis/imperfect_predictor_choice_boundary.py --out ../outputs/public_conjoint_audit
python analysis/verify_matched_moment_proposition.py --out ../outputs/public_conjoint_audit
python analysis/imperfect_finite_pool_validation.py --input-dir ../work/gai_data --out ../outputs/public_conjoint_audit --replications 500
python analysis/imperfect_strong_signal_validation.py --input-dir ../work/gai_data --out ../outputs/public_conjoint_audit --replications 500
python analysis/imperfect_population_targets.py --input-dir ../work/gai_data --out ../outputs/public_conjoint_audit
python analysis/imperfect_population_targets.py --input-dir ../work/gai_data --out ../outputs/public_conjoint_audit --truth-scale 3 --prefix strong_signal_pool
python analysis/write_imperfect_prediction_report.py --out ../outputs/public_conjoint_audit
```

The imperfect-prediction phase grid has80 conditions with300 draws each. Its asymmetric case estimates both saddle multipliers. The fixed comparison power uses the existing covariance formula for an orthogonal slope; it is not claimed jointly optimal under proportional asymptotics. A reversed prediction chooses the ordinary zero-power endpoint in that formula and is retained separately.

The two finite-support benchmarks have three prediction cases, two label budgets and500 repetitions per condition each. The stronger benchmark multiplies the same frozen utility vector by3. Every prediction/method comparison within a signal group shares the generated rows and outcomes. The soft-pooling method uses the same frozen prediction as its comparators; it is an oracle only in the correctly calibrated case. Every failure remains in status counts. Conditional coverage and issued-and-covers use explicit different denominators. The two generated trial CSVs and per-repetition unavailable-case JSONs stay local; the public package retains aggregates. No new human or LLM answers are collected.

`--resume` on either finite-support script accepts only its matching source/helper hashes and replication count. The population-target script independently integrates over both proxy values on the finite attribute support. Its fixed-power score identity does not prove finite-sample unbiasedness for estimated weights.
