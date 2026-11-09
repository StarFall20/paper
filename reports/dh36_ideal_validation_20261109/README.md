# DH36 ideal-assignment scenario: 2026-11-09

Read `全量模型与论点验证报告.md` and `assumptions.json` first. All 360 exported records enter the full-sample models. Raw provenance remains unchanged. Ideal independent records and complete 180:180 random assignment are assumptions, not certified facts.

## Reproduction

Python 3.12 and packages in requirements.txt are required. Place controlled raw exports at PRIVATE/V1_raw.xlsx and PRIVATE/V2_raw.xlsx. That directory must be writable because private fold/split files are stored there. It is never uploaded.

```sh
python dh36_assumption_conditioned_validation.py --input-dir PRIVATE --instrument instrument.json --out RESULTS --permutations 49999 --mc-repetitions 500 --calibration-permutations 199
python dh36_finalize_validation.py --input-dir PRIVATE --instrument instrument.json --out RESULTS --permutations 999999 --null-repetitions 5000 --null-permutations 999 --power-repetitions 1000
python dh36_exact_marginal_checks.py --input-dir PRIVATE --instrument instrument.json --out RESULTS
python dh36_structural_closure_audit.py --input-dir PRIVATE --instrument instrument.json --out RESULTS
python dh36_report_validation.py --input-dir PRIVATE --instrument instrument.json --out RESULTS
```

The first stage preserves exploratory numerical outputs. The second stage supplies the final high-precision tests, boundary-repaired panel model, independent-start panel CV and extended calibration. The initial mixed-price model and initial chi-square power table are diagnostics, not the final models or calibrated power estimates. The first-stage panel CV used full-data fitted parameters as numerical starting values; it is retained only as a diagnostic. The final independent_start_panel_joint_CV.csv uses three fixed starting values in every training fold and supersedes that result.

Read randomization_tests_high_precision.csv, exact_marginal_task_tests.csv, independent_start_panel_joint_CV.csv, random_intercept_model_coefficients.csv, null_calibration_extended.csv and power_DGP_calibrated.csv for final results. Finite-sample exactness under joint exchangeability is stronger than the marginal sufficiency null. The exact taskwise Bonferroni/Holm analysis addresses the marginal null under arbitrary within-record dependence. Prospective power is conditional on a simulated DGP, not a guarantee for recruitment.

All outputs are aggregate-only. aggregate_cluster_moments.json permits an independent reconstruction of the observed Wald statistic; it does not encode the full participant response distribution needed for exact permutations. Raw exports, IPs, timestamps, identifiers and individual answer vectors remain private. Files referencing private input directories describe required local inputs; those directories are intentionally not part of the release.

The structural closure supplement checks multinomial nuisance information and profiled KL, the one-dimensional raw profile step, exploratory version-contrast decomposition and the scope of nine candidate maps. It does not alter the main null, the primary p-values or the original provenance.
