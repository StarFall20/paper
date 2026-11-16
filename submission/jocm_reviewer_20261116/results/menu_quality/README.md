# Probability quality of finite whole-menu estimates

Follow-up started 15 November 2026; package verification continues on 16 November. This supplement reuses every cohort and dimension of the existing three-alternative availability experiment. It introduces no estimator, new respondents or LLM answers. The supplement was prepared before parent-manuscript integration. The 16 November parent release incorporates Section 5.7, Appendix F5 and Table F3; its current pointer and visual-review record identify the reviewed manuscript.

## Result and interpretation

All 1000 original Gaussian cohorts, six budget/dimension conditions and five methods give 15000 records. All 9000 original support classifications reproduce. The three original methods retain their exact power settings; human ridge and hard-label pooling supply explicit additional objectives. All 11018 supported or penalized numerical fits pass the stated stopping checks. Unavailable fits remain in the data.

At 80 labelled menus and 52 common noise slopes, the corrected variance-reference objective is finite on 269/500 draws, compared with 32/500 for ordinary logit and zero for unit power. On those same 269 draws, median new-menu human log loss is 1.8524 for correction, 1.6837 for human ridge and 1.0207 for pooling. With ridge fallback in the 231 unavailable correction cases, all-allocation mean loss is 2.264907, versus 1.696402 for direct ridge; the paired difference is 0.568505 with allocation Monte Carlo standard error 0.128705. Fallback is a separately stated operational rule, not the unpenalized estimator.

The [six-condition table](summary/tableF3.md) uses shared finite numerical denominators. All [30 method cells](results/quality_summary.csv), [paired comparisons](results/paired_with_ridge.csv) and [fallback results](results/ridge_fallback.csv) retain adverse and favorable outcomes. Conditional means can have heavy tails; medians, 95th quantiles, maxima and parameter errors are all retained. The [24 availability strata](summary/all_availability_strata.csv) are a secondary descriptive analysis conducted after the primary risk outputs; no stratum is dropped.

These results support a finite-estimation analysis, not a claim that maximizing availability improves probability forecasts or that this fixed oracle power is optimal for high-dimensional risk. The fixed ridge penalty is not cross-validated. Pooling changes the human target under the noisy proxy model. This experiment supplies no general ranking or interval-coverage result.

## Model, methods and risk

The original sampling model has three alternatives, utility signal 1.5, independent proxy error 0.25, equal labelled and auxiliary budgets, 40 or 80 labels, and noise slopes per label 0.15, 0.4 or 0.65. Common coefficients, alternative constants and the signal slope are fitted. Each original cohort supplies every dimension and method. The original [plan](reports/menu_extension_20261111/plans/hard_menu_phase_plan_20261111.json), [generator](analysis/hard_menu_matched_phase.py) and [9000 availability records](reports/menu_extension_20261111/hard_outputs/hard_menu_matched_phase_trials.csv) are included and pinned.

The follow-up [plan](analysis/menu_quality_plan_20261115.json) preceded risk fitting, after the original availability results had been inspected. It is not a preregistration. Methods are ordinary human MLE, the existing oracle variance-reference power 0.2195760406786864, unit correction, human ridge with penalty equal to the coefficient norm squared divided by twice the human label budget, and equal-menu-weight pooling of observed and hard predicted categories. Every fitted coordinate receives the ridge penalty.

Human probability loss is conditional expected loss on independent fresh Gaussian alternative profiles. Given a fitted common coefficient, the noise utility has exactly the distribution of its coefficient norm times independent alternative normals. This reduces the integral without generating new test choices. Eight independent scrambled Sobol normal integrals use 16384 nodes per fit in total. Integration scramble errors and allocation Monte Carlo errors describe different quantities. Squared coefficient error uses the original reference-category constant basis and known generating coefficient.

## Verification

The [primary verifier](verification/complete_verification.json) checks every numerical root against the former moment objective, reconstructs coefficient errors and reconciles all 30 cells. Maximum original-objective score norm is below 1.0e-9 and maximum objective discrepancy is below 2.0e-14. A separately coded larger integral checks 67 fits selected by deterministic stride and the 20 largest coefficient norms; each uses 262144 antithetic integration evaluations. Its scramble tolerances are numerical diagnostics, not deterministic error bounds. The [software checks](verification/software_checks.json) verify derivatives, pooling identity, Gaussian rotation and exact zero/truth risk identities. The [summary check](summary/summary_verification.json) independently reconciles paired and fallback denominators.

No downloaded research code is executed. All inputs are existing generated research records. Raw private questionnaires, participant identifiers, authentication information and external article full texts are absent. Established PPI correction, ordinary support geometry, ridge and pooling retain their attribution in the parent manuscript; these numerical results are not originality or acceptance certification.

## Reproduce

Install the recorded [dependencies](requirements.txt). From this directory, first verify the package. Each reproduction output directory must be fresh.

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python verify_package.py
python run_reproduction.py --out regenerated
python run_reproduction.py --out regenerated_full --full
```

The default reconstructs the losslessly stored checkpoint and raw CSV, then reruns all saved-root, metric and larger-integral checks. The full option reconstructs all original cohorts and refits every stated method, then runs the same checks. Both compare the nine scientific result/summary outputs with the saved versions; run times are excluded. Coefficient data are split into bounded binary chunks solely for reliable public backup. [source_record.json](source_record.json) supplies individual and reconstructed hashes. No data download is needed.
