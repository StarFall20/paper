# Compensated attainment under a regular covariance budget

Start with the [complete findings and controls](evidence/README.md),[derivation](evidence/theory.md),[primary-source scope](evidence/closest_prior_scope.json) and [contribution decision](evidence/contribution_decision.json). The package preserves all195population conditions and5000selected-illustration method records,including raw numerical unknowns and their verified resolutions. It contains generated model data and prior cohort hashes; no questionnaires,participant information or downloaded empirical pickle inputs are included. The57-page manuscript remains separate and unchanged.

Use Python3.12 with the pinned requirements. Run from this directory with a fresh regenerated/study directory:

```sh
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python verify_package.py
python analysis/budgeted_attainment.py --plan analysis/attainment_variance_budget_plan_20261113.json --out regenerated/study/results
python analysis/verify_budgeted_attainment.py --root regenerated/study --plan analysis/attainment_variance_budget_plan_20261113.json --out regenerated/study/verification
python analysis/budgeted_attainment_phase.py --root regenerated/study --plan analysis/attainment_variance_budget_phase_plan_20261113.json --reference-hashes inputs/previous_source_cohort_hashes.csv --out regenerated/study/phase
python analysis/verify_budgeted_attainment_phase.py --root regenerated/study --plan analysis/attainment_variance_budget_phase_plan_20261113.json --reference-hashes inputs/previous_source_cohort_hashes.csv --out regenerated/study/phase_verification
python analysis/summarize_budgeted_attainment.py --root regenerated/study --out regenerated/study/summary
```

For verification of retained evidence,run either verifier with --root evidence and a fresh --out path. The full workflow above regenerates the optimization grid,independent KKT/integration checks,all5000primary support programs,all5000actual weighted-loss checks,and final verified tables. Runtime fields and SVG metadata can differ. Raw numerical unknown counts can vary with solver behavior; the final verifier resolves them and reports both stages. Final tables use every verified cohort rather than treating unknown states as failed fits.

The source closure contains11current Python files and two explicitly historical diagnostic snapshots. External research code is not copied or executed. The independent problem is a classical concave optimization with a SOC constraint,using this model's compensated profile. Its known-scale,balanced Gaussian and frozen-weight conditions accompany the results. MSE/coverage improvements and broad estimator superiority are not established. The selected illustration retains a stronger labelled-only control.
