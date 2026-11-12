# Positive scales and auxiliary-budget limits

Start with the [result and complete counts](evidence/README.md), [proof](evidence/proof.md), [primary-prior comparison](evidence/closest_prior_scope.json) and [independent verification](evidence/verification/verification.json). This supplement uses generated model data. It contains no questionnaire responses, participant information or downloaded empirical inputs. The preceding main manuscript is separate.

Use Python3.12 and the pinned requirements. Commands run from this directory; regenerated outputs are excluded from the manifest. The numerical verifier needs no network or external data.

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
python verify_package.py
python analysis/verify_radial_auxiliary_budget.py --root evidence --plan analysis/radial_auxiliary_budget_plan_20261112.json --reference inputs/previous_radial_verified.csv --out regenerated/verification
python analysis/plot_radial_auxiliary_budget.py --root evidence --out regenerated/figures
```

For full generation, use a fresh output directory:

```sh
python analysis/radial_auxiliary_budget_audit.py --plan analysis/radial_auxiliary_budget_plan_20261112.json --reference inputs/previous_radial_verified.csv --out regenerated/generation
```

The verifier covers all12000reported records and10000distinct original-objective programs. The2000duplicated low-ratio records are labelled. It independently integrates all52population values and retains all paired reversals. Random cohort identities match the preceding radial study. The full generation command reproduces the120scaled-interval and12projection checks too. Runtime timestamps can differ; scientific arrays and verification tables are checked separately. Mathematical derivation, specific-theorem priority and journal contribution assessment have separate evidence requirements.
