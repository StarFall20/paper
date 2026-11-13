# Row weights, finite attainment and regular covariance

Start with [complete results](evidence/README.md), [derivation](evidence/theory.md), [preceding scale proof](evidence/scale_boundary_proof.md), [primary-prior comparison](evidence/closest_prior_scope.json) and [independent verification](evidence/verification/verification.json). This package uses generated model data on1000previous cohorts; it adds no respondents, questionnaire or AI answers. The reviewed57-page manuscript remains separate and unchanged.

Use Python3.12 and the pinned requirements. Run commands from this directory. Use fresh output directories; regenerated files are excluded from the archive inventory.

```sh
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python verify_package.py
python analysis/row_weight_attainment_audit.py --plan analysis/row_weight_attainment_plan_20261113.json --reference inputs/previous_radial_verified.csv --out regenerated/generation
python analysis/verify_row_weight_attainment.py --root evidence --plan analysis/row_weight_attainment_plan_20261113.json --out regenerated/verification
python analysis/summarize_row_weight_attainment.py --root evidence --out regenerated/summary
```

The generation command reconstructs all5000weighted conditions and132population rows. The independent verifier constructs the actual weighted objective and checks every condition,negative direction,source draw and population integral. It uses no effective-design classifier. The summary separately reconciles all paired transitions and local covariance-cost expansions. The source closure contains9unchanged Python files and no external research code. Public empirical pickle inputs are not shipped or needed. The optional source files containing fetch/load functions are import dependencies only; these commands never load or execute external code.

The result concerns specified known scale and frozen row weights in the same compensated loss. Finite support identity does not imply equal coefficients,high-dimensional MSE improvement or interval coverage. Generic weighted PPI is established prior. No exhaustive Scopus/Scholar search or originality/journal-readiness certificate is claimed.
