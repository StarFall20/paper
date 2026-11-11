# Independent-training label-budget diagnostic

This is a separate, fully simulated follow-up to the frozen manuscript package
`submission/finite_choice_20261110`. It does not establish publication readiness.

See [the Chinese assessment](训练成本与投稿主张复核_20261111.md) for the question,
all planned conditions, retained failures, interpretation and source reading levels.

From this directory, install `requirements.txt` in a separate environment, then run:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python work/training_budget_diagnostic_20261111.py --out outputs/training_budget_diagnostic_20261111
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python work/verify_training_budget_20261111.py
```

The first command regenerates all 800 draws and 4,800 method rows. The second
reconciles every row and summary, independently certifies all 397 ordinary-logit
separation cases, and reconstructs eight selected designs. It is not an independent
repeat of all 800 model draws. Original raw numerical support labels remain in the
trial CSV; the separation CSV certifies their ordinary-logit boundary cases.

All trial rows are statistical simulations with known truth. No private questionnaire
data, downloaded research datasets, participant identifiers, model API credentials,
or generated questionnaire answers are included. Copied dependency modules retain
their original source. Their data-loading entry points are not run by this audit.

The plan was fixed before the main simulation. The included script and plan hashes
match the completed run. Every file except this manifest is listed in `MANIFEST.sha256`.
Fixed powers and a fixed ridge penalty describe this diagnostic, not an optimized
end-to-end method comparison or a general dominance claim.
