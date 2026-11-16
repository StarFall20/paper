# Independent discrete profiles in compensated choice logit

This 16 November follow-up extends the scope of the existing model. Its [derivation](theory.md) uses Oymak–Tropp's established matrix theorem and the classical pivot-coordinate rank argument. It states the independent-entry conditions and proves the interior-probability and centered-rank bridges. It does not introduce a new estimator or claim a new general universality law. Parent-manuscript integration is tracked separately.

All 1000 original Gaussian model cohorts are reused. Human choices, prediction categories, signal profiles, dimensions and powers remain unchanged. Each raw noise driver supplies Gaussian, symmetric two-level and symmetric three-level profiles **before** within-menu centering. These are 3000 coupled design variants, not 3000 independent cohorts. The [54 cells](results/summary.csv), [36 paired differences](results/paired_with_gaussian.csv) and [six-row table](summary/tableF4.md) retain every result. No human, AI or new independent model cohort is added.

At 80 labels and 52 noise slopes, finite corrected estimates number 269, 274 and 276 out of the same 500 cohorts under the three laws. Finite-budget differences reach 3.8 percentage points across the retained comparisons. The asymptotic transfer result does not assert equal finite-sample frequencies, discrete-profile prediction risk or interval validity. Dependent effect coding, feasibility-restricted or fixed efficient designs, panels, cross-fitting and predictions using the remaining attributes are outside the stated transfer conditions.

An independent verifier reconstructs all profiles, checks every 8652 exterior recession direction, every 3287 ordinary separator and 322 fixed-margin interior programs. One saved direction exceeds its nominal unit-L1 constraint by 1.75e-7; the verifier divides every direction by its positive norm when necessary and checks the correspondingly rescaled gap. The initial failure and explicit repair remain in the package. Primary classifications and directions are unchanged.

Use Python 3.12 and the recorded packages in requirements.txt. Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. Run from this directory:

```sh
python verify_package.py
python run_reproduction.py --out regenerated
python run_reproduction.py --out regenerated_full --full
```

The default workflow assembles the exact saved records and verifies them. The full workflow reruns every one of the 27000 classifications from the existing seed, then performs the same checks. A fresh output directory is required. The split checkpoint and compressed CSV are lossless storage only. Raw questionnaire data, participant identifiers and downloaded article full texts are excluded. Numerical agreement is not independent proof review, a priority certificate or evidence that the journal's contribution threshold is satisfied.
