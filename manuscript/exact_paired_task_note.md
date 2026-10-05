# Exact randomized paired-task benchmark

## Purpose

The paired-task test is the only independent innovation currently under
review. This benchmark tests its core identification claim with an exact
randomized instrument. Every respondent receives three focal pairs. Within a
pair, all alternatives receive the same raw time and/or cost shift, so the
additive candidate's utility differences are identical. Pair position is
randomized among nuisance tasks, and the experimental pair label, not the
choice, defines the contrast.

The development half of the respondents estimates the candidate model. The
held-out half supplies pair contrasts. The statistic subtracts the candidate
probability difference from the observed one-hot choice difference and uses a
respondent-cluster multiplier bootstrap. The implementation is
`analysis/exact_paired_task_benchmark.py`.

## Boundary run

The current run uses 50 replications, 300 respondents per replication, three
focal pairs per respondent, four nuisance tasks, and 199 multiplier draws. The
base candidate has train and car constants plus generic time and cost. The
oracle repair adds the single term used by the DGP when applicable.

| DGP | additive candidate | oracle-term candidate |
|---|---:|---:|
| additive | 0.06 | not applicable |
| omitted nonlinear term | 0.84 | 0.18 |
| omitted threshold | 1.00 | 1.00 |
| omitted interaction | 1.00 | 0.98 |
| random cost sensitivity | 0.04 | not applicable |

The additive null and random-cost boundary have rejection rates near the
declared 5% level. The base candidate has high power against the nonlinear
and interaction shifts. The threshold and interaction repair rows still
reject after the correct term is supplied. This means the current estimator
or pair design has not passed the repair gate. The result is useful because it
blocks an invalid claim: a rejection cannot yet be interpreted as a localized
omitted term when the repaired candidate remains rejected.

The threshold condition is especially sensitive to the small number of focal
tasks that cross the hinge. A credible paper needs a better-balanced design,
more varied threshold crossings, a stable MNL optimizer, and a predeclared
repair test before presenting the paired-task test as a method contribution.

## Decision

The exact design is a promising independent object with a falsifiable null,
but it is not submission-ready. The main manuscript must not claim that it
localizes omitted mechanisms until the repair gate passes. If the repair gate
continues to fail after design and optimizer fixes, this innovation is
dropped and the paper returns to a narrower recoverability benchmark.

The output is `results/exact_paired_task_benchmark.csv`; the public Swissmetro
file cannot validate this exact design because it lacks the randomized focal
pairs. The observational fallback and its orientation failure are recorded
in `manuscript/observational_equivalence_note.md`.
