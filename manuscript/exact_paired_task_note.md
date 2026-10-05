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
oracle repair adds the single term used by the DGP when applicable. Estimation
uses a damped Newton update with a monotone loss check.

| DGP | additive candidate | oracle-term candidate |
|---|---:|---:|
| additive | 0.06 | not applicable |
| omitted nonlinear term | 0.84 | 0.04 |
| omitted threshold | 1.00 | 0.02 |
| omitted interaction | 1.00 | 0.02 |
| random cost sensitivity | 0.04 | not applicable |

The additive null and random-cost boundary have rejection rates near the
declared 5% level. The base candidate has high power against the nonlinear,
threshold, and interaction shifts. Rejection falls back to the declared size
after the corresponding term is supplied. The repair gate now passes.

Pair-type localization is directional rather than exact. For the nonlinear
condition, rejection is 0.74 for the time shift, 0.06 for the cost shift, and
0.68 for the joint shift. For the threshold condition it is 0.10, 1.00, and
1.00. For the interaction condition it is 0.28, 1.00, and 1.00. The joint
shift changes more than one raw locus, so the interaction result cannot be
described as uniquely isolated. A four-cell factorial contrast was audited as
the next design repair. Its probability-scale mixed difference reduced
rejection after an interaction term was added, but it generated a high
rejection rate when an interaction and a nonlinear main effect coexisted. The
audit is recorded in `manuscript/factorial_contrast_note.md`; it does not
support term-level localization.

The threshold condition is especially sensitive to the number of focal tasks
that cross the hinge. A credible paper needs a better-balanced design and a
predeclared repair test before presenting the paired-task test as a method
contribution.

## Decision

The exact design is a promising independent object with a falsifiable null.
The null, power, and repair gates pass for detection. Term-level localization
remains outside the supported claim: the first-order pair labels do not isolate
a unique raw locus, and the four-cell probability contrast fails when main
effects are nonlinear. The method remains high-risk until a purpose-built
paired-task supplement and a predeclared scope limited to invariance violation
are completed.

The output is `results/exact_paired_task_benchmark.csv`; the public Swissmetro
file cannot validate this exact design because it lacks the randomized focal
pairs. The observational fallback and its orientation failure are recorded
in `manuscript/observational_equivalence_note.md`.
