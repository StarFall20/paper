# Round 27: corrected Swissmetro DCE model audit

## Data and preparation

The analysis uses the public Swissmetro file distributed with Biogeme. It
retains valid choices for PURPOSE 1 and 3, applies the GA discount to train
and Swissmetro costs, and keeps the stated-preference availability flags. The
result is 6,768 observations from 752 respondents. The nine `CHOICE=0` rows
and the other purpose groups are excluded before estimation. The raw-file
checksum and download record remain in `data/provenance_swissmetro_2026-10-07.md`.

## Converged candidate model

The base model has train and car alternative-specific constants and generic
time and cost coefficients. BFGS maximization uses the availability-aware
log-sum-exp likelihood and its analytic gradient. Estimates are

| term | estimate |
|---|---:|
| train ASC | −0.7012 |
| car ASC | −0.1546 |
| time | −1.2779 |
| cost | −1.0838 |

The in-sample log score is −0.78771 and accuracy is 0.67642. The respondent
grouped two-fold out-of-fold log score is −0.80455 and accuracy is 0.66903.
These are descriptive fit values, not evidence that the representation is
sufficient.

## Specification benchmark

The base model is compared with three prespecified alternatives on the same
respondent split. The quadratic model adds generic squared time and cost; the
log model adds generic `log(1+x)` terms; the alternative-specific model gives
each mode its own time and cost coefficient.

| model | LR p vs base | two-fold OOF log score | OOF accuracy |
|---|---:|---:|---:|
| base | — | −0.80455 | 0.6690 |
| quadratic | 1.14×10⁻⁵⁰ | −0.78441 | 0.6720 |
| log | 4.85×10⁻⁴⁹ | −0.78859 | 0.6714 |
| alternative-specific | 2.67×10⁻¹⁰⁹ | −0.76862 | 0.6711 |

The public DCE therefore contains predictable departures from the base
utility form. A rejection by a conventional enriched model does not identify
which raw-coordinate direction generated it. It is an external benchmark for
the proposed audit, not a validation of a randomized fibre null.

## Observational fibre fallback

The corrected script fits the base model on a training respondent fold and
matches held-out rows within respondent by their complete three-alternative
utility-difference vector. Pair construction uses a frozen covariate
orientation; it does not use the observed choices. Across tolerances 0.01,
0.02, 0.05, and 0.10 and orientations index, time, and cost, the run gives
149--2,188 pairs. All twelve respondent-cluster multiplier p-values are at
least .10.

These p-values are feasibility diagnostics only. The file does not record a
one-member-per-pair random assignment, and approximate matching changes the
conditioning event. The result cannot establish conditional sufficiency or
identify a behavioural cause.

## Inferential increment audit

The finite-support support-complete score was compared with a standard
conditional-randomization LR that adds one unrestricted profile probability for
each supported fibre member. With 200 replications, both procedures had the
same 1.000 rejection rate for a dictionary-orthogonal departure; null rates
were .030 and .045. This shows that the support-complete basis is a useful
coverage certificate, but it is not by itself a new inferential alternative to
a saturated profile LR.

The files are `results/swissmetro_model_summary_corrected.csv`,
`results/swissmetro_specification_benchmark.csv`,
`results/observational_equivalence_swissmetro_corrected.csv`, and
`results/support_complete_vs_saturated_benchmark.csv`.
