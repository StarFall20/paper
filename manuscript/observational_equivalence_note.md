# Observational matched-task fallback: evidence and limits

## Why this branch exists

The preferred innovation is an exact randomized pair of choice tasks whose
candidate-model utility-difference vectors are equal. The public Swissmetro
file does not contain a balanced block of such tasks. This note records the
weaker fallback so that the empirical limitation is measurable rather than
hidden.

The fallback freezes a candidate model on development respondents, forms
within-respondent pairs among held-out tasks using two utility-difference
coordinates, and evaluates the choice contrast on the held-out labels. Pair
construction uses alternative attributes, availability, and frozen
coefficients. It never uses the observed choice. An approximate pair is
residualised by the candidate probability difference. The test statistic is a
respondent-level Wald norm of the three alternative contrasts, with a
Rademacher multiplier bootstrap over respondent clusters.

The implementation is `analysis/observational_equivalence_test.py`. The
Swissmetro output is `results/observational_equivalence_swissmetro.csv` and
the simulation output is `results/observational_equivalence_simulation.csv`.

## Simulation boundary check

The simulation contains one exact focal pair per respondent, randomises its
position among nuisance tasks, and uses 400 respondents. The nonlinear and
interaction coefficients are deliberately visible so the test has a fair
power check; they are not calibrated estimates for Swissmetro. With 40
replications and 149 multiplier draws per replication, the rejection rates
were:

| condition | rejection rate | mean matched pairs |
|---|---:|---:|
| additive null | 0.050 | 403.3 |
| omitted nonlinear term | 0.225 | 400.3 |
| omitted interaction | 0.175 | 400.7 |
| respondent random cost sensitivity | 0.050 | 404.1 |

The null size is close to 5% in this run. Power is modest for the
observational statistic even with an engineered focal pair. The random-cost
condition does not reject because the paired transformation preserves the
linear cost differences for every fixed cost coefficient; this is a useful
boundary. A violation cannot be interpreted as generic heterogeneity without
additional pair designs that change the relevant omitted locus. The exact
randomised binary prototype remains the stronger feasibility evidence, with a
0.06 null rejection rate and 0.986 power against its declared quadratic
alternative in `results/equivalent_pair_test.csv`.

## Swissmetro audit

The public Biogeme file has 10,728 rows, 1,192 respondents, and nine choice
situations per respondent. With pair orientation defined by original row order,
the strict cross-fitted fallback produced:

| coordinate tolerance | pairs | respondents with a pair | p-value |
|---:|---:|---:|---:|
| 0.01 | 126 | 60 | 0.427 |
| 0.02 | 236 | 106 | 0.440 |
| 0.05 | 1,371 | 474 | 0.430 |
| 0.10 | 3,430 | 901 | 0.477 |

The orientation sensitivity is decisive. Using the same pairs, ordering the
tasks by total travel time gives p-values 0.430, 0.387, 0.003, and 0.033 at
the four tolerances. Ordering by total cost gives 0.130, 0.460, 0.847, and
0.977. The sign convention is label-free in each case, yet the result changes
sharply. This is evidence that the observational fallback is confounded by
task order, attribute geometry, or approximate-match error. It cannot support
a specification claim on Swissmetro.

The p-values are descriptive. At 0.01 and 0.02 the sample is too small for a
strong test. At 0.05 and 0.10 the number of pairs increases while the equality
restriction weakens, and the result depends on the declared orientation. The
public file therefore cannot supply the main empirical validation of the
paired-task innovation.

## Overlap and novelty boundary

Fok and Paap (Journal of Choice Modelling, 2025) already develop composite-
likelihood and GMM overidentification tests from binary alternative pairs.
Those tests use pairs of alternatives within a choice set to diagnose the MNL
and IIA restrictions. The present pivot uses pairs of *choice tasks* with
different attribute decompositions and equal candidate utility differences.
Its null is sufficiency of the candidate utility basis across tasks, not the
binary-pair implication of IIA. Utility-neutral design and invariance papers
also provide adjacent precedents. The manuscript must cite these boundaries
and must not claim to be the first misspecification test.

The candidate remains high-risk until a systematic literature search and a
purpose-built paired-task instrument confirm that the task-level equality is
not already established under another name. If the exact instrument cannot be
fielded, this fallback is a negative feasibility result and the paper should
return to the narrower recoverability benchmark.

## Current decision

The fallback is retained as an audit and a reproducible limitation. It is not
promoted to the main empirical contribution. The main pivot still requires an
exact randomized supplement or a different public dataset, plus a larger
simulation with threshold, interaction, process, and candidate-basis repair
conditions.

## References used for the boundary audit

- Fok and Paap, *New misspecification tests for multinomial logit models*,
  Journal of Choice Modelling 54 (2025):
  https://doi.org/10.1016/j.jocm.2024.100531
- JOCM aims and scope: https://shop.elsevier.com/journals/journal-of-choice-modelling/1755-5345
- Olschewski, *Empirical underidentification in estimating random utility
  models*: https://doi.org/10.1111/bmsp.12256
- Latent utility and permutation invariance:
  https://doi.org/10.1016/j.jeconom.2024.105844
