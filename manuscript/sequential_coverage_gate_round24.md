# Sequential coverage design gate, round 24

## Candidate transferred from adaptive DCE design

Recent sequential DCE work updates priors or designs to improve parameter
precision. Pérez-Troncoso's JOCM study compares update frequencies for that
objective, and Mao, Kessels and van der Zanden use simulated annealing to
improve Bayesian D-efficient construction. Adaptive conjoint work also
distinguishes population from individual targets. These papers establish the
nearest method boundary:

- [Pérez-Troncoso (2022)](https://doi.org/10.1016/j.jocm.2022.100357)
- [Mao, Kessels and van der Zanden (2025)](https://doi.org/10.1016/j.jocm.2025.100551)
- [Optimal adaptive Bayesian design in choice experiments (2025)](https://doi.org/10.1007/s11002-025-09768-4)

The proposed transfer was narrower: use a first wave to estimate which
candidate-preserving fibre is under-exposed, choose the second-wave fibre from
that estimate, and perform inference only on an independently randomized
second wave. This would target coverage of residual directions rather than
precision of candidate coefficients.

## Falsification protocol

The finite support uses $\phi(a,b)=a+b$, residual features $(a-b,ab)$, a
240-respondent pilot, and a 240-respondent held-out wave. The adaptive rule
selects the fibre with the largest estimated residual separation. The static
comparator preselects the central fibre. The null test uses the second-wave
assignment only, so pilot-driven selection cannot enter the reference
distribution.

The check is a planning gate. It is designed to fail if adaptation merely
repackages the existing support criterion or if it sacrifices a weak direction.

## Gate result

Across 500 replications:

| condition | adaptive rejection | static central-fibre rejection | adaptive fibre 1 / 2 / 3 |
|---|---:|---:|---:|
| null | .040 | .074 | 0.00 / 1.00 / 0.00 |
| $h_1=a-b$ | 1.000 | 1.000 | 0.00 / 1.00 / 0.00 |
| $h_2=ab$ | .048 | .026 | 0.00 / 1.00 / 0.00 |
| mixed | 1.000 | 1.000 | 0.00 / 1.00 / 0.00 |

The adaptive rule selected the same fibre in every replication because the
declared finite support makes that fibre dominate the separation score for both
directions. It produced no power gain over the static design and did not
change the structural coverage rank. The method-transfer candidate is
therefore rejected as a primary contribution.

## What the failure teaches the main paper

Sequential updating is useful only when the feasible support contains genuinely
different coverage profiles and the update rule changes which directions can
be exposed. A larger algorithm cannot create a missing fibre. The main paper
should keep the static rank certificate and report sequential adaptive coverage
as future work, conditional on a richer support, an explicit respondent-burden
cost, and an exact split-sample randomization reference.

The reproducible gate is in
`analysis/sequential_fibre_coverage_gate.py`, with results in
`results/sequential_fibre_coverage_gate.csv`. The normal-tail calculation is a
planning diagnostic; a final adaptive DCE would require the preregistered
cluster randomization statistic and an independent validation wave.

## Decision

This candidate does not replace fibre-conditional sufficiency. Its negative
result strengthens the anti-stitching argument: importing an adaptive-DCE
algorithm without a new estimand does not constitute an innovation.
