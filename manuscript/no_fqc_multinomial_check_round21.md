# Multinomial full-menu implementation gate

## Purpose

The original binary fibre check did not establish the multinomial condition.
This gate uses eight three-alternative reflected arms. For arm (g), the
candidate vector (b_g) is identical in the two orientations, so the complete
pairwise vector

\[
\phi_g=(b_{g1}-b_{g2},\ b_{g1}-b_{g3},\ b_{g2}-b_{g3})
\]

is unchanged. The maximum plus/minus deviation is exactly zero in all eight
arms.

The response is represented by two independent contrasts relative to the third
alternative. Let (C=[(1,0,-1);(0,1,-1)]). The local departure and declared
nuisance columns are computed from the full probability vector before applying
(C). The score uses the signed one-hot outcome from the randomized
orientation. Under the candidate null its block covariance is

\[
\Omega_g=4C\operatorname{diag}(p_{0g})C^\top,
\]

because the orientation coin makes the signed mean zero. This is the correct
metric for the implemented score; it retains the covariance across the two
independent menu contrasts. A paired-difference estimator uses the equivalent
centered multinomial covariance with a different known multiplier.

## Certificate

The eight-arm block has residual rank 3 and residual information eigenvalues

\[
 1.202888,quad 13.788401,quad 15.480000.
\]

The nuisance-orthogonal score has

\[
\max_j |G_j^\top w_\star|=3.03\times10^{-18}.
\]

These are pre-outcome design quantities. The exact menu-preservation table is
in `results/no_fqc_multinomial_design.csv`.

## Planning benchmark

The benchmark uses 200 replications, 600 respondents, and 499 respondent-level
sign flips. The statistic is the same projected score in every condition; the
raw comparator uses the unprojected candidate-departure score.

| condition | projected | raw |
|---|---:|---:|
| null | .025 | .080 |
| omitted residual | 1.000 | 1.000 |
| declared scale | .030 | .545 |
| declared framing | .040 | .145 |
| strong declared framing | .045 | .835 |
| combined residual and nuisance | 1.000 | 1.000 |
| unlisted process | .470 | .965 |

The projected statistic stays near the nominal level for declared nuisance
directions and retains power for the declared residual. The unlisted process
is deliberately detectable after projection; this is a boundary result, not a
claim that the nuisance library is complete.

## What this proves

This gate proves that the method can be implemented with a full multinomial
menu vector, a finite covariance metric, and a nuisance-orthogonal score. It
does not prove the sufficiency relation in human data. The empirical version
must preserve every candidate menu coordinate, record the assignment coin, and
report usable within-fibre support.
