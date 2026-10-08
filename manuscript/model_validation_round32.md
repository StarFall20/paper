# Model validation and stress tests, round 32

## Purpose

These runs test three distinct claims: (i) whether the support certificate has
the advertised rank on the declared support; (ii) whether the support-complete
score can expose directions that a hand-built dictionary misses; and (iii)
whether the score is an inferential object different from a saturated fibre
alternative. The third comparison is intentionally a boundary test.

## Structural checks

* The 3x3 candidate-precision endpoint design has candidate-summary variance
  4.00 but fibre exposure rank 0.
* The support-complete design has rank 4, the exact dimension of the four
  non-singleton-fibre contrasts, and minimum exposure eigenvalue 0.1222.
* The smart-device reflected-task pool has full candidate-fibre rank 3. The
  maximin and algebraic constructions return ranks 3 and minimum eigenvalues
  1.648 and 0.483 respectively.
* Singleton fibres, zero-mass profiles, and directions that are constant within
  a fibre are treated as structural null cases. They cannot be made visible by
  increasing the sample size.

## Monte Carlo results

The locked n=800 run used 500 replications per condition and 499 conditional
randomization draws per replication. It produced 2,500 rows.

| DGP | support-complete score rejection | saturated fibre LR rejection |
|---|---:|---:|
| null | 0.062 | 0.064 |
| aligned | 1.000 | 1.000 |
| interaction | 1.000 | 1.000 |
| hidden quadratic | 1.000 | 1.000 |
| hidden orthogonal | 1.000 | 1.000 |

The null rate is close to the 5% target given 500 replications; its Monte Carlo
standard error is about 0.010. Every non-null direction was detected at the
chosen effect size. The score and saturated LR have the same practical boundary,
which confirms that the paper should not claim a new inferential family.

A second n=400 run used 1,000 replications and 199 randomization draws. It gave
null rejection rates of 0.038 for the score and 0.043 for the saturated LR, with
1.000 rejection for all four non-null directions. The two null rates bracket the
nominal 5% level within Monte Carlo error and reproduce the same power boundary.
Its output is stored separately so the higher replication count is not mixed with
the locked n=800 result.

## Interpretation

The simulations support a narrow conclusion. A design can be highly informative
for the candidate summary and still have zero exposure to a discarded-coordinate
contrast. Once all supported fibre contrasts are included, the score detects the
same finite-support alternatives as the saturated fibre LR. The methodological
increment is consequently the **pre-outcome design certificate and repair rule**,
not a claim of superior power over the saturated alternative.

Power remains conditional on effect size, response model, fibre weights, and
sample size. A positive eigenvalue establishes structural visibility; it does not
establish a useful power level for a human DCE.
