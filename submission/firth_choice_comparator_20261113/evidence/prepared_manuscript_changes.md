# Prepared changes for the next manuscript release

These passages and rows are prepared from the completed replay. They have not yet been inserted into the currently reviewed manuscript or its figures. Preserve the existing mathematical claims and every original result when integrating them.

## Section 2.2 attribution

Convex support and recession criteria are established tools for likelihood existence (Geyer, 2009). Sablica et al. (2026, Theorems 1 and 4) give overlap and estimable-span uniqueness results for latent polyhedral categorical likelihoods, together with LP detection in Section 4.1. Proposition 1 applies the support calculation to the corrected moment in (3).

## Section 3.2 version reference

Replace the explicitly labelled v2 assumption reference with Koriyama and Bellec (2025; arXiv v3, Assumption 2.5, item 2). Unit-power labelled rows additionally fail the strict-convexity condition in item 1.

## Section 5.1 additional comparator

Firth estimation is an established response to separation in small choice datasets (Kessels et al., 2019). We maximize the human log likelihood plus one half the log determinant of its expected information, using the same labelled rows and no auxiliary predictions. The three fixed starts are zero, a human ridge estimate and its negative. We retain the smallest penalized negative log likelihood among converged positive-Hessian fits. This numerical selection does not certify global optimality. Intervals use inverse expected information at the returned fit, following the conventional Wald construction (Kosmidis and Firth, 2021).

## Table 2 additional rows

| Setting | Estimator | Points | MSE | Coverage | Issued and covers |
| --- | --- | --- | --- | --- | --- |
| 20 labels oracle | Human Firth | 500/500 | 1.9494 | 0.999 | 0.999 |
| 150 labels overconfident | Human Firth | 500/500 | 0.1088 | 0.963 | 0.963 |

The Firth rows use model-information Wald intervals. The other rows retain their documented covariance conventions. All coverages average 11 marginal intervals within each repetition; they are not simultaneous coverage.

## Section 5.2 interpretation

Human-only Firth returns finite fits in all 500 repetitions of both displayed settings. With 20 labels its MSE is 1.9494 and its mean marginal interval width is 5.9394. Coverage of 0.9995 accompanies these wide intervals. With 150 labels its MSE is 0.1088 and coverage is 0.9631. On the same 500 samples, its MSE exceeds the dimension-rule DY value by 0.007746, with Monte Carlo standard error 0.000581. The comparison retains established finite-estimate alternatives when assessing the corrected objective.

## Appendix C numerical convention

For a human design X, let I(beta) be the summed binomial expected information. The Firth criterion subtracts one half log det I(beta) from the summed human negative log likelihood. The adjusted score is the usual human score plus X-transpose times h times (one half minus mu), where h contains weighted leverages. Conventional Wald intervals use I(beta) inverse, without an additional factor of sample size. Independent determinant finite differences check the score, and direct inverse-information calculations check every reported covariance. One of the 1000 designs has two verified local minima across the three starts; every initial result and all independent rechecks are retained. No profile-likelihood intervals are claimed.

## Additional reference entries

Geyer, C.J. (2009). Likelihood inference in exponential families and directions of recession. Electronic Journal of Statistics 3, 259–289. https://www.stat.umn.edu/geyer/gdor/

Kessels, R., Jones, B., Goos, P. (2019). Using Firth's method for model estimation and market segmentation based on choice data. Journal of Choice Modelling 31, 1–21. https://doi.org/10.1016/j.jocm.2018.12.002

Kosmidis, I., Firth, D. (2021). Jeffreys-prior penalty, finiteness and shrinkage in binomial-response generalized linear models. Biometrika 108(1), 71–82. https://doi.org/10.1093/biomet/asaa052
