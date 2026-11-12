# An observable reference for the same corrected objective

12 November 2026. This finite-design identity extends the interpretation of the manuscript's oracle example. The loss, scalar variance convention and convex support criterion are established. The algebra below is not proposed as a new general calibration or inference method.

Fix any finite, observed coefficient vector b and put mu_b(x)=sigmoid(x^T b). The vector b may be an existing human estimate, a ridge pilot or a specified reference. Freeze b and the original prediction g during the comparison. For a positive base power a, consider powers l in[a,1] and predictions

g_l = mu_b + (a/l)(g-mu_b).

They remain in[0,1]. With D(f;beta)=L_N(f;beta)-L_m(f;beta), affinity of logistic loss in its response gives, for every coefficient beta,

Q_(l,g_l)(beta)=Q_(a,g)(beta)+(l-a)D(mu_b;beta).

Each individual loss with response mu_b has zero gradient at b. Every objective on the path therefore has exactly the same gradient at the observed b. More strongly, its separate labelled and auxiliary score rows at b are unchanged:

labelled: X{mu_b-Y+a(g-mu_b)},

auxiliary: a X(mu_b-g).

Their complete empirical covariance middle matrix is unchanged. Their Hessians differ by(l-a){H_N(beta)-H_m(beta)}. A sandwich using the actual path-dependent Hessian need not be equal; equality is asserted for the score middle, or for a reference sandwich that uses the same pooled-information bread.

For the existing pooled-information trace convention used in the manuscript, the prediction-score vector at b scales by a/l. Its labelled cross covariance scales by a/l and its pooled covariance by(a/l)^2. If the original untruncated weight a is strictly between zero and one, recomputing that same rule gives l. The existing local criterion cannot distinguish the path through its realized reference score covariance. At a=0, positive powers instead use g_l=mu_b; the prediction-score variance vanishes and the recomputed ratio can be undefined. These endpoints must be reported separately.

This statement is entirely samplewise. A fitted reference is not the true coefficient, and data reuse does not become an independent-sample theorem. The identity supplies no equality of covariance at the unknown truth, no general unbiasedness claim for fitted predictions and no valid growing-dimensional interval. The oracle theorem and its probability boundary retain their own stated assumptions.

## Exact scalar illustration

Take two labelled rows with x=(2,4), y=(1,1), two auxiliary rows with x=(1,1), a specified observed reference b=0 and original predictions(0.9,0.9,0.5,0.5). No true probability function is needed. The existing scalar weight convention gives a=15/88.

The labelled covariance between the true and prediction scores, using divisor m, is1/5. The pooled prediction-score variance, using divisor m+N-1, is44/75. The convention thus gives(1/5)/(2*44/75)=15/88. Its factors of two from the symmetrized numerator and denominator cancel.

The common gradient at b is-57/44. Since every x is positive, the upper support endpoint for the path's scalar moment is3-2l. The positive-direction recession cost is exactly

9/44-l.

The moment itself is positive throughout the path. Consequently a unique finite minimizer exists for15/88<=l<9/44. At l=9/44 the moment is on the support boundary and has no finite stationary point. For9/44<l<=1 the objective is unbounded below. All transformed predictions stay strictly inside(0,1), all realized score rows and empirical score middles agree, and the recomputed variance weight equals l at every path point. The base positive-direction recession gap is3/88; at unit power it is-35/44.

The [exact rational calculation](../../results/observable/high/exact_example/observable_rational_example.json) uses fractions and no optimizer. Its finite/boundary/exterior claims are independently checked with the existing numeric support code. This is a deliberately selected deterministic illustration, not a frequency estimate or a separate contribution from classical one-dimensional support geometry. It connects the manuscript's existence question to observable local variance calculations.

## Relation to the public-data follow-up

The50-label study uses the same human reference and original500 allocations as the retained public comparison. It has no observed existence difference on its5000 path targets. The20-label follow-up uses a fixed ridge reference and the same block permutation. It separates strictly positive interior weights from zero weights, so that undefined variance ratios cannot be presented as equivalent selected weights. Both studies are finite-design diagnostics, outside the Gaussian oracle probability theorem.
