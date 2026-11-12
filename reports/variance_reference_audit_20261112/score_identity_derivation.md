# Identical truth scores and different existence boundaries

12 November 2026. This is a consequence within the same compensated-logit model. The estimator and variance criterion are established. The finite-solution statement uses Theorem1 of the current manuscript.

Let X contain the fitted attributes and intercept, mu(X)=sigmoid(X^T beta0), and let g0=E[Y|X,Z] be a frozen conditional-mean prediction based on additional information Z. For s in(0,1], define

g_s=mu+s(g0-mu).

Write D(f;beta)=L_N(f;beta)-L_m(f;beta), and let alpha=tau/(1+tau). For s in[alpha,1], use the existing population variance-reference power lambda_s=alpha/s, which lies in[0,1]. All comparisons use the same labelled and auxiliary rows, choices and g0.

## Full covariance and exact finite-sample score identity

Put A=X(mu-Y), B0=X(mu-g0), H=E[XX^T mu(1-mu)] and M=E[B0 B0^T]. The conditional-mean identity gives E[A B0^T]=M and E[B0]=0. For a fixed finite parameter dimension, the score covariance at any power lambda is

H-2 lambda s M+(1+1/tau)lambda^2 s^2 M.

Thus the scalar lambda_s above minimizes every covariance quadratic form simultaneously. Its sandwich covariance is

H^(-1)[H-alpha M]H^(-1)/m,

independent of s in[alpha,1]. This is the established oracle PPI++ variance rule specialized to this conditional-mean family. No estimated-weight accuracy is assumed or inferred under growing dimension.

Logistic loss is affine in its response, giving the exact identity, for every coefficient beta,

Q_(lambda_s,g_s)(beta)=L_m(Y;beta)+alpha D(g0;beta)+(lambda_s-alpha)D(mu;beta).

For every finite sample, grad D(mu;beta0)=0. Every objective on this path has EXACTLY the same gradient at beta0. The identity is stronger than equal covariances: it holds on the same realization. Their Hessians at a general beta differ by

(lambda_s-alpha)[H_N(beta)-H_m(beta)],

where H_N and H_m are empirical mean logistic information matrices. The predictor shrinkage and compensating reference weight change empirical curvature while preserving the truth score. Population Hessians agree. These facts follow directly from the existing corrected objective; generic differences between first-order and finite-sample behavior are established in GMM and PPI research.

## A concrete growing-dimension separation

Use the manuscript's Gaussian model with gamma1.5,proxy flip error0.1,tau1 and an estimated intercept. The two powers are lambda1=0.5 for the conditional-mean oracle and lambda0.5=1 for the halfway-shrunk oracle. The direct population calculation gives critical noise-slope/label ratios0.5554089493 and0.3363154547. Ordinary logit has boundary0.3902860522. The half-shrunk prediction stays strictly inside(0,1).

For a dimension ratio strictly between0.3363154547 and0.3902860522, Theorem1 gives existence probability tending to zero for the half-shrunk reference and to one for ordinary logit and the unshrunk reference. Both prediction-assisted references have identical fixed-dimensional oracle covariance and exactly identical finite-sample truth scores. The displayed boundary values are numerical evaluations of Theorem1;100/200-node agreement is a diagnostic,not a rigorous quadrature error bound.

At zero utility signal the same construction has an independently reduced scalar formula. For gamma0,epsilon0.1 andtau1,the boundaries are0.4350669349 at(s,lambda)=(0.5,1) and0.7401252045 at(1,0.5),compared with ordinary0.5. The conditional score variance is0.68 times the human reference in BOTH cases.

## A strict inequality with rational bounds

The zero-signal example can be separated without treating numerical quadrature stability as a proof. Let f(a)=E[(H-a)_+^2] for a standard normal H. The half-shrunk unit-power boundary is1-inf_(u>=0)J(u),where

J(u)=0.13u^2+f(0.7u)+f(0.3u).

Convexity of f gives J''>=0.26. The global lower bound inf J>=J(1)-J'(1)^2/0.52 therefore supplies an upper bound for that boundary. The unshrunk half-power formula supplies a lower boundary bound by evaluating its minimized criterion at rescaled v1. Exact rational interval arithmetic yields

shrunk boundary <0.445257 <0.47 <0.5 <0.728249 <unshrunk boundary.

The script uses alternating Taylor bounds for the normal density and integral,with the normalization constant bounded through Machin's identity and integer square-root comparisons. The exact rational endpoints are saved. At noise dimension ratio0.47,the manuscript theorem therefore gives opposite existence limits for the two oracle references,while ordinary logit exists with probability tending to one. Both prediction-assisted fixed-dimensional covariance matrices equal17/25 of the human covariance,and their finite-sample truth scores agree exactly.

The boundary expressions are continuous near gamma0 by the same compact multiplier and scale controls used in Theorem1. The strict inequalities consequently persist for a nonzero neighborhood of utility signals. This continuity statement does not specify the size of that neighborhood. The gamma1.5 calculation above is a separately evaluated example.

## What this comparison resolves

The example isolates an existence issue that does not require estimating the tuning weight,charging training labels,or worsening the local score covariance. It gives a reason to check the finite corrected objective even when the oracle covariance calculation favors prediction assistance. It introduces no new estimator or recommended calibration procedure. The construction uses the true attribute-only probability as an analytical reference; a practical analyst does not know that function. Neither displayed existence gap implies a uniformly better estimator or a valid growing-dimension confidence interval.

The variance rule and its fixed-dimensional interpretation come from [PPI++,Section6.1 Proposition2](https://arxiv.org/html/2311.01453v2); its consistency and asymptotic-normality results use the stated fixed-parameter regularity. [Mani et al.,Sections3-4](https://arxiv.org/html/2505.20178v2) already establish finite-sample costs of estimating PPI++ weights for mean estimation. The present oracle-weight comparison studies availability of a logit root at growing dimension and removes weight-estimation noise from its construction. Generic differences between first-order and finite-sample behavior remain prior knowledge. This scoped comparison does not establish priority over every related theorem.

The finite matched experiment is a separate follow-up selected after this population calculation. Its protocol must be fixed before sampling and retain all methods and unresolved cases. Closest-prior equivalence and publication sufficiency remain under assessment.
