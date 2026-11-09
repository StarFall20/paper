# Candidate result: finite solutions of a rectified binary-choice loss

Working derivation, 9 November 2026. **The mathematical candidate is under verification; independent novelty and submission readiness remain unproven.** This is a new question arising from the observed pseudo-label clipping. The former fibre/rank framework remains retired.

The finite-design convex-support criterion is classical. Unbiased corrected risks can also fail to have finite minimizers in other literatures. The candidate increment assessed here is narrower: an exact finite-label existence probability and a calibration-dependent proportional-dimensional boundary for an infinite-auxiliary binary-logit estimator. A targeted search has not yet established priority.

## 1. A single estimation problem

Let independent labelled observations satisfy

\[
X_i\sim N_p(0,I_p),\qquad Y_i\sim\operatorname{Bernoulli}(1/2),
\quad X_i\perp Y_i,\quad i=1,\ldots,m.
\]

The choice model includes an intercept and p nonconstant contrasts:
\(P(Y=1\mid X)=\sigma(\alpha+X^\top\beta)\). Its true parameter is zero.
The frozen auxiliary predictor is the constant \(q\in[0,1]\), independent of the labelled data. An unlimited independent auxiliary pool supplies the population imputed loss. The unweighted rectified loss of the standard PPI form is

\[
Q_m(\alpha,\beta)=
E_X\ell(\alpha+X^\top\beta,q)
+\frac1m\sum_i\{\ell(\alpha+X_i^\top\beta,Y_i)-\ell(\alpha+X_i^\top\beta,q)\},
\]

where \(\ell(u,y)=\log(1+e^u)-yu\). Therefore

\[
Q_m=A(\alpha,\beta)-\bar Y\alpha-S_q^\top\beta,
\qquad A=E\log(1+e^{\alpha+X^\top\beta}),\quad
S_q=m^{-1}\sum_iX_i(Y_i-q).
\tag{1}
\]

This is an unbiased risk estimate. No statement here concerns adaptive power tuning, arbitrary predictors, clustered labels or a general multinomial menu.

## 2. Attainable moments

Write \(\phi\) and \(\Phi\) for the standard normal density and distribution function, and define
\(r(t)=\phi\{\Phi^{-1}(t)\}\) for \(0<t<1\), with \(r(0)=r(1)=0\).

The closed attainable-moment set is the classical Gaussian lift zonoid, explicitly described in [Borell (2009), Theorem 4.3](https://alea.impa.br/articles/v6/06-05.pdf). Its specialization here is

\[
\mathcal C=\{(t,s):0\leq t\leq1,\ \|s\|\leq r(t)\}.
\tag{2}
\]

To establish (2), take any measurable \(h(X)\in[0,1]\) with \(Eh=t\). For a fixed unit vector v, the greatest \(E[v^\top Xh(X)]\) is attained by selecting the upper t-fraction of \(v^\top X\), yielding r(t). Rotation gives the norm bound. Conversely, mix this threshold rule with the constant t to realize every radius from zero to r(t), in any direction. The end slices contain only zero.

For \(w=(a,b)\), the support function is \(h_{\mathcal C}(w)=E(a+X^\top b)_+\), and
\[
h_{\mathcal C}(w)\le A(w)\le h_{\mathcal C}(w)+\log2.
\tag{3}
\]

Consequently, (1) has a unique finite minimizer if and only if
\[
0<\bar Y<1,\qquad\|S_q\|<r(\bar Y).
\tag{4}
\]

Proof: an interior target has a positive support gap on the unit sphere, so (3) makes Q coercive. Its Hessian is positive definite at every finite parameter. An exterior target has a separating ray along which Q tends to minus infinity. A boundary target has no finite stationary point: logistic probabilities lie strictly between zero and one and cannot attain an extremal threshold moment. Thus a boundary infimum is unattained. These are convex-support arguments, not a new generic existence theorem.

## 3. Exact finite-label probability

Let \(K=\sum_iY_i\). Conditional on \(K=k\),

\[
S_q\sim N_p\left(0,\frac{k(1-q)^2+(m-k)q^2}{m^2}I_p\right).
\]

For \(p\ge1\), (4) gives the candidate exact formula

\[
P\{\text{finite minimizer}\}=
\sum_{k=1}^{m-1}\binom mk2^{-m}
F_{\chi_p^2}\!\left(
\frac{m^2r(k/m)^2}{k(1-q)^2+(m-k)q^2}
\right).
\tag{5}
\]

The intercept explains the omission of k=0,m. Formula (5) does not require a Gaussian approximation to the labelled score; its conditional Gaussian law is exact.

## 4. Proportional dimension

If p,m tend to infinity with p/m tending to \(\kappa>0\), then

\[
\bar Y\to\tfrac12,\qquad
\|S_q\|^2\to\kappa\{\tfrac14+(q-\tfrac12)^2\}.
\]

The candidate critical ratio is

\[
\kappa_c(q)=\frac{2}{\pi\{1+4(q-1/2)^2\}}.
\tag{6}
\]

Below this boundary the probability of a finite minimizer tends to one; above it the probability tends to zero. No boundary-equality limit is asserted. The limiting boundary is 2/pi for q=1/2 and 1/pi for q=0 or1. Thus constant auxiliary predictions with larger Brier error can make existence harder, even with unlimited auxiliary covariates and an unbiased correction.

For comparison, the ordinary null-model logistic MLE with an intercept has the classical finite-sample probability \(1-2^{-(m-1)}\sum_{j=0}^{p}\binom{m-1}j\), and critical ratio1/2. This comparison belongs to [classical geometric probability](https://doi.org/10.7146/math.scand.a-10655) and [logistic separation theory](https://arxiv.org/abs/1804.09753); it is not a new result here.

## 5. Nonzero signal and frozen low-index predictions

A broader candidate boundary follows from the same estimation problem. Let
\(P(Y=1\mid X)=\sigma(\gamma X_1)\), with fixed \(\gamma\ge0\). Let the frozen predictor g be any bounded function of a fixed number of Gaussian projections, whose span contains the signal direction. The predictor is independent of the labelled sample. Define

\[
a(\gamma)=E[G\sigma(\gamma G)],\quad c=\phi(0),\quad
R_g=E\{Y-g(X)\}^2>0.
\]

In the population-auxiliary corrected loss, the target moment is
\(E[Wg(X)]+m^{-1}\sum_iW_i\{Y_i-g(X_i)\}\), where \(W=(1,X)\).
Its intercept converges to1/2. In the fixed predictor/signal subspace the slope moment converges to \(a(\gamma)e_1\). Conditional on the labelled residuals and this subspace, each remaining Gaussian coordinate of the correction is independent normal with variance \(m^{-2}\sum_i\{Y_i-g(X_i)\}^2\). Consequently its squared norm converges to

\[
a(\gamma)^2+\kappa R_g.
\]

The candidate nonnull boundary is therefore

\[
\kappa_c(\gamma,g)=\frac{c^2-a(\gamma)^2}{R_g}.
\tag{7}
\]

This requires a fixed-dimensional predictor subspace; a predictor trained on all growing coordinates of the same labels is not covered by this argument. The real-data LP check does not rely on this Gaussian restriction.

Several cases make (7) testable. For a constant q, \(R_g=1/4+(q-1/2)^2\). For the probability oracle \(g=\sigma(\gamma G)\), \(R_g=b(\gamma):=E[\sigma(\gamma G)\{1-\sigma(\gamma G)\}]\), with \(a(\gamma)=\gamma b(\gamma)\) by Gaussian integration by parts. For its hard threshold \(g=1\{G>0\}\), \(R_g=2E[1\{G>0\}\sigma(-\gamma G)]\). For the reversed probability, \(R_g=1-3b(\gamma)\).

The probability oracle and its hard threshold have the same classification boundary, but different Brier risks and candidate existence regions. This is the proposed practical distinction. It is not asserted as new until the relevant corrected-risk and high-dimensional literature has been checked. The conditional-mean optimality of Brier risk itself is classical.

## 6. A large finite auxiliary pool

Let the independent auxiliary pool contain N covariates, W=(1,X), and replace the first expectation in Q by its auxiliary average. Write C_N for its empirical attainable-moment zonotope. Its support function is the average of (W'w)_+. Symmetrization and the contraction inequality for the 1-Lipschitz positive-part map imply

\[
E\sup_{\|w\|\le1}|h_{C_N}(w)-h_C(w)|
\le C\, E\left\|N^{-1}\sum_{j=1}^N\epsilon_j W_j\right\|
\le C\sqrt{(p+1)/N}.
\]

The constant C is universal; the Rademacher signs are independent. The last inequality follows from Jensen's inequality and E||W||²=p+1. The Euclidean Hausdorff distance between compact convex sets equals the uniform support-function distance, so this bounds their distance in probability. Also, for any frozen g in [0,1],

\[
E\left\|N^{-1}\sum_j W_jg(X_j)-E[Wg(X)]\right\|^2
\le(p+1)/N.
\]

Consequently both set and target-moment errors vanish when N/p tends to infinity. Under either strict inequality in (6) or (7), the population target is separated from the Gaussian boundary by a dimension-independent positive Euclidean distance with probability tending to one: its intercept tends to 1/2 and its slope norm tends to a constant strictly below or above c. An interior target remains in the interior of C_N by the support inequalities; an exterior target retains a separating direction. Gaussian auxiliary designs have full column rank almost surely once N≥p+1. The finite-pool loss therefore has the same strict-side existence limit. This argument uses an independent auxiliary pool; it is not an exact finite-N formula or a result for fitted same-label predictors. Formula (5) remains a population-auxiliary statement.

## 7. Scope and practical implications

The [PPI family](https://arxiv.org/abs/2311.01453) already supplies corrected risks and confidence-set constructions. Fixed-dimensional guarantees are compatible with this candidate proportional-dimensional result. A missing finite point estimate does not invalidate confidence-set inversion. The contribution cannot be described as a refutation of PPI.

In a finite categorical DCE, one can check the classical mean-support condition directly with a linear program. The Gaussian formula must not be used as a universal sample-size rule for that design. The released vaccine data also use different, fitted nuisance predictions and a nonnull target.

Adding a positive quadratic penalty on all parameters yields a finite penalized estimate, while changing the estimating equation. An exterior corrected moment has no finite unpenalized root. Clipping pseudo-labels changes the equation too; for labelled one-hot outcomes it removes the inverse-propensity correction exactly. A finite solver output therefore needs both an objective definition and a score check.

## Verification status and remaining originality work

- Direct original-X,Y simulation is complete: 160 exact-probability cells and 480 signal/predictor cells, 2000 repetitions each. Nine interior population roots and six exterior rays were checked by numerical integration. These checks support implementation and algebra, not priority.
- Separate the classical support argument from the exact new candidate boundary; inspect semi-supervised, corrected-risk and high-dimensional likelihood literature for an equivalent result.
- The fixed-target500-allocation vaccine audit is complete. All seven exterior cases across the two corrected estimators have a separate positive-gap separating-ray check. No exterior case was silently removed from failure counts.
- Assess substantive JOCM value and current human-application scope. None of these requirements is fulfilled by a green checksum or a self-review score.
