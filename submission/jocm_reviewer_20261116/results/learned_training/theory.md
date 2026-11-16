# A worked independently trained prediction case

This note instantiates the current manuscript's conditional training result. It does not propose a new learner or a general teacher–student theory. The empirical signed-label feature mean is a classical Hebbian construction. Saglietti and Zdeborova (2022, equations 13 and 23–24) already use such a mean and its signal-plus-Gaussian-noise representation in a different Gaussian-mixture distillation model. The present observations follow elementary conditional Gaussian calculations and laws of large numbers.

## Fixed resources and observed information

Let the total human label budget be \(B\), the training allocation be \(t/B\to\alpha\in(0,1)\), and the correction allocation be \(m=B-t\). Hold \(N/B\to\nu>0\) fixed. Let the number of Gaussian attribute slopes satisfy \(d/B\to\rho>0\). Both training and audit attribute rows are independent standard Gaussian vectors. The human choice follows

\[
P(Y=1\mid X)=\sigma(\gamma X_1),\qquad |\gamma|<\infty.
\]

An observed binary proxy \(Z\) flips the choice with probability \(\epsilon=1/4\), independently conditional on \(Y\). This is a statistical model for additional prediction information; no human or LLM answers were collected. The proxy is available on both training and auxiliary rows. The true auxiliary choices used to define the simulation's joint law are hidden from the learner and the corrected objective.

Train the coefficients by

\[
\widehat v_X=t^{-1}\sum_{i=1}^t(2Y_i-1)X_i,\qquad
\widehat v_Z=t^{-1}\sum_{i=1}^t(2Y_i-1)(2Z_i-1).
\]

Freeze the prediction \(g(X,Z)=\sigma(X^T\widehat v_X+\widehat v_Z(2Z-1))\). The intercept is zero. No true utility coefficient or conditional choice probability enters this training rule. Direct human logit receives the same union of all \(B\) human labels.

## Explicit training limits

Write \(a_t=(\widehat v_X)_1\) and \(b_t^2=\sum_{j=2}^d(\widehat v_X)_j^2\). Then

\[
a_t\ \longrightarrow\ a_\gamma
 =E[S\tanh(\gamma S/2)],\qquad
\widehat v_Z\ \longrightarrow\ 1-2\epsilon,\qquad
b_t^2\ \longrightarrow\ \rho/\alpha
\]

in probability. The first and second statements are ordinary laws of large numbers. Conditional on the training signal coordinates, choices and proxies, each remaining coefficient is independent \(N(0,1/t)\). Consequently \(t b_t^2\) has exactly a chi-square distribution with \(d-1\) degrees of freedom. This proves the third convergence and closes the training-convergence condition for this particular rule. It establishes no such convergence for an arbitrary learning algorithm.

Conditional on training, rotate the noise coordinates along their fitted coefficient direction. A fresh audit row has independent standard normal coordinates \(S,T\) and the remaining Gaussian columns. Its prediction is exactly

\[
g_t(S,T,Z)=
\sigma\{a_t S+b_t T+\widehat v_Z(2Z-1)\}.
\]

The limiting prediction is

\[
g_{\rho,\alpha}(S,T,Z)=
\sigma\{a_\gamma S+\sqrt{\rho/\alpha}\,T+(1-2\epsilon)(2Z-1)\}.
\]

The fitted utility includes the \(T\) direction. Treating it as external to the utility would give the wrong cone constraints. The remaining fitted Gaussian column count is \(d-2\); this fixed difference does not change the proportional ratio.

## Existence at the total label budget

Apply Proposition 3 and the conditional-training argument of Appendix E2, retaining \(Z\) in the conditioned tuple. The dimension and auxiliary ratios entering the compensated boundary are

\[
\kappa=\frac{\rho}{1-\alpha},\qquad
\tau=\frac{\nu}{1-\alpha}.
\]

For every fixed positive power, the strict existence sides are determined by comparing \(\kappa\) with the existing boundary evaluated at \(g_{\rho,\alpha}\). Its dependence on \(\rho\) through the training noise must be retained. This is an implicit budget condition, not a claim that there is always a unique crossing in \(\rho\).

The same convergence proof applies to the existing all-positive-power envelope. A ratio above that envelope excludes a finite estimate for every power selection simultaneously, including zero via its ordinary audit-label separation face. Below it, strictly positive powers tending to zero recover the stated existence limit. This does not make those estimates accurate or establish valid growing-dimensional intervals.

At \(\gamma=1.5,\rho=0.3,\nu=1\), the computed envelopes are 0.477758 for \(\alpha=0.25\) and 0.510186 for \(\alpha=0.5\). The required correction ratios are 0.4 and 0.6. Thus the second allocation lies above the all-power envelope, even though its trained predictor has smaller limiting classification error, 0.292939 versus 0.334517. The direct human ratio remains 0.3, below its ordinary boundary 0.390286. The numerical envelope values have independent normal-integral evaluations; they are not rational interval certificates.

These allocations use the same total human and auxiliary resource counts. Their simulation draws are independent across allocation conditions; only the within-condition power/direct-human comparisons are paired.

## An elementary necessary budget bound

For the frozen limiting prediction put \(r=|Y-g|\), \(R_1=E r\) and \(R_2=E r^2>0\). Evaluate the existing limiting-cone variational objective at zero nuisance multiplier. Its value at scale zero is \(1/2\), its right derivative is \(\sqrt{2/\pi}(1-R_1)\), and its second derivative for nonnegative scale is at most \(-R_2\). The labelled lower-half-line term supplies this curvature; the auxiliary term is concave. Hence

\[
\kappa_\infty
\leq \frac12+\frac{(1-R_1)^2}{\pi R_2}.
\]

This is a necessary budget bound obtained by restricting an existing variational formula and applying a quadratic upper bound. It is not a new separation theorem. In the worked case, it can screen extremely small correction allocations. It is weaker than the computed envelope and does not exclude the 50% training allocation by itself. Replacing \(R_1,R_2\) by empirical estimates would require additional uncertainty analysis before claiming a guarantee.

## The attributes-only negative control

Removing \(Z\) gives the same explicit training limit with no proxy term. Under the correctly specified human model, any frozen predictor measurable only in fitted \(X\) satisfies

\[
E[X(\mu-Y)\{X(\mu-g)\}^T]=0.
\]

The regular corrected-score covariance therefore equals the human covariance plus a positive-semidefinite prediction-score term proportional to \((1+1/\tau)\lambda^2\). Its variance-optimal regular power is zero, apart from degenerate ties. Positive fixed powers in this control are diagnostics, not an efficiency recommendation. Its 4000 simulated cohorts are preserved separately.

With the observed proxy, conditional covariance of \(Y\) and \(g\) at fixed \(S,T\) is

\[
\mu(1-\mu)(1-2\epsilon)(g_+-g_-)>0,
\]

where \(g_\pm=\sigma(a_\gamma S+\sqrt{\rho/\alpha}T\pm(1-2\epsilon))\). The proxy-informed case has a genuine additional score-information channel. The finite positive powers examined here are still fixed diagnostics, not a claim of optimal estimated tuning or a superior estimator.

## Contribution and scope

The training mean, its overlap law, Gaussian rotation, support LP and all-power envelope retain their established attribution. The useful completion is an explicit, fully trained case of the same compensated objective, with both human and auxiliary budgets charged. The result qualifies recommendations to spend labels improving a predictor. Its importance for the target journal remains a contribution assessment; this note alone is not proof of an independent major innovation.

The main 61-page manuscript and its reviewed Word files remain unchanged in this follow-up. Empirical transport, clustered choices, final estimator prediction quality, confidence interval validity and optimal allocation are outside this calculation.

Reference: Saglietti, L. and Zdeborova, L. (2022). Solvable Model for Inheriting the Regularization through Knowledge Distillation. PMLR 145, 809–846. https://proceedings.mlr.press/v145/saglietti22a.html
