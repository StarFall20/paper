# Direct substitution into the nearest row-class formula

Retain the manuscript's compensated logit. Multiplying the objective by the human sample size gives labelled and auxiliary scalar losses

$$l_L(u)=(1-\lambda)\log(1+e^u)-(Y-\lambda g)u,\qquad l_U(u)=\frac{\lambda}{\tau}\{\log(1+e^u)-gu\}.$$

Here 0<λ<1 and 0<g<1. Differentiation gives

$$l'_L(u)=(1-\lambda)\sigma(u)-Y+\lambda g,\qquad l'_U(u)=\frac{\lambda}{\tau}\{\sigma(u)-g\}.$$

Every labelled row with Y=0 is strictly increasing; every labelled row with Y=1 is strictly decreasing. Every auxiliary row is coercive. All these rows are strictly convex. Their categories are independent of λ throughout (0,1), for a fixed predictor. Each labelled infimum is minus infinity. These are direct algebraic properties of the specified loss.

## A strict-power population pair

Let Y be balanced, let Z independently flip Y with probability e=1/4, and freeze the calibrated predictor g=1/4+Z/2. Keep the same Gaussian attribute and choice law, fitted intercept and τ=1 at both powers λ=1/4 and λ=3/4. The labelled/auxiliary category vector and its joint law with the signal are identical at both powers. Both powers are strictly below one, so no affine row is used.

The manuscript's balanced reduction gives κλ=sup(v≥0) Jλ(v), where

$$J_\lambda(v)=2-(1-e)\{f(-ev)+f((\lambda^{-1}-1+e)v)\}-e\{f(-(1-e)v)+f((\lambda^{-1}-e)v)\}-f((1-e)v)-f(ev).$$

The registered calculation supplies exact rational enclosures around

$$\kappa_{1/4}=0.6624600373302011\ldots,\qquad\kappa_{3/4}=0.4841966704264584\ldots.$$

Since Jλ''≤−e(1−e), a feasible value Jλ(v) and Jλ(v)+Jλ'(v)²/[2e(1−e)] enclose the global optimum. Strict derivative brackets isolate the unique maximizer. The separate original-distance implementation verifies both enclosures and the strict inequalities

$$\kappa_{3/4}<11/20<\kappa_{1/4}.$$

Thus the existing main theorem gives opposite strict existence limits at q/m→11/20, under the same predictor and data law. Any proposed boundary depending only on the joint law of signal and row categories must be identical for this pair. That statistic is insufficient for this class of compensated losses.

This establishes a failure of extending a category-only formula after dropping its finite-row-infimum hypothesis. It does not contradict the original formula or exclude other general theorems. It uses the manuscript's existing main theorem; it is a comparison corollary and not a new independent proof of that theorem.

## Why the finite-infimum assumption matters

The signed recession rates here are

$$l_L^\infty(d)=(1-\lambda)d_+-(Y-\lambda g)d,\qquad l_U^\infty(d)=\frac{\lambda}{\tau}(d_+-gd).$$

In the human-favourable direction d=2Y−1, the labelled rate is −λ|Y−g|<0. The auxiliary rates are nonnegative. Finite attainment depends on their joint compensation, including magnitudes. An individual auxiliary term can grow along an escaping coefficient sequence while negative labelled terms grow in the opposite direction. Bounding all remaining terms by finite individual infima cannot control that sum.

Koriyama and Bellec's formal ICML 2025 article, Assumption 2.5(2) and Lemma A.2, explicitly requires finite individual infima. Its proof on internal page 12 bounds the other rows by those infima. Our losses invalidate that step. This pinpoints the failed hypothesis and proof step rather than treating a different notation or an added sample block as evidence of novelty. [Official publication record](https://proceedings.mlr.press/v267/koriyama25a.html).

## Exact finite-design check at two strictly convex powers

Appendix B5 already supplies an intercept-and-slope design. It has labelled x=(−1,1), choices (0,1), base predictions (1/10,9/10), auxiliary x=(−2,0,0,0,0,0,0,2) and predictions (1/4,1/2,1/2,1/2,1/2,1/2,1/2,3/4). Its observed-reference path is gλ=1/2+(gbase−1/2)/(2λ).

Compare λ=1/2 and λ=3/5. Exact original-row calculations give strictly interior predictions, identical row categories, positive curvature weights, a weighted Gram matrix I₂, corrected moment (1/2,29/80) and reference Hessian I₂/4. Reflection symmetry minimizes the intercept at zero for each slope. On that line the derivative is

$$F_\lambda(\beta)=\frac{1-\lambda}{2}\tanh(\beta/2)+\frac{\lambda}{4}\tanh\beta-\frac{29}{80}.$$

It is strictly increasing. At half power, F(0)<0 and F(log64)=4045/852176>0 give a unique finite root. At power 3/5, the positive-direction recession is −1/80, making the objective unbounded below. Both use strictly convex rows and full auxiliary rank. This exact check uses an existing designed example, not a new Gaussian experiment or empirical dataset. The path changes predictions; the population pair above instead holds predictions fixed.

## What remains established general theory

Condition on the low-dimensional data and define the manuscript's cone C=cone(product of corrected intervals)∩ker(Vᵀ). The remaining Gaussian matrix is independent. For n=m+N and q independent constraints, its kernel has the law of a Haar subspace of dimension n−q. Substituting that subspace into Amelunxen et al.'s Theorem I places the transition at q≈δ(C). This is a direct application of established kinematics. It supplies the transition once δ(C) is known; it does not itself evaluate the model-specific limit δ(C)/m. [Examined theorem](https://arxiv.org/pdf/1303.6672).

The manuscript evaluates that limit by retaining the corrected intervals, the fitted intercept and signal constraints, the ordered infimum and supremum, and uniform multiplier and scale controls. Those calculations and their controlled consequences are the candidate contribution. The underlying Gaussian transition is not new. This comparison rules out one direct-substitution objection and explicitly accepts the broader geometric dependence. It does not certify sufficient JCM value or global result priority.
