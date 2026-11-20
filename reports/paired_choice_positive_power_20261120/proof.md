# The repeated-task boundary at a fixed interior correction weight

Use the Gaussian two-sample design of the [unit-weight result](../paired_choice_boundary_20261120/proof.md). Each respondent has a fixed number $K$ of binary tasks. The labelled and auxiliary samples contain $m$ and $N$ respondents, with $N/m\to\tau>0$. The $K$ attributes for each slope and respondent have covariance $R$, a positive definite matrix with unit diagonal. Slopes and respondents are independent. Labels are independent fair Bernoulli variables, independent of all attributes. The forecast is $1/2$, the fitted model includes an intercept, and the correction weight $\lambda$ is fixed in $(0,1)$.

Let $S$ be a vector of $K$ independent fair signs, $G\sim N(0,I_K)$ independent of $S$, $T=R^{1/2}$, and $B=[-1,1]^K$. The critical number of slopes per labelled respondent is
$$
\kappa_R(\lambda)=
\max_{u\geq0}\left[
 E\{\|G\|^2-\operatorname{dist}^2(G,uT((1-\lambda)B-S))\}
 +\tau E\{\|G\|^2-\operatorname{dist}^2(G,u(\lambda/\tau)TB)\}
\right].
$$
When $q/m\to\kappa<\kappa_R(\lambda)$, the corrected loss has a unique finite minimizer with probability tending to one. When $q/m\to\kappa>\kappa_R(\lambda)$, it is unbounded below with probability tending to one.

## Finite sample reduction

Write $s_{ai}=2Y_{ai}-1$ for the labelled signs and $t=2p-1$ for trial choice probabilities. Let $W$ be the combined untransformed Gaussian slope matrix and $\mathcal T$ the block diagonal matrix with $T$ repeated for every respondent. After multiplying the score equations by $Km$, they become
$$
 W^T\mathcal T v=0,\qquad 1^Tv=0,\qquad
 v_L=(1-\lambda)t_L-s,\qquad v_U=(\lambda/\tau_m)t_U,
$$
where $\tau_m=N/m$. For a fixed $0<b\leq1$, constrain $t$ to the box $[-b,b]^{K(m+N)}$. The sliced set of candidate $v$ is compact and convex. It is nonempty with probability tending to one by the law of large numbers for the labelled signs and the strictly positive aggregate width of the two boxes. Each labelled coordinate of $v$ has magnitude at least $\lambda$. Thus the sliced set is bounded away from zero and lies strictly on one side of a hyperplane. Its conic hull after transformation by $\mathcal T$ is a closed pointed convex polyhedral cone $C_b$. Existence of a feasible $t$ is equivalent to a nonzero intersection of $C_b$ with the nullspace of $W^T$. Conditional on the signs, this nullspace is uniformly oriented and independent of the cone.

## Asymptotic statistical dimension

Express squared Gaussian projection onto $C_b$ as the maximum over $u\geq0$ and the sliced blocks of $\sum_a [2uG_a^TTv_a-u^2\|Tv_a\|^2]$. With the sum constraint removed, the two role-specific terms are independent block averages. In their limit, the same expression as in $\kappa_R(\lambda)$ applies with $B$ replaced by $bB$ in both terms.

Uniform convergence on bounded scales follows from a uniform Lipschitz envelope proportional to $1+\|G_a\|$. All maximizing scales remain bounded in probability. Indeed, $2uG_a^TTv_a-u^2\|Tv_a\|^2\leq 2\|G_a\|^2-u^2\|Tv_a\|^2/2$ and the minimum norm over a labelled block is at least $\lambda\sqrt{K\lambda_{\min}(R)}$. Since the objective at $u=0$ is zero, this supplies a common bound on the relevant scales. The population maximum is positive because the one-sided derivative at zero is the expected support of the two positive-width centered boxes, with the mean shift contributing zero.

For the constrained lower bound, fix a positive deterministic population maximizing scale. The unconstrained block maximizers are unique. The simultaneous sign change $(G_a,s_a)\mapsto(-G_a,-s_a)$ negates the maximizer on each labelled block. The same symmetry holds on auxiliary blocks. Consequently the mean of the maximizing coordinates converges to zero. Endpoints formed by taking all $t$ coordinates equal to $-b$ or $b$ give limiting total coordinate sums per $Km$ of $-b$ and $b$, respectively. Mix the maximizers toward the endpoint on the other side of zero by the common fraction needed to impose $1^Tv=0$. The fraction converges to zero and the normalized objective change tends to zero by the same integrable envelopes.

The normalized projection values are dominated by the normalized squared Gaussian norms, a uniformly integrable family. Taking conditional expectations given the signs shows that $\delta(C_b)/m$ converges in probability to the stated maximum with $bB$. This argument can set the projection value to zero on the vanishing event that the slice is empty.

For the upper side, use $b=1$. The strict gap between $q$ and $\delta(C_1)$ dominates the $O(\sqrt{K(m+N)})$ transition width in the approximate conic kinematic formula. The target lies outside the moment support with probability tending to one. If $q\geq K(m+N)$, the Gaussian nullspace is already trivial almost surely. For the lower side, choose $b<1$ close enough to one by continuity of the population objective on the compact scale range. A nonzero cone intersection produces probabilities strictly inside $[0,1]$. The positive-weight design spans the coefficient space almost surely when $q+1\leq K(m+N)$, which holds on this strict lower side. These witnesses and the fixed-menu support result give the claimed existence statements. At $\lambda=1$, the labelled term becomes $-u^2K$ in the population formula. Rescaling $u$ by $\tau$ recovers the unit-weight calculation.

## Explicit separation at correction weight 0.90

Set $K=2$ and $\tau=1$. For $R=I$, the labelled squared distance in either coordinate has expectation $D(-u\lambda)+D(u(2-\lambda))$, where $D(z)=E(G_1-z)_+^2$. The auxiliary distance is $2D(u\lambda)$. Evaluating the resulting objective at $u=0.48$ gives a lower bound above $0.750$.

For within-person correlation $\rho=0.99$, use eigen-coordinates. Set $A=\sqrt{2(1+\rho)}$ and $B_0=\sqrt{2(1-\rho)}$. The labelled parallelogram is contained in a rectangle of half-widths $u(1-\lambda)(A,B_0)$ and center on either axis. The two center types, up to independent signs, are $u(A,0)$ and $u(0,B_0)$, each with probability $1/2$. The centered auxiliary rectangle has half-widths $u\lambda(A,B_0)$. Expected distances to the rectangles bound the population objective above. In each labelled type, a distance term $D(-u\lambda A)$ or $D(-u\lambda B_0)$ has second derivative at least $\lambda^2 A^2$ or $\lambda^2 B_0^2$, respectively. The rectangle upper objective is thus strongly concave with curvature at least $2\lambda^2$. Evaluating it and its first derivative at $u=0.405$ gives the upper bound $0.634481$, below $0.635$. The nominal design ratio $0.663$ is strictly between the bounds. The rectangle and scalar-normal calculations establish the strict inequality independently of the polygon integration.

For all $R$, the same-task joint law is unchanged. At the zero coefficient, the prediction score vanishes row by row. The population Hessian is $I/4$, and the covariance of the averaged human score per respondent is $I/(4K)$ by independent fair labels. The ordinary Logit exact existence count also agrees across the two positive definite $R$ matrices by conditional general position and Cover's count. In this strict comparison, both labelled and auxiliary menus have positive weights in the corrected log partition.

The result uses a fixed weight and a balanced, calibrated null design. The argument supplies the interior-weight analogue of the unit-weight comparison. The cluster-robust score covariance and population tuning problem in [Broska and Howes](https://arxiv.org/abs/2610.09601v1) are distinct quantities. The Gaussian orientation step uses the [approximate conic kinematic formula](https://doi.org/10.1093/imaiai/iau005).
