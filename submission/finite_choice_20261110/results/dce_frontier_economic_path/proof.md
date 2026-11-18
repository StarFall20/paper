# Approaching a positive correction-power endpoint

Consider a fixed collection of finite menus with alternative attributes (x_{ij}\in\mathbb R^p). Write the existing compensated loss as
$$
 Q_\lambda(\beta)=
 \sum_i w_i(\lambda)\log\sum_j\exp(x_{ij}^\top\beta)-b(\lambda)^\top\beta .
$$
The curvature weights $w_i$ and moment $b$ are affine in $\lambda$. Drop menus whose weights are identically zero. Assume the remaining weights are positive near an upper feasible endpoint $c\in(0,1)$. Their within-menu contrasts span $\mathbb R^p$. Suppose $b(\lambda)$ is interior to its weighted menu support just below $c$, and $b(c)$ is in the relative interior of a facet of the support at $c$ with outward unit normal $d$. An alternative is on the menu face when it attains $M_i=\max_j x_{ij}^\top d$. Denote its index set by $T_i$ and write $\Delta_{ij}=M_i-x_{ij}^\top d$. Let $\Delta_*>0$ be the smallest positive gap among the active menus. Assume the normal slack is transverse, $s(\lambda)=\sum_i w_i(\lambda)M_i-b(\lambda)^\top d=a(c-\lambda)$ with $a>0$.

Let $u_*\in d^\perp$ minimize the conditional face loss at $c$. That loss retains only $T_i$ in each log sum and restricts the coefficient to $d^\perp$. The assumptions make this minimum unique and finite, because the tied contrasts span $d^\perp$ and the projected moment is in their weighted support's relative interior. Define
$$
 A=\Delta_*\sum_i w_i(c)\,
   \frac{\sum_{j:\Delta_{ij}=\Delta_*}\exp(x_{ij}^\top u_*)}
        {\sum_{k\in T_i}\exp(x_{ik}^\top u_*)}.
$$
Then the minimizing coefficient for power $\lambda$ has the expansion
$$
 \widehat\beta(\lambda)=t(\lambda)d+u_*+O(c-\lambda),\qquad
 t(\lambda)=\frac{\log(1/(c-\lambda))+\log(A/a)}{\Delta_*}+o(1).
$$
The $O(c-\lambda)$ vector lies in $d^\perp$. The separate $o(1)$ term describes the normal coordinate.

## Derivation

At each interior root, the normal score is exactly
$$
 a(c-\lambda) = \sum_i w_i(\lambda)\sum_j p_{ij}(\widehat\beta)\Delta_{ij}.
$$
As $\lambda\uparrow c$, the total probability outside the face, for each active menu is therefore $O(c-\lambda)$, using the positive lower bounds on weights and gaps. For any decomposition $\beta=td+u$, $u\in d^\perp$, the conditional probabilities on $T_i$ equal the face softmax at $u$, because the terms $t x_{ij}^\top d=tM_i$ agree there. The difference between the projected full moment and the projected conditional-softmax moment is $O(c-\lambda)$, uniformly over $u,t$ at these roots. Changing weights and target to their values at $c$ also changes the projected score by $O(c-\lambda)$. The smooth face moment map is one-to-one, onto the relative interior, with an invertible Hessian on $d^\perp$. Its local inverse at $b(c)$ implies $u=u_*+O(c-\lambda)$. Global strict convexity ensures that the local inverse is the one at the actual face score.

For a menu with a positive gap, the off-face to face probability ratio now has bounded stable within-face utilities. Its limit zero forces $t\to+\infty$. Substitution in the normal score and factoring out $\exp(-t\Delta_*)$ gives $a(c-\lambda)=A\exp(-t\Delta_*)[1+o(1)]$. This proves the expansion without interchanging a probability limit and an optimization problem.

## Price-equivalent and forecast consequences

For coefficient positions $k$ and $r$ for Bere and price, the derived price equivalent is $-10\widehat\beta_k/\widehat\beta_r$ under the source scaling. If $d_r\neq0$, its algebraic limit is $-10d_k/d_r$ and the price sign eventually equals the sign of $d_r$. If $d_r=0$ and $u_{*,r}\neq0$, the price coefficient remains finite. A nonzero $d_k$ then gives logarithmic divergence of the ratio, with coefficient $-10 d_k/(u_{*,r}\Delta_*)$ per unit of $\log(1/(c-\lambda))$. If $d_k=0$ as well, the ratio tends to $-10u_{*,k}/u_{*,r}$. The zero limiting denominator requires separate analysis. A positive price coefficient gives an algebraic ratio only. For a negative price coefficient, the deterministic utility comparison for a premium $P$ is $\widehat\beta_k+(P/10)\widehat\beta_r>0$. Its eventual sign is set by $d_k+(P/10)d_r$ when this is nonzero, and by $u_{*,k}+(P/10)u_{*,r}$ when the normal contrast is zero and the latter quantity is nonzero.

For any fixed evaluation menu and chosen alternative $y$, its log loss obeys
$$
 \log\sum_j e^{x_j^\top\widehat\beta} - x_y^\top\widehat\beta=
 t(\lambda)\Delta_y^{\mathrm{eval}}+
 \log\sum_{j:\Delta_j^{\mathrm{eval}}=0}e^{x_j^\top u_*}-x_y^\top u_*+o(1).
$$
The sample average has coefficient $\overline{\Delta_y^{\mathrm{eval}}}/\Delta_*$ on $\log(1/(c-\lambda))$. A fixed evaluation set has a strictly positive log-loss coefficient when at least one observed choice is outside its menu's maximizing set. If the evaluation set lies entirely on those sets, the leading coefficient vanishes and its loss has a finite limit.

These fixed-design conclusions apply only when the endpoint checks above pass. The proof does not cover a lower-dimensional face's potentially multiple diverging directions, weight one with vanishing menu weights, an unknown price-sign restriction or limits that take a sample size to infinity at the same time. The empirical quantities are deterministic fitted coefficients and allocation summaries, with no population welfare or sampling-uncertainty interpretation.

## Attribution

[Geyer (2009)](https://www.stat.umn.edu/geyer/gdor/) and [Rinaldo, Fienberg and Zhou (2009)](https://www.stat.cmu.edu/~arinaldo/papers/Rinaldo_Fienberg_Zhou_2009.pdf) supply the directions of recession, conditional face model and exponential-family normal-fan geometry. The positive endpoint derivation uses those ideas and the elementary exact score identity along the affine compensated-power path. Candidate value lies in quantifying the coefficient path from the original menu gaps and verifying its preference and forecast consequences on the released design. Convex support, facet limits and the monetary coefficient ratio are established prior.

## Scalar check using all the gaps

The leading-gap limit can converge slowly when the location of the first useful gap depends on a small probability. A scalar calculation keeps every gap. Hold the tangent coordinate at the face estimate $u_*$ and solve the full normal score at the chosen power for $t_*(\lambda)$. The left side, as a function of $t$, is continuous and strictly decreasing. Its endpoint values are zero and the weighted sum of the menus' largest gaps. The interior score slack lies strictly between those values, so the scalar root is unique.

A computable sensitivity bound connects this root to the full fit. For a fixed power and slack, denote the scalar root at an arbitrary tangent by $t(u)$. The normal score derivative in $t$ is minus the sum of weighted within-menu gap variances. Its derivative in $u$ is the sum of weighted within-menu covariances between gaps and tangent features. Define $K$ as the maximum over within-menu pairs with different gaps of the norm of their tangent-feature difference divided by the absolute difference of their gaps. The pairwise covariance identity bounds the covariance norm by $K$ times the corresponding gap variance. The implicit function theorem and integration along the segment from $u_*$ to the fitted tangent $u$ give
$$
 |t(u)-t_*(\lambda)|\le K\lVert u-u_*\rVert .
$$
This comparison uses the same fixed power and slack. It requires positive curvature in the normal score and uses no observed evaluation choices.

To bound the tangent error from the design alone, let $D$ be the largest within-menu projected feature difference, $\mu$ be the smallest eigenvalue of the endpoint face Hessian on $d^\perp$, and put $r=1/(2D)$. Within this radius, the ratio of each conditional face probability to its value at $u_*$ is at least $e^{-Dr}$. The pairwise variance form of the face Hessian gives curvature at least $\mu e^{-2Dr}$ in the ball. At the full root the endpoint face score has norm at most $B(c-\lambda)$, where
$$
 B=D a/\Delta_*+\sum_i |w'_i|\max_{j\in T_i}\lVert P_d x_{ij}\rVert+\lVert P_d b'\rVert .
$$
Here $P_d$ projects onto $d^\perp$. The first term bounds the effect of off-face mass using the exact slack. The remaining terms bound the power-induced changes in the face weights and target. If $B(c-\lambda)< r\mu e^{-2Dr}$, the fitted tangent lies inside the ball. Indeed, along its radial direction, the gradient component at radius $r$ is at least $r\mu e^{-2Dr}$ and cannot decrease at larger radii by convexity. The same radial integration inside the ball yields $\lVert u-u_*\rVert\le B(c-\lambda)e^{2Dr}/\mu$. The numeric check reports when this sufficient condition is too weak. The resulting bounds apply to the scalar approximation, with no guarantee of forecast quality for the underlying estimator.
