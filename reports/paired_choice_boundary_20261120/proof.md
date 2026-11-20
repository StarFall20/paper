# Existence with correlated tasks within respondents

Consider independent labelled and auxiliary samples with $m$ and $N$ respondents, with $N/m\to\tau\in(0,\infty)$. Each respondent has $K$ binary tasks. Conditional on the predictors, the human choices are independent fair Bernoulli variables. The prediction is $1/2$ on every task, and the correction weight is one. The fitted contrast contains an intercept and $q$ slopes. For each slope and respondent, the vector of $K$ attributes is $N_K(0,R)$. Slopes, respondents and sample roles are independent. The fixed matrix $R$ is positive definite and has unit diagonal.

Write $T=R^{1/2}$, $B=[-1,1]^K$ and $G\sim N_K(0,I_K)$. Define

$$
\kappa_R=\max_{u\ge0}\left[
\tau E\{\|G\|^2-\operatorname{dist}^2(G,uTB)\}
-\tau^2 K u^2
\right].
$$

**Proposition.** If $q/m\to\kappa<\kappa_R$, a unique finite minimizer exists with probability tending to one. If $q/m\to\kappa>\kappa_R$, the corrected objective is unbounded below with probability tending to one. The boundary uses slopes per labelled respondent. Slopes per labelled task have critical ratio $\kappa_R/K$.

## Finite sample reduction

Let $s_{ai}=2Y_{ai}-1$ on the labelled sample. Use $W_L,W_U$ for the independent standard Gaussian slope matrices before within-respondent transformation, and $T_L,T_U$ for their block transformations with repeated blocks $T$. Let $\tau_m=N/m$. At unit correction, subtract the auxiliary probability $1/2$ and write $t=2p_U-1$ for a candidate auxiliary probability vector. The intercept and slope score equations are

$$
\mathbf 1^T t=\tau_m\mathbf 1^T s=:a,
\qquad
W_U^T T_U t=\tau_m W_L^T T_Ls,
\qquad t\in[-1,1]^{NK}.
$$

Conditional on $s$, the right side of the slope equation equals $vg$ in distribution, where $v=\tau_m\|T_Ls\|>0$ and $g$ is standard Gaussian in $\mathbb R^q$, independent of $W_U$. A feasible moment exists exactly when the nullspace of $[W_U^T,-g]$ intersects the following cone at a nonzero point:

$$
C_b=\{(uT_U t,uv): u\ge0,\ t\in[-b,b]^{NK},\ \mathbf 1^Tt=a\},\qquad b=1 .
$$

For (b\in$0,1]$, the sliced box is nonempty with probability tending to one because $a/(NK)\to0$. Conditional on a nonempty box, $C_b$ is a closed pointed convex polyhedral cone. Its last coordinate fixes the positive scale at any nonzero point. The Gaussian nullspace is independent of the cone conditional on the labels.

## Statistical dimension

For independent $H_1,\ldots,H_N\sim N_K(0,I_K)$ and $Z\sim N(0,1)$, the squared projection on $C_b$ is given by maximizing

$$
\sum_{j=1}^N\{2u H_j^TTt_j-u^2\|Tt_j\|^2\}
+2Zuv-u^2v^2
$$

over $u\ge0$ and the constrained box above. Dropping the sum constraint gives the sum of $N$ separate block projection expressions. The label signs are independent within a block, so $v^2/m\to\tau^2\operatorname{tr}(R)=\tau^2K$. The linear term divided by $m$ vanishes uniformly on compact sets. The normalized objective without the sum constraint converges uniformly on compact scale sets to

$$
\Psi_b(u)=\tau E\{\|G\|^2-\operatorname{dist}^2(G,ubTB)\}-\tau^2Ku^2.
$$

For the uniform law, write each block objective as the maximum over the fixed box $[-b,b]^K$. On a compact scale set, differences are bounded by a constant times $(1+\|H_j\|)|u-u'|$. The bound $\sum_j\|H_j\|^2/m+2Z^2/m-u^2v^2/(2m)$ keeps all optimal scales in a bounded region with probability tending to one. The probability bound uses the fact that the value at zero scale is zero.

The intercept sum does not change the limit. At a positive deterministic maximizer $u_b$ of $\Psi_b$, the block maximizers are unique because $T$ is invertible. They are odd functions of $H_j$, so the mean ordinary coordinate over all the maximizers converges to zero. The desired mean $a/(NK)$ also converges to zero. Move all the block maximizers toward either the all-$b$ or all-$(-b)$ vector by the same mixing fraction, choosing the endpoint on the side of the desired mean. The fraction tends to zero and enforces the sum exactly. The loss of objective divided by $m$ tends to zero by the integrable linear envelope in the block normals. A zero maximizer would already be feasible and attain zero. In fact the optimal scale is positive for $b>0$, since the population objective has a positive right derivative at zero.

The projection value divided by $m$ is bounded by $(\sum_j\|H_j\|^2+Z^2)/m$ and hence is uniformly integrable. Conditional expectation given $s$ now yields $\delta(C_b)/m\to\max_u\Psi_b(u)$ in probability. One may define the projection value as zero on the vanishing event of an empty sliced box.

## Strict sides and an interior witness

If $q\geq NK+1$, the Gaussian matrix has full column rank almost surely, so its nullspace is trivial. For a nontrivial nullspace, the approximate conic kinematic formula applies because, conditional on the labels, its orientation is uniform and the ambient dimension is $NK+1=O(m)$. The fixed positive gap in ratios dominates the $O(\sqrt{NK})$ transition width. Above $\kappa_R$, use $b=1$. The nullspace misses the cone aside from zero, which puts the target outside the auxiliary moment support. Its separating direction has negative recession cost.

Below $\kappa_R$, take $b<1$ sufficiently close to one. The maximum of $\Psi_b$ tends to that of $\Psi_1$ as $b\uparrow1$, by continuity on the same compact scale set. The nullspace meets $C_b$ at a nonzero point with probability tending to one. The recovered probabilities are strictly between zero and one. Also $\kappa_R\leq\tau K$, so the strict lower side has $q+1\leq NK$ eventually. Since $T_U$ is invertible, the augmented auxiliary design with its intercept has full rank almost surely. The strictly interior probability representation gives the desired unique finite minimizer.

## Two tasks with identical marginal choice information

For two tasks, set $R_\rho=\left(\begin{smallmatrix}1&\rho\\\rho&1\end{smallmatrix}\right)$. Let $F(a)=1-E(|Z|-a)_+^2$ for a scalar standard Gaussian $Z$. Independent tasks give the value $\max_u[-2\tau^2u^2+2\tau F(u)]$. As $\rho\uparrow1$, the transformed box converges to a segment of half-length $2u$, giving the limit $\max_u[-2\tau^2u^2+\tau F(2u)]$. Convergence follows from Hausdorff convergence on the common compact scale range. $F$ is strictly concave, $F(0)=0$, and the limiting maximizer is positive, so the second value is strictly smaller. This argument uses positive definite $R_\rho$ throughout the finite-estimation theorem.

The chosen finite comparison can also be bounded without the two-dimensional integral. In eigen-coordinates the transformed box lies in the rectangle of half-lengths $2u$ and $c u$, where $c=\sqrt{2(1-\rho)}$. Thus

$$
\kappa_{R_\rho}\leq\max_{u\ge0}\{-2\tau^2u^2+\tau F(2u)+\tau F'(0)c u\}.
$$

For $\tau=1,\rho=0.99$, the right side is less than $0.551$. Evaluate the strongly concave objective at $u=1/3$ and add the square of its derivative divided by eight. Evaluating the independent-task objective at $u=0.4$ gives a lower bound above $0.670$. These bounds put the test ratio $q/m=0.585$ strictly between the two boundaries.

Every individual task has the same Gaussian slope distribution across $R$, and its observed choice and frozen prediction have the same joint law. At the true coefficient zero the prediction correction to the score is zero on every task. Let $x_i$ include the intercept. The population Hessian is $E(x_i x_i^T)/4=I/4$. The covariance of the average human score in a respondent's $K$ tasks is $I/(4K)$, because labels are independent fair signs and their conditional score means are zero. For fixed parameter dimension, the prospective per-respondent sandwich covariance is $4I/K$ in every design.

Ordinary logit also has the same exact probability of existence across positive definite $R$, for $q+1\leq Km$. The labelled augmented rows have general position almost surely. A weak separator with some nonzero margins can be perturbed to a strict separator by making the at most $q$ tied margins positive. The tied rows are independent, and a sufficiently small perturbation keeps strict margins positive. There is no nonzero direction tying every row under full rank. Cover's counting theorem then gives separation probability $2^{-(Km-1)}\sum_{j=0}^q {Km-1\choose j}$, conditional on any design in general position. Both designs therefore approach ordinary finite existence at $q/m=0.585$ with $K=2$.

The comparison isolates within-respondent attribute correlation in a correctly calibrated null model. It supplies a counterexample to identifying the corrected threshold from task marginals, the usual local sandwich information, or the ordinary existence probability. The theory concerns a fixed positive definite Gaussian correlation across equally sized panels and uses unit correction. The released choice designs in the paper do not supply estimates of this theoretical correlation parameter.

- The Gaussian orientation result used above is the approximate conic kinematic formula of Amelunxen, Lotz, McCoy and Tropp. [Published theorem](https://doi.org/10.1093/imaiai/iau005)
- The ordinary-logit count is Cover's function-counting theorem. [Original article](https://isl.stanford.edu/~cover/papers/paper2.pdf)
- Cluster-robust PPI supplies separate fixed-dimensional inference and tuning results. [Broska and Howes](https://arxiv.org/abs/2610.09601)
- Correlation in ordinary pattern classification has an established geometry. [López, Schröder and Opper](https://arxiv.org/abs/cond-mat/9505093)
