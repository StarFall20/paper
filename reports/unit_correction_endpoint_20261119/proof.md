# Choice estimates as the correction weight approaches one

Let $H=Q_0$ denote ordinary labelled choice log loss and let $U=Q_1$ denote unit correction on the same finite design. For $s=1-\lambda$ the exact path is

\[
Q_{1-s}=(1-s)U+sH.
\]

In $U$, only the auxiliary menus supply a log partition. The labelled prediction error contributes a linear term. Assume the combined within-menu contrasts span the fitted coefficient space. Write $h=H^\infty$ for ordinary labelled recession. It is a nonnegative, continuous polyhedral function.

## Existence up to unit correction

If the unit target lies outside its auxiliary moment polytope, a direction of negative unit recession also has negative recession for sufficiently small $s>0$. The rest of the result concerns a feasible unit target.

Let $T_i$ be the sets of tied auxiliary alternatives in the minimal face containing the unit target. Pick one $x_i^*$ in each $T_i$. Their within-menu differences span a space $E$. Write $N=E^\perp$ and $a_{ij}=\Pi_N(x_i^*-x_{ij})$ for the auxiliary alternatives outside $T_i$. Define $K$ as the directions $d$ in $N$ satisfying $a_{ij}^Td\geq0$ for all outside alternatives. When no outside alternatives occur, set $K=N$.

The unit recession is zero exactly on $K$. The unit target can be represented with positive mass on every member of every $T_i$. The difference between the maximum directional utility and the mean under that representation is nonnegative in every menu. Their weighted sum is zero exactly when all members of each $T_i$ maximize directional utility. This requires orthogonality to their within-menu contrasts and the displayed inequalities.

A unique finite optimizer exists for every $0<s<1$ precisely when $h(d)>0$ for every nonzero $d\in K$. A feasible unit target guarantees nonnegative recession of $U$. Hence the recession of $Q_{1-s}$ is strictly positive on all nonzero directions exactly under the stated condition. A zero direction precludes a finite optimizer by the fixed-design support criterion. An ordinary labelled fit with full rank and an interior target supplies this condition directly.

## A finite unit objective with several minimizers

If every auxiliary alternative belongs to $T_i$, the unit minimum is attained. Its minimizer set is a single affine space $u_*+N$, where $u_*$ is the unique minimizing projection in $E$. Under the existence condition above, $H$ has a unique minimizer $\beta_*$ on this affine space. The path converges to $\beta_*$.

To see this, optimality against $\beta_*$ gives $U(\widehat\beta_s)\to\min U$ and $H(\widehat\beta_s)\leq H(\beta_*)$. Strict moment interiority in $E$ makes the $E$ component bounded and convergent to $u_*$. On $N$, $h$ is uniformly positive on the unit sphere whenever $N$ is nonzero. Ordinary loss is bounded below by $h$ and $h$ is Lipschitz. The inequality for $H$ therefore bounds the remaining components. Every limit point minimizes $H$ on the affine space, which establishes convergence. In particular, losing curvature at unit correction can select a finite limit even when the coefficient is unidentified at the endpoint itself.

## A feasible unit target on a proper auxiliary face

Suppose at least one auxiliary alternative is outside $T_i$. Assume the existence condition holds. Set $L=\log(1/s)$. The following programs select the leading coefficient:

1. Minimize $h(v)$ over $v\in N$ subject to $a_{ij}^Tv\geq1$ for each outside auxiliary alternative. Use one linear epigraph variable per labelled menu to write this as a linear program. Let $V$ be the set of optimal $v$.
2. In each labelled menu choose an alternative $r_i$ maximizing utility at a relative interior point of $V$. For every labelled alternative define the gap $b_{ij}^Tv=(x_{ir_i}-x_{ij})^Tv$ on $N$. These gaps are nonnegative throughout $V$.
3. Form the affine families $a_{ij}^Tv-1$ for outside auxiliary alternatives and $b_{ij}^Tv$ for labelled alternatives. On the current set, keep the gaps that vary. Maximize their minimum and restrict to its maximizing set. Repeat until all gaps are constant.

The final set is a singleton $v_*$. The path satisfies $\widehat\beta_s=L v_*+O(1)$, and its $E$ component converges to the conditional face coefficient. For an evaluation menu with chosen alternative $y$, the coefficient of $\log(1/s)$ in its negative log probability is $\max_j (x_j-x_y)^Tv_*$. A zero coefficient gives bounded loss.

### First scale

Write $\widehat\beta_s=u_s+n_s$ in $E\oplus N$. For every fixed trial coefficient, optimality and $H\geq0$ bound the limiting value of $U(\widehat\beta_s)$ above by the trial unit value. Consequently $U(\widehat\beta_s)\to\inf U$. Decompose $U$ into the conditional tied-face log loss in $u_s$ plus nonnegative outside log-sum terms. The face target is in its relative interior and its contrasts span $E$, so the conditional face objective is coercive and strictly convex there. Thus $u_s\to u_*$.

Project the score onto $N$. The auxiliary part is minus the weighted sum of outside probabilities times their gaps. Its magnitude in any fixed direction is $O(s)$ because the labelled score is bounded. An exposing direction is positive on all outside gaps. It follows that each outside fitted probability is $O(s)$. Ratios to a tied alternative give $a_{ij}^T n_s\geq L-O(1)$.

Compare $n_s$ with $Lv$ for a solution of the first program, holding $u_s$ fixed. The auxiliary outside log sums at the trial point are $O(s)$ and they are nonnegative at the optimizer. For all $n$, $H(u_s+n)=h(n)+O(1)$ uniformly because $u_s$ is bounded and a finite log sum differs from its largest input by a bounded amount. Thus $h(n_s)\leq L\min_{v\in V}h(v)+O(1)$. The first program is feasible by scaling an exposing direction, and its minimizing set is compact because $h$ is uniformly positive on the unit part of $K$. The first program's optimal set is a polyhedron. Hoffman's linear error bound applied to its dilate by $L$ gives $n_s=Lz_s+O(1)$ with $z_s\in V$.

### Selection within the first optimal set

The function $h$ is a positive sum of convex menu-wise maximum functions plus an affine term. Constancy on $V$ implies equality in each menu's convexity inequality on $V$, so each maximum is affine there. An alternative maximizing at a relative interior point therefore maximizes everywhere on $V$. This proves the anchor claim in the second step.

At any recursion set $W$ enclosed in $V$, write $n_s=Lz_s+e_s$ with $z_s\in W$ and bounded $e_s$. For a comparison $z^+\in W$, hold $e_s$ and $u_s$ fixed and use $Lz^++e_s$. After dividing the objective by $s$, the sum of the human affine terms cancels because it is $Lh(z)$ plus fixed constants. In each human or auxiliary log sum, factor out the contribution of the gaps constant on $W$. These factors are bounded above and below by positive constants, using the anchor term, bounded offsets and nonnegative gaps. The remaining human log increments are comparable to sums of $\exp(-L b_{ij}^Tv)$ uniformly on $W$. The remaining auxiliary log increments are multiplied by $(1-s)/s$, so they are comparable to sums of $\exp[-L(a_{ij}^Tv-1)]$. Multiplicative constants are independent of $s$.

If the largest achievable minimum of the varying gaps is $\alpha$, compare with a maximizing $z^+$. Optimality implies that the smallest varying gap at $z_s$ is at least $\alpha-O(1/L)$. Apply Hoffman's bound again to obtain an $O(1)$ distance to the dilated new set. Each step lowers its dimension because finitely many nonconstant affine gap level sets cannot cover its relative interior. When all gaps are constant, any difference of points is orthogonal to the auxiliary tied contrasts, the outside auxiliary gaps and the labelled within-menu contrasts. The combined rank assumption gives a singleton. This proves the coefficient expansion and the forecast rate follows by taking the maximum utility in the evaluating menu.

## Relation to prior results

The positive-weight upper-endpoint proof in the main study has a different final gap family because both labelled and auxiliary partitions keep positive weight at its limit. At unit correction, a labelled gap of size $a$ and an auxiliary gap of size $a+1$ contribute on the same scale after dividing by $s$. This is the concrete reason to include both families. The error bound and successive max-min construction use established polyhedral and exponential-penalty arguments. [Hoffman](https://nvlpubs.nist.gov/nistpubs/jres/049/4/V49.N04.A05.pdf) · [Cominetti and San Martín](https://www.ceremade.dauphine.fr/~carlier/Cominetti-SanMartin.pdf) · [Weighted supervised logistic existence](https://doi.org/10.1080/03610926.2016.1260742)
