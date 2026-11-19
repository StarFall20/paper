# Choice logit at a positive weight limit on several faces

Let (c\in(0,1)) be a positive upper limit and write \(\delta=c-\lambda\). Use the normalized compensated objective

\[
Q_\delta(\beta)=\sum_i w_i(\delta)\log\sum_{j\in A_i}\exp(x_{ij}^{\mathsf T}\beta)-b(\delta)^{\mathsf T}\beta.
\]

The available alternatives \(A_i\) and the utility attributes are fixed. Assume menu weights and the corrected target are affine in \(\delta\), weights are positive at zero, and menu contrasts span the parameter space. Assume \(b(0)\) is on the boundary of the attainable moment polytope and \(b(\delta)\) is interior for all sufficiently small positive \(\delta\). These conditions allow the minimal face at the endpoint to have any positive codimension.

## The limiting direction and a choice prediction

Find the minimal face containing \(b(0)\). Let \(T_i\) be the alternatives maximizing an exposing direction in menu \(i\). Pick \(x_i^*\) from \(T_i\). Let \(E\) be spanned by all within-menu contrasts among alternatives in \(T_i\), and let \(N=E^\perp\). Define outside gaps \(a_{ij}=\Pi_N(x_i^*-x_{ij})\) for \(j\notin T_i\) and the inward moment change

\[
\Pi_N\{\textstyle\sum_i w_i(\delta)x_i^*-b(\delta)\}=\delta s.
\]

First minimize \(s^{\mathsf T}v\) over \(v\in N\) subject to \(a_{ij}^{\mathsf T}v\geq1\) for every outside alternative. The optimal set is compact. If it contains multiple points, disregard gaps constant on the set. Maximize the smallest remaining gap over the set and retain the maximizing set. Repeat with the gaps still varying on that set. The procedure returns a unique \(v_*\).

**Proposition.** The optimizer satisfies \(\widehat\beta_\delta=v_*\log(1/\delta)+O(1)\). Its component in \(E\) converges to the finite conditional logit coefficient using \(T_i\). For an arbitrary fixed evaluation menu and observed choice \(y\),

\[
-\log P_{\widehat\beta_\delta}(y)=
\left[\max_{j\in A}(x_j-x_y)^{\mathsf T}v_*\right]\log(1/\delta)+O(1).
\]

A positive bracket gives the divergence rate. If the observed choice maximizes the leading utility, its loss stays bounded. The same formula with the mean bracket holds on any fixed evaluation sample. In the single-facet case the first program chooses the exposing unit normal divided by the smallest positive gap, yielding the rate already used in the manuscript.

The calculation requires only the finite fitted design, corrected endpoint and inward slope. A linear program over menu masses with row totals \(w_i(0)\) and moment \(b(0)\) identifies \(T_i\). An alternative is in \(T_i\) exactly when its mass can be positive in some feasible solution. A functional exposing the minimal face rules out positive mass for other alternatives. Relative interior of the minimal face allows a representation using each of its generating alternatives. The example program checks each mass separately.

## Proof

Write \(\widehat\beta_\delta=u_\delta+n_\delta\) in \(E\oplus N\). Let \(p_{ij}(\delta)\) be its probabilities. The first-order condition matches their weighted moments to \(b(\delta)\). At the boundary, positive mass is feasible for every tied alternative simultaneously by averaging feasible witnesses. Weighted entropy has a unique maximizer among these feasible masses. It puts positive mass on every tied alternative, because the entropy gain from adding a previously zero feasible mass dominates the loss on positive masses near zero.

The softmax probabilities uniquely maximize weighted entropy given their menu totals and moment. For any feasible masses at zero and at some fixed positive \(\delta_0\), a convex mixture with weight \(\delta/\delta_0\) is feasible at \(\delta\), by affinity of weights and target. The feasible mass arrays are uniformly bounded. Along a subsequence, optimal masses converge to masses feasible at zero. Entropy is continuous, including at zero masses, and the mixture permits comparison against the optimal boundary masses. The limiting masses thus maximize boundary entropy and agree with its unique solution. For tied alternatives, fitted log probability ratios are exactly their utility contrasts applied to \(u_\delta\). Their positive probability limits and the span of these contrasts give \(u_\delta\to u_*\). This is the conditional fit in \(E\).

Projecting the score onto \(N\) gives the exact relation

\[
\sum_i\sum_{j\notin T_i} w_i(\delta)p_{ij}(\delta)a_{ij}=\delta s.
\]

An exposing direction is strictly positive on every outside gap. Taking its inner product proves each outside probability is \(O(\delta)\). Relative to a tied alternative, an outside probability is \(\exp(-a_{ij}^{\mathsf T}n_\delta)\) times a factor bounded above and away from zero, using \(u_\delta\to u_*\). With \(L=\log(1/\delta)\) we get \(a_{ij}^{\mathsf T}n_\delta\ge L-O(1)\). At any fixed interior \(\delta>0\) all probabilities are positive, so the same relation writes \(s\) as a strictly positive combination of the outside gaps. These gaps span \(N\) because all menu contrasts span the whole parameter space. This also proves compactness of the first optimal set. At fixed objective value, each gap on its feasible set is bounded above by the positive combination and the other gaps' lower bounds.

For fixed \(u=u_\delta\), the part of the objective varying with \(n\), up to an \(n\)-independent term, is

\[
\delta s^{\mathsf T}n + \sum_i w_i(\delta)\log\left(1+\sum_{j\notin T_i}C_{ij}(u)\exp(-a_{ij}^{\mathsf T}n)\right),
\]

where the factors \(C_{ij}(u)\) are positive and uniformly bounded above and below. Compare the minimum at \(n_\delta\) with \(Lv\) for any optimizer \(v\) of the first program. The logarithmic terms at the comparison point are \(O(\delta)\) and nonnegative at the minimum. This gives \(s^{\mathsf T}n_\delta\le L\min s^{\mathsf T}v+O(1)\). Adding a fixed multiple of the exposing direction makes \(n_\delta\) satisfy all gap lower bounds with right-hand side \(L\). The linear objective bound then holds in both directions up to \(O(1)\). Hoffman's error bound for a consistent finite system of linear inequalities, with the optimal objective enforced as an equality, gives \(n_\delta=Lz_\delta+O(1)\) for a point \(z_\delta\) in the first optimal set. [Hoffman's error bound](https://nvlpubs.nist.gov/nistpubs/jres/049/4/V49.N04.A05.pdf)

Suppose this distance bound holds on the current maximizing set \(V\). Write \(n_\delta=Lz_\delta+e_\delta\) with \(z_\delta\in V\) and bounded \(e_\delta\). Let \(\alpha\) be the best minimum over gaps that still vary on \(V\), and pick any maximizing \(z^+\in V\). Compare the minimum with \(Lz^++e_\delta\), keeping \(u=u_\delta\). The linear term and every constant-gap term agree at the two points. In every log, factor out one plus the sum of the constant-gap exponentials. These factors are bounded because all outside exponentials at the optimizer are \(O(\delta)\) and constant terms agree. The remaining log sum at the comparison point is \(O(e^{-L\alpha})\). At the optimizer it is bounded below by a positive multiple of the largest remaining exponential, using the strictly positive limiting menu weights. Optimality therefore gives \(\min_\mathrm{varying} a_{ij}^{\mathsf T}n_\delta\ge L\alpha-O(1)\). Apply the same error bound to the new maximizing set to preserve \(n_\delta=Lz_\delta+O(1)\) there. Each stage reduces the dimension. If the minimum equalled its maximum on the whole prior set, finitely many hyperplanes from nonconstant gaps would cover its relative interior, which is impossible. The remaining set is a singleton once no gap varies because the outside gaps span \(N\). Finally, finite log sums differ from largest utility by a bounded amount. Applying this to the evaluating menu yields the displayed rate and bounded remainder.

## Attribution

The endpoint entropy dual and conditional face use established exponential-family and recession geometry. The successive max-min gaps correspond to the center of an exponential-penalty trajectory in linear programming. The calculation here proves the same selection for grouped menu log-sum-exp terms with an estimated tangent fit, moving menu weights and a moving corrected moment. The prediction statement ties the selected direction to choice loss on a specified evaluation menu. The algorithmic center and the linear error bound are inherited tools. [Geyer](https://www.stat.umn.edu/geyer/gdor/) · [Cominetti and San Martín](https://doi.org/10.1007/BF01582224) · [Hoffman](https://nvlpubs.nist.gov/nistpubs/jres/049/4/V49.N04.A05.pdf)
