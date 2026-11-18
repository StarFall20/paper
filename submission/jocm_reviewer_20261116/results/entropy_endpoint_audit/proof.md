# Boundary slopes of the optimized choice objective and held-out loss

Use the finite-menu setting and the single-facet assumptions in Appendix C13 of the 18 November manuscript. For power \(\lambda\), the menu weight is \(w_i(\lambda)\), the corrected moment target is \(b_\lambda\), and the logit loss is
\[
Q_\lambda(\beta)=\sum_i w_i(\lambda)\log\sum_j e^{x_{ij}^{\mathsf T}\beta}-b_\lambda^{\mathsf T}\beta .
\]
Weights and target are affine. All active weights stay positive near a transverse endpoint \(c\in(0,1)\). The moments are interior immediately below \(c\). At \(c\), the target lies in the relative interior of a single facet with outward unit normal \(d\). Set \(\delta=c-\lambda\) and write \(\widehat\beta=t d+u\), \(u\perp d\). For menu \(i\), write \(M_i=\max_j x_{ij}^{\mathsf T}d\), \(g_{ij}=M_i-x_{ij}^{\mathsf T}d\), and \(T_i=\{j:g_{ij}=0\}\). The smallest positive gap is \(g_*\). The normal slack is \(a\delta\) with \(a>0\).

## Entropy representation and prior work

Let \(p_i\) be a probability vector on the available alternatives and let \(H(p_i)=-\sum_j p_{ij}\log p_{ij}\). The elementary log-sum variational identity gives
\[
V(\lambda)=\inf_\beta Q_\lambda(\beta)
=\max_{p_i\in\mathcal{S}_i:\,\sum_i w_i(\lambda)\sum_j p_{ij}x_{ij}=b_\lambda}
     \sum_i w_i(\lambda) H(p_i) .
\]
For every feasible \(p\), the right objective bounds \(Q_\lambda(\beta)\) from below. The fitted softmax probabilities are feasible at an interior optimum and give equality. At \(c\), feasibility forces every \(p_i\) to be supported on \(T_i\) because positive off-face mass creates strictly positive normal slack. The tied-set softmax at its finite tangent optimum \(u_*\) attains the entropy maximum. Sending \(t\) to \(+\infty\) at \(u_*\) makes the full loss approach this same value. This proves equality at \(c\) as an infimum.

Log-partition and entropy duality and exponential-family face limits are established results. Cominetti and San Martín and Weed study entropy-penalty paths with a fixed linear program. This representation of the correction-power path has unit entropy coefficient and affine movement in the moment and normalization restrictions for masses \(q_{ij}=w_i(\lambda)p_{ij}\). The published fixed-program convergence statements alone give no explicit formula for the changing-moment logit multiplier. The logarithmic coefficient expansion follows directly from standard softmax tails and the normal moment equation. Treat it as a specialized boundary calculation. [Wainwright and Jordan, Theorem 3.4](https://www.cs.columbia.edu/~blei/fogm/2025F/readings/WainwrightJordan2008.pdf) · [Rinaldo, Fienberg and Zhou](https://arxiv.org/abs/0901.0026) · [Cominetti and San Martín](https://www.ceremade.dauphine.fr/~carlier/Cominetti-SanMartin.pdf) · [Weed](https://proceedings.mlr.press/v75/weed18a.html)

## Value and derivative limits

Define
\[
Z_i^T(u)=\sum_{j\in T_i}e^{x_{ij}^{\mathsf T}u},\quad
A=g_*\sum_i w_i(c)\,\frac{\sum_{j:g_{ij}=g_*}e^{x_{ij}^{\mathsf T}u_*}}{Z_i^T(u_*)},\quad
B=\sum_i w'_i\log Z_i^T(u_*)-b'^{\mathsf T}u_* .
\]
The existing coefficient expansion gives \(u-u_*=O(\delta)\) and \(g_*t=\log(A/(a\delta))+o(1)\). For \(R_i=\sum_{j\notin T_i}\exp(x_{ij}^{\mathsf T}u-tg_{ij})/Z_i^T(u)\), the weighted sum of \(\log(1+R_i)\) is \((a/g_*)\delta+o(\delta)\). This follows by retaining the alternatives with gap \(g_*\) and applying \(a\delta=Ae^{-g_*t}[1+o(1)]\). The tied objective has zero tangent gradient at \(u_*\), so its value at \((\lambda,u)\) is \(V(c)-\delta B+O(\delta^2)\). The remaining normal term in the full loss is \(a\delta t\). Therefore
\[
V(\lambda)-V(c)=\frac{a}{g_*}\delta\{\log(A/(a\delta))+1\}-\delta B+o(\delta).
\]
For interior \(\lambda\), the derivative is the partial derivative of \(Q_\lambda\) at the optimizer. The normal part is \(-at\), the tied part tends to \(B\), and the off-face log sum contributes \(o(1)\). Thus
\[
V'(\lambda)=-\frac{a}{g_*}\log(A/(a\delta))+B+o(1).
\]

For the second derivative, let \(J_\lambda\) be the covariance Hessian at \(\widehat\beta\), written in normal and tangent coordinates, and \(q=\partial_\lambda\nabla Q_\lambda(\widehat\beta)\). The normal Hessian entry is \(a g_*\delta[1+o(1)]\), because the leading-gap alternatives contribute the leading weighted second gap moment and the square of each first moment contributes \(O(\delta^2)\). Cross entries are \(O(\delta)\). The tangent block converges to the positive-definite tied Hessian. The partial slope \(q_d=-a+O(\delta)\) and \(q_\perp=O(1)\). The Schur complement solve of \(J_\lambda\widehat\beta'=-q\) gives \(\delta t'\to1/g_*\) and \(u'=O(1)\). Consequently,
\[
\delta V''(\lambda)=-\delta q^{\mathsf T}J_\lambda^{-1}q\longrightarrow-a/g_* .
\]
These limits use derivatives of the smooth interior solution. Differentiating an unspecified \(o(1)\) in the coefficient expansion would be insufficient.

For any fixed evaluation menus, let \(\overline g^{\,e}\) be the mean gap of their observed alternatives along \(d\), using the available menu maximizer. The derivative of the evaluated loss with respect to \(t\) tends to \(\overline g^{\,e}\); its derivative in \(u\) stays bounded. Hence \(\delta L'_e(\lambda)\to \overline g^{\,e}/g_*\). With \(\overline g^{\,e}>0\), the optimized corrected objective decreases while evaluation log loss increases sufficiently close to \(c\). This conclusion concerns the declared normalization of \(Q_\lambda\) and continuation of its changing target. An additive differentiable offset with bounded derivative leaves the limiting derivative signs unchanged. The derivative summarizes the fitted objective. Forecast evaluation uses the separate held-out loss.

## Scope

The entropy representation explains the geometry in familiar terms and the derivative formulas are specialized consequences of the same normal-score argument. The choice-model contribution of these formulas is the precise local disagreement between the fitted objective and probability forecasts, and its measurement on actual menus. The fixed design argument leaves the proportional-dimensional existence theorem as a separate result. The derivative diagnostics reuse existing coefficients and evaluation respondents.
