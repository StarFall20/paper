# A single covariance solve can certify global support

Let $S_i$ be a finite nonempty choice set, $w_i>0$ its weight, $x_{ij}$ its available feature vectors, and $b$ the corrected moment. Consider

$$
L(\beta)=\sum_i w_i\log\sum_{j\in S_i}\exp(x_{ij}^{\mathsf T}\beta)-b^{\mathsf T}\beta.
$$

Write $C=\sum_i w_i\operatorname{conv}\{x_{ij}:j\in S_i\}$, the weighted Minkowski sum. For arbitrary strictly positive probabilities $p_i$ on each available set, define

$$
\mu_i=\sum_j p_{ij}x_{ij},\qquad
H_p=\sum_i w_i\operatorname{Cov}_{p_i}(x),\qquad
s=\sum_i w_i\mu_i-b.
$$

Assume $H_p$ is positive definite. This is equivalent to the menu contrasts spanning the coefficient space. Set $d=-H_p^{-1}s$ and define

$$
q_{ij}=p_{ij}\{1+(x_{ij}-\mu_i)^{\mathsf T}d\},\qquad
r(d)=\sum_i w_i\max_{j\in S_i} x_{ij}^{\mathsf T}d-b^{\mathsf T}d.
$$

1. If every $q_{ij}>0$, then $b\in\operatorname{int} C$ and $L$ has a unique finite minimizer.
2. If $r(d)<0$, then $b\notin C$ and $\inf L=-\infty$.
3. These two conclusions are mutually exclusive and invariant under a nonsingular linear recoding of the features. The original support calculation resolves the remaining cases.

The change in each row sum is zero by definition of $\mu_i$. The change in weighted moment equals $H_p d=-s$. Positive probabilities on every available alternative give $b$ as a relative-interior mixture. Equivalently, for any nonzero $u$, the projection $u^{\mathsf T}x_{ij}$ varies within at least one menu and hence $u^{\mathsf T}b<h_C(u)$. Full span gives interior. Interior gives a uniform positive gap over the unit sphere and coercivity of $L$. The log-sum-exp Hessian is positive definite for every finite coefficient vector under the same contrast-span condition, which gives uniqueness. Along $td$ for $t>0$, the difference $L(td)-t r(d)$ is bounded between zero and $\sum_i w_i\log |S_i|$. This proves the second claim. For $x'_{ij}=Ax_{ij}$, $b'=Ab$, and nonsingular $A$, the direction becomes $d'=A^{-\mathsf T}d$, leaving the menu products and unnormalized recession unchanged. [Geyer (2009)](https://www.stat.umn.edu/geyer/gdor/)

The signed solution $q$ is also the unique minimizer of
$\tfrac12\sum_i w_i\sum_j (q_{ij}-p_{ij})^2/p_{ij}$ with the menu totals and corrected feature moment imposed as equalities. Indeed, any feasible perturbation $e$ has zero row sums and zero weighted feature moment. Its cross term with the proposed solution is $\sum_i w_i\sum_j e_{ij}(x_{ij}-\mu_i)^{\mathsf T}d=0$, so the objective increases by a strictly positive quadratic term for each nonzero perturbation. This is chi-square calibration with menu indicators and the model features as calibration variables. At fitted softmax probabilities $d$ also equals the Newton direction of $L$. Frozen auxiliary predictions provide a second admissible anchor when their available probabilities are strictly positive. [Deville and Särndal (1992)](https://www.tandfonline.com/doi/abs/10.1080/01621459.1992.10475217)

Writing $\nu^2=s^{\mathsf T}H_p^{-1}s$ and $R=\max_{ij}\sqrt{(x_{ij}-\mu_i)^{\mathsf T}H_p^{-1}(x_{ij}-\mu_i)}$, Cauchy–Schwarz gives the sufficient margin $\nu R<1$. The componentwise inequalities used in claim 1 retain the signs of individual changes. The proof uses exact arithmetic. The audit reports numerical margins above $10^{-8}$, original-unit normalized recession below $-10^{-8}$, and equality residuals below $10^{-10}$.
