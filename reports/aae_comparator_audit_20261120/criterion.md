# Matching all-soft and compensated choice criteria

Write $L_S(g;\beta)$ for the mean negative log likelihood on menu set $S$ using probability vectors $g$ as labels. Let $M_S(g)$ be the corresponding mean feature moment and let $\Delta=M_L(Y)-M_L(g)$ be the labelled residual moment. The negative log likelihood for one menu with normalized soft labels is its log partition function minus the soft-label moment times $\beta$. Consequently, with $L$ denoting labelled menus and $U$ auxiliary menus, the full compensated objective satisfies
$$
Q_\lambda(\beta)
=(1-\lambda)L_L(g;\beta)+\lambda L_U(g;\beta)-\Delta^T\beta,
\qquad 0\leq\lambda\leq1.
$$

At unit power, the first two terms reduce to the auxiliary-only criterion in equation (3) of Wang, Zhang and Zhang (2026). At $\lambda=N/(m+N)$ they give an equally weighted all-soft criterion on the pooled menus, corresponding to the menu pooling specified in their Section 5.3. Both comparisons use the same fixed probability array. On the pooled menus, the loss with observed labels on $L$ and soft labels on $U$ subtracts $[m/(m+N)]\Delta^T\beta$ from the all-soft criterion. Thus the three pooled objectives have the same log partition function and residual multipliers $0$, $m/(m+N)$ and $1$, respectively.

If $\Delta=0$, the all-soft and compensated losses agree at matched menu weights. With full contrast rank, a common finite optimizer implies $\Delta=0$ because their gradients differ by a constant. The first-order equations for a finite unpenalized first-stage Logit fitted on the same labelled menus give $\Delta=0$ when that model contains the focal attribute score directions. The identity is conditional on the realized predictions. Reusing labels to fit such a model calls for its own inference analysis. Regularization and out-of-fold predictions generally do not obey the in-sample score equations. Van der Laan and van der Laan (2026, Section 3.1) show the corresponding mean-estimation principle with a constant calibration direction. Lee and Kim (2026, Section 4) study population conditional-score calibration and first-order efficiency.

## The largest feasible residual multiplier

Fix either the auxiliary or the pooled menu weights $w_i$ and their all-soft moment $b_0$. Let $K$ be the set of feasible mean feature moments on these menus. If each prediction vector is strictly positive and the positive-weight contrasts span the parameter space, $b_0$ is interior to $K$. For $\Delta\ne 0$, the ray $b_0+a\Delta$, $a\geq0$, reaches its unique last feasible point at
$$
a_* = \min_{d:\Delta^Td=1}\{h_K(d)-b_0^Td\},
$$
where $h_K(d)=\sum_i w_i\max_j x_{ij}^T d$. To see this, the half-space description of $K$ imposes $a\Delta^Td\leq h_K(d)-b_0^Td$ for all $d$. Directions with $\Delta^Td\leq0$ impose no upper bound on $a$. The remaining directions normalize to $\Delta^Td=1$. A supporting direction at the ray endpoint has a positive dot product with $\Delta$ because $b_0$ is interior. At $0\leq a<a_*$ the moment is interior by convexity and the loss attains a unique finite minimum. At $a=a_*$ the moment lies on the boundary and a finite coefficient cannot attain it. At $a>a_*$ a minimizing dual direction gives strictly negative recession slope $a_*-a$. If $\Delta=0$, every residual multiplier has the original interior moment.

The binary calculation uses a primal LP with one probability per menu and a separate dual LP with one bound for the support of each menu. It holds the menus and their weights fixed throughout. The frontier measures the size of this particular residual moment that the specified menus can support.

## Sources

- [Wang, Zhang and Zhang, Marketing Science version of record](https://doi.org/10.1287/mksc.2025.0009) and [authors' v3, Sections 3.3 and 5.3](https://arxiv.org/html/2412.19363v3).
- [Van der Laan and van der Laan, Section 3.1](https://arxiv.org/html/2604.21260#S3.SS1).
- [Lee and Kim, Section 4](https://arxiv.org/html/2606.08730#S4).
- [Angelopoulos, Duchi and Zrnic, PPI++ v2](https://arxiv.org/abs/2311.01453v2).
