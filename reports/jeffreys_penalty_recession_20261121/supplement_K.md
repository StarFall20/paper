## Appendix K Jeffreys penalties for the corrected objective

The conventional Jeffreys log-information penalty gives finite ordinary binary Logit estimates under full rank (Kosmidis and Firth, 2021). Kessels et al. (2019) use Firth estimation for discrete choice data. For the corrected mean loss (2), let $I_\lambda(\beta)=\nabla^2 Q_\lambda(\beta)$ and add $-a\log \operatorname{det} I_\lambda(\beta)$ with $a\geq0$. The convention of applying one half of the log determinant to $mQ_\lambda$ gives $a=1/(2m)$ on the mean scale. A constant overall rescaling of the information changes the penalized loss only by a constant.

For each unordered pair in a positively weighted menu, form $v_{ijk}=x_{ij}-x_{ik}$. Write $\Delta_{ij}(d)=\max_{h\in A_i}x_{ih}^Td-x_{ij}^Td$. Let a basis $B$ consist of as many independent pair rows as there are coefficients. Define

$$T(d)=\min_B\sum_{(i,j,k)\in B}[\Delta_{ij}(d)+\Delta_{ik}(d)]. \qquad (K.1)$$

Proposition K1. Assume the active pair rows span the coefficient space. The penalized objective equals $Q_\lambda^\infty(\beta)+aT(\beta)$ up to a bounded remainder on the whole coefficient space. If this homogeneous function is positive in every nonzero direction, a finite minimizer exists. A negative direction gives an objective unbounded below. If the function is nonnegative and vanishes in a nonzero direction, the infimum is finite and the criterion fails to be coercive.

The information matrix is the sum of $w_i p_{ij}p_{ik}v_{ijk}v_{ijk}^T$ over these pairs, where $p_{ij}$ is the fitted probability. The coefficient of each pair is bounded above and below by positive constants times $\exp(-\Delta_{ij}(\beta)-\Delta_{ik}(\beta))$. These constants are uniform in $\beta$ because the logit denominator after factoring out its largest term lies between one and the number of available alternatives. By Cauchy–Binet the determinant is a sum of positive basis terms. Its exponent is governed by the least costly basis, which gives $-\log\operatorname{det} I_\lambda(\beta)=T(\beta)+O(1)$ uniformly. The log-partition terms likewise differ from their maxima by bounded remainders. A nonzero direction cannot have a zero-cost basis because each zero-cost pair is orthogonal to that direction. Thus $T$ is positive on the unit sphere. Continuity and homogeneity give the three stated conclusions. At the threshold, the argument classifies boundedness and coercivity. Attainment can require a separate check.

For binary menus, (K.1) reduces to the minimum sum of $|v_i^Td|$ over bases. If the unpenalized objective has a negative recession direction, the smallest coefficient above which the objective is coercive is

$$a_*=\max_{d\ne0}\frac{-Q_\lambda^\infty(d)}{T(d)}. \qquad (K.2)$$

When $a<a_*$ the objective is unbounded below. For a certificate along one direction, it suffices to provide one independent basis and show that even its larger cost leaves a negative penalized slope.

In Table 1 design C, the positive slope direction has cost $-1/32$ and its least basis cost is one. The human-sample convention gives $a=1/64$ and penalized cost $-1/64$. The strict coercivity threshold in this design is $1/32$. To see this, write $z$ and $b$ for the intercept and slope directions. The unpenalized cost is $7|z|/16+\max(|z|,|b|)/16-3b/32$. On $0\leq|z|\leq b$ the least basis cost equals $b$. Outside this cone the unpenalized cost is already positive for nonzero directions. This derives the global threshold and exhibits descent under the conventional coefficient.

We checked the same penalty in the 19 exterior vaccine designs selected in Appendix J. We used their previously certified separating directions and recovered a full-rank basis of active menu contrasts for each. All 19 rays still certify unboundedness under $a=1/(2m)$. At 50 labelled choices and weight 0.75, the upper bounds on the penalized slopes round to $-0.000344$ with the human-trained predictor and $-0.002117$ with the AI-trained predictor. These are tests on selected exterior designs. For ordinary human-only Logit the recession cost is nonnegative, so any positive coefficient gives coercivity by Proposition K1.
