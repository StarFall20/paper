# Frequency replacement and finite choice estimation

This check extends the fixed-support construction in Appendix J. It holds the observed human choices, prediction rule, correction weight and library of menus fixed. It asks how much of the nominal auxiliary menu distribution can be replaced before some reallocation makes the corrected logit objective lose its finite minimizer.

## Design perturbation

Let $q_j>0$ be the nominal frequency of menu $j=1,\ldots,J$, with frequencies normalized to sum to one. Let $q(t,r)=(1-t)q+tr$, where $r$ is any probability vector on those same menus. This is proportional removal from the nominal distribution followed by unrestricted replacement across the library. For $t<1$, all original menus keep positive weight. The total variation change is $t\,\mathrm{TV}(r,q)$ and can vary across replacement distributions at a given $t$. The mixture follows the contamination form of Huber (1964). [Huber](https://doi.org/10.1214/aoms/1177703732)

## A finite set of linear programs

Write $x_{ia}$ for the feature vector of labelled alternative $a$ in menu $i$, $x^U_{ja}$ for its counterpart in the auxiliary library, and $\mu_j=\sum_a g_{ja}x^U_{ja}$ for the fixed predicted menu moment. Let $b_L$ and $\mu_L$ be the average observed and predicted feature moments on the $n$ labelled choices. At correction weight $0<\lambda<1$, the corrected target is feasible exactly when there exist nonnegative masses $z^L_{ia},z^U_{ja}$ satisfying

$$
\sum_a z^L_{ia}=(1-\lambda)/n,\qquad
\sum_a z^U_{ja}=\lambda\{(1-t)q_j+t r_j\},
$$

$$
\sum_{i,a}z^L_{ia}x_{ia}+\sum_{j,a}z^U_{ja}(x^U_{ja}-\mu_j)
=b_L-\lambda\mu_L .
$$

For each library menu $k$, set $r=e_k$ and maximize $t$ over these constraints and $0\le t\le1$. Write its value as $\tau_k$ and set $\tau=\min_k\tau_k$. Assume the nominal corrected fit is finite, with full rank on the active within-menu contrasts. For all $0\le t<\tau$, every replacement distribution $r$ yields a unique finite fit. If $\tau<1$, there is a vertex replacement with a finite unattained infimum at $t=\tau$ and an objective unbounded below for $\tau<t<1$.

At each vertex replacement with $t<\tau$, interpolate the nominal interior masses with a feasible mass vector from a larger replacement fraction. The affine constraints preserve feasibility, and the positive nominal mass makes the interpolated masses strictly positive. At fixed $t$, mix these positive vectors with weights $r_k$ to cover the replacement simplex. At a first failing vertex with $\tau<1$, the optimum of the linear program is feasible and lies on the boundary. If it had a strictly positive mass representation, a small increase in $t$ could be absorbed while preserving the moment equation because the within-menu contrasts span the coefficient space. Any larger $t$ along that vertex is infeasible.

With a finite full-rank ordinary labelled fit, the threshold is nonincreasing in $\lambda$. At fixed design, the criterion at a lower positive weight is a convex combination of ordinary logit and the criterion at the higher weight. If the latter has an interior target, both criteria grow without bound as the coefficient norm grows. Their combination does as well.

## Checks on released choices

We use the first previously defined auxiliary draw in each of 37 ordinary-interior allocations from the released RADIANT food workbook. Each draw contains 384 choices on 34 menus. The original and perturbed samples use the same frozen prediction on any common menu. Each weight gives 36 baseline interior cases and one exterior case. The table summarizes the 36 baseline interior cases at each weight.

| Correction weight | Minimum replacement threshold | Median replacement threshold | Threshold at most 25% |
| ---: | ---: | ---: | ---: |
| 0.75 | 14.00% | 44.25% | 10 of 36 |
| 0.90 | 8.34% | 33.38% | 11 of 36 |

The computation solves 2,448 vertex programs and checks all 1,224 paired vertex values for the predicted ordering across weights. In each of 72 interior cases, two original-score support formulations recover an interior and an exterior fit on opposite sides of the calculated threshold. An analytical toy instance gives $79/105$, which agrees with the linear program. The cell counts describe overlapping allocations on one released dataset. The thresholds describe worst-case frequency reallocation under the proportional replacement scheme. Field prevalence would require a separate sampling model.

Liu et al. (2026) optimize an upstream calibration rule against Wasserstein shifts and then study the resulting soft-label preference estimates. This check fixes the prediction rule and classifies feasibility of the signed correction across a specified family of auxiliary frequencies. [Liu et al., Sections 3.4 and 4.3](https://doi.org/10.2139/ssrn.7500939)

The submission reproduction archive contains the full numerical table and the archived choice-design scripts. Use that archive with the public RADIANT workbook. The recorded result hash identifies the complete per-menu output, and `summary.json` reports only aggregate findings.
