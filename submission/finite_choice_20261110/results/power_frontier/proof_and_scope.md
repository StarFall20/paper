# Finite-design admissible powers and the closest new priors

## A classical support calculation on the existing objective

Keep the observed menus, choices and frozen predictions fixed. Combine identical attribute menus by their exact summed weights. Write their curvature weights as w(lambda)=wL+lambda(wU-wL), and their corrected moment as b(lambda)=bY+lambda(bUg-bLg). The feasible weighted probabilities z satisfy

    z_kj >= 0
    sum_j z_kj = w_k(lambda)
    sum_kj F_kj z_kj = b(lambda)
    0 <= lambda <= 1.

These are linear constraints. Their joint feasible set is compact, and lambda=0 is feasible using the observed choices. Its projection onto lambda is a closed interval [0,c]. One linear program maximizes c. For c=1 this records only the endpoint of the convex power domain; it does not certify interior probabilities or auxiliary curvature rank at unit power.

Suppose the stacked positively weighted contrast rows have full parameter rank and some lambda0 in (0,c) has strictly positive probability coordinates. Convex combinations of that witness with the feasible endpoint at zero or c give strictly positive coordinates at every lambda in (0,c). Thus either the open interval has an interior witness throughout or none of its points does. Testing c/2 distinguishes these structural possibilities in exact arithmetic. If c<1 and its target were interior, full contrast rank would allow a small moment perturbation beyond c, contradicting maximality. Its endpoint is therefore a boundary target. The floating-point implementation retains numerical zero and proximity states separately.

This is a direct use of classical convex support and parametric linear programming. It is a computational consequence for the manuscript's existing loss, not the independent theoretical contribution. A finite root remains distinct from a useful probability forecast.

## Independent upper witnesses

For each uncapped optimum, the equality dual supplies a coefficient direction d. On original task rows the recession is affine in power:

    R(lambda;d) = H_infinity(d) + lambda * {prediction_U_infinity(d) - prediction_L_infinity(d)}.

The stored dual is normalized so its slope is -1; R(c;d) is numerically zero. This proves an upper support bound, independently checked by direct task-level maxima and actual predicted responses. A primal witness proves the matching feasible bound. Twenty-nine separately assembled, uncompressed LPs check all actual budget/cap/midpoint strata. No Gaussian generic-position argument is used for this categorical panel.

## A newly located direct empirical prior

[Vadlamani, Chang, Parthasarathy and Paul (2026), TC-PPI v1](https://arxiv.org/html/2609.34156v1), Section 5 and Appendix F, already report quasi-separation and unstable low-label Bradley–Terry fits. Appendix C.1 writes the corrected GLM score; Theorem 7 explicitly fixes the parameter dimension. Sections C.3–C.5 give local regularity and root results. The general empirical failure mechanism and that score are established prior. The present proportional-dimensional boundary and its matched-information consequences require their own comparison.

## Exact containment in a general score framework

[Zhang, Zhang and He (2026), DiPPI v1](https://arxiv.org/html/2609.32006v1), Sections 2.1–2.3, already retain categorical predictive probabilities. Theorem 1 and Appendix B.1 establish a local root under regular smooth fixed-dimensional conditions. Its pooled score formula also contains the manuscript's scalar corrected score. Let r=m/N and let U(beta;X,y) be the human choice score. Substituting

    h(beta;X,Q) = (1+r) * lambda * U(beta;X,g)

into their pooled formula gives exactly

    average_L U(beta;X,Y) - lambda*average_L U(beta;X,g)
      + lambda*average_U U(beta;X,g).

The factors cancel because the pooled mean includes the labelled rows. This identity applies to the common-coefficient menu score F'*(p(beta)-y) as well. It is our algebraic comparison, not a claim that their fitted conditional-score estimator always optimizes the manuscript's objective. Their local existence result must receive attribution; retaining a probability simplex or introducing generic score correction is not new.

## Other screened overlap

[Zhou (2026), Black-Box Assisted Regression v1](https://arxiv.org/html/2606.25743v1), Sections 3–5, studies nonparametric squared-error minimax rates indexed by predictor proximity. Its phase terminology does not by itself identify the same finite-minimum event. [PPI-SVRG v1](https://arxiv.org/html/2601.21470v1), Sections 3–5, studies optimization updates and conditional error floors; Theorem 5.2 assumes strong convexity of the human empirical loss. These readings supply no automatic novelty certificate. [Local PPI](https://arxiv.org/html/2409.18321), Section 2.1, discusses negative kernel weights and potentially unbounded local regression losses, a different use of negative weights. It is not evidence that all prediction corrections fail.

The direct theorem comparison against category-based M-estimation and general Gaussian kinematics remains in the [preceding report](https://github.com/StarFall20/paper/tree/backup/dh36-20261024/reports/direct_theorem_comparison_20261118). The new search does not establish global priority or sufficient JCM value. Its concrete editorial consequence is to cite the direct Bradley–Terry failure prior and retain only the specific boundary calculation and controlled consequences as candidate originality.
