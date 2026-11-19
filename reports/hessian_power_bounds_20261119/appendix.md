### B8 Sharp bounds from the zero-point Hessian

Keep the labelled sample and predictor in Table 1. Suppose there are $n$ auxiliary contrast pairs $(u_i,-u_i)$ with $z=0$, and $n$ rows at each of the two fixed informative profiles. The magnitudes satisfy $0\leq u_i\leq a$. Write $q=\sum_i u_i^2/n>0$ and $M=\sum_i u_i/n$. The slope entry of the auxiliary Hessian at zero is $q/8$. The local objective, gradient, Hessian and empirical score covariances depend on the magnitudes only through $q$. This statement holds for all correction weights. In particular, the varying rows have zero auxiliary scores at zero.

The exact compatible interval for $M$ has endpoints

$$M_-=\{ka+\sqrt{nq-ka^2}\}/n,\qquad M_+=\sqrt q,\qquad k=⌊nq/a^2⌋.$$

For the upper endpoint, equal magnitudes attain equality in Cauchy–Schwarz. For the lower endpoint, place $y_i=u_i^2$ in $[0,a^2]$ at fixed sum $nq$. If two entries are interior, transfer mass from the smaller to the larger until one reaches an endpoint. Concavity of the square root makes the sum of roots weakly decrease. Iterating leaves at most one interior entry, which gives $M_-$. The feasible squared vectors form a convex set. The segment between these two optimizing vectors realizes every intermediate mean magnitude by continuity.

Symmetry gives a zero profiled intercept, and the positive-slope recession of (2) is $\{(1-\lambda)/2+\lambda M-3/8\}/4$. The negative-slope recession is strictly positive. For $q>0$ the profiled score increases strictly, so a positive, zero or negative right-hand limit gives the three respective attainment states. Substituting the endpoints of the interval gives sharp bounds on this right-hand limit. At a specified positive weight, all compatible designs have a finite solution exactly when the lower bound is positive. For Table 1, $n=8$, $a=1$ and $q=1/4$. The compatible upper limits of feasible weights fill $[1/2,1]$. A design can reach the boundary at its limit.

With three pairs and the same cap and mean square, $M_-=\sqrt{3}/6$. Every compatible design gives a finite solution at weight 0.55. At 0.6, three compatible designs reproduce all three outcomes. The original-row support programs verify the witnesses. The general bounded-feature Newton-decrement sufficient condition in Bach (2010, Proposition 2) compares 0.530330 to 0.0625 at the shared zero-point diagnostic with the public cap. The exact bound here also uses the fixed count and paired structure.
