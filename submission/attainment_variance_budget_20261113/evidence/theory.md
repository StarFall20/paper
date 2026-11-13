# The largest compensated attainment region at a fixed regular covariance cost

13 November 2026. The derivation stays within the preceding balanced, known-scale compensated logit model. The finite-dimensional convex program below uses classical second-order cone constraints and KKT optimality. The specific objective is the previously derived compensated existence profile. The reviewed57-page manuscript remains unchanged while this consequence is assessed.

## 1 The same objective and a specified covariance budget

Retain X=D(1,S,W), known positive bounded scale D independent of the Gaussian profiles, fair human choices, proxy flip probability e in(0,1/2), and calibrated frozen g=e+(1−2e)Z. Normalize E D²=1. Fixed row weights enter all three compensated loss terms with the original sample-count denominators. At a fixed auxiliary ratio tau and correction power lambda>0, their existence boundary and regular covariance multiplier are

$$K(w)=\sup_{v\geq0}EJ(vw(D)D),\qquad F(w)=\frac{E[w(D)^2D^2]}{(E[w(D)D^2])^2}. \qquad (V1)$$

J is the scalar compensated profile in the preceding scale proof. It satisfies J(0)=0,J'(0)=sqrt(2/pi)/lambda and J''≤−e(1−e) on the nonnegative half-line. Full regular covariance is an established lambda,tau factor times F(w). This budget concerns fixed-dimensional oracle covariance; it imposes no proportional-dimensional error or coverage bound.

Let the scale take finitely many positive values d_j with probabilities p_j>0. A prescribed cost C≥1 asks for the largest possible K(w) over all positive weights with F(w)≤C. The original covariance-optimal weight is constant. The unconstrained existence-optimal weight is inverse scale, whose cost is C_inv=1/(E D)².

## 2 Exact convex reduction

Put x_j=vw(d_j)d_j and use the weighted inner product <x,y>_p=sum p_j x_j y_j. The covariance budget becomes

$$\|x\|_p\leq\sqrt C\,\langle d,x\rangle_p. \qquad (V2)$$

It is a second-order cone constraint. The requested largest boundary is exactly

$$B_C=\max_{x\geq0}\left\{\sum_jp_jJ(x_j):\|x\|_p\leq\sqrt C\langle d,x\rangle_p\right\}. \qquad (V3)$$

To prove equality, every feasible weight and scalar v produce a feasible x with the same profile value. Conversely, a positive maximizing x defines w(d_j)=x_j/d_j. The cone allows arbitrary positive common scaling, so optimizing along that ray recovers the maximum in(V3). The previous exact recession identity connects this profile to the actual weighted logit support event. It does not replace its fitted likelihood by a coefficient-equivalent virtual likelihood.

The objective is strongly concave and bounded above by a linear function of norm minus e(1−e)norm²/2. The feasible cone is closed and convex, so a unique maximizer exists. At C>1, a positive unweighted ray supplies Slater feasibility. A zero coordinate cannot be optimal: increasing it from zero raises the objective and reduces the cone constraint to first order. Thus the maximizing weights are strictly positive. At C=1,Cauchy–Schwarz restricts x to a nonnegative multiple of d; this is the original unweighted boundary. At C≥C_inv,the unique maximizer is x=v_*1, where v_* maximizes J; this is the inverse-scale endpoint.

For 1<C<C_inv,the constraint binds. A slack maximizer would be the unconstrained constant x, which violates the budget. For any larger budget below C_inv,the previous maximizer is strictly feasible; moving slightly toward v_*1 improves its objective. Hence B_C is strictly increasing on[1,C_inv]. This gives the global budget curve over finite-support row weights, beyond the previously examined power path.

## 3 Computable optimality checks

Write g(x)=norm(x)_p−sqrt(C)<d,x>_p. At an intermediate budget, positivity and Slater's condition give a multiplier eta>0 with

$$J'(x_j)=\eta\left\{\frac{x_j}{\|x\|_p}-\sqrt C\,d_j\right\},\qquad g(x)=0. \qquad (V4)$$

These conditions are sufficient for the global optimum by classical convex duality. They also show that optimized effective scales x_j increase strictly with d_j: for fixed eta and norm(x), the equation J'(x)−eta*x/norm(x)=−eta*sqrt(C)d has a strictly decreasing left side. Monotonicity of the row weights x_j/d_j is not needed for the result.

For a positive numerically feasible candidate x and eta≥0, let r_j=J'(x_j)−eta{ x_j/norm(x)_p−sqrt(C)d_j }. Strong concavity and convexity of g imply the explicit diagnostic bound

$$0\leq B_C-\sum_jp_jJ(x_j)\leq-\eta g(x)+\frac{\|r\|_p^2}{2e(1-e)}. \qquad (V5)$$

Indeed, for a feasible y, strong concavity bounds the objective difference by <J'(x),y−x>_p−e(1−e)norm(y−x)²/2. The multiplier term is at most−eta*g(x); maximizing the residual linear term minus the quadratic gives(V5). A floating-point implementation reports the primal violation,stationarity residual,slack,multiplier and computed bound separately. Interval arithmetic would be needed to turn numerical function evaluation into a rigorous machine-precision certificate.

## 4 A closed form after the auxiliary-ratio limit

Along the existing variance power lambda_tau=tau/(1+tau), take the proportional sample limit at each fixed tau before tau tends to infinity. The preceding profile calculation gives, uniformly on compact x sets,

$$J(x)\longrightarrow\sqrt{2/\pi}\,x-e(1-e)x^2. \qquad (V6)$$

For a fixed finite-support scale law,strong concavity and the bound on J'(0) localize every optimizer to one bounded set. Thus the optimized values in(V3) converge to the optimized quadratic problem. This is an iterated limit,with no joint sample/auxiliary-ratio assertion.

Let mu=E D in(0,1) and sigma=sqrt(1−mu²). For 1≤C≤1/mu²,the resulting budget curve is

$$B_C^{\infty}=\frac{\{\mu/\sqrt C+\sigma\sqrt{1-1/C}\}^2}{2\pi e(1-e)}. \qquad (V7)$$

At larger budgets it equals1/{2*pi*e(1−e)}. To derive(V7),write T=w(D)D,normalize norm(T)_2=1,and decompose 1=mu*D+sigma*U with U orthogonal to D and norm(U)=1. The covariance budget requires <D,T>≥1/sqrt(C). Maximizing E T gives

$$T_C= A_CD+B_C^0,\quad B_C^0=\frac{\sqrt{1-1/C}}{\sigma},\quad A_C=\frac1{\sqrt C}-\mu B_C^0. \qquad (V8)$$

Both coefficients are nonnegative on this budget interval,so T_C is positive and feasible. Cauchy–Schwarz in the orthogonal complement bounds every other feasible T by the value in(V7). Equality determines its direction uniquely. The corresponding row weight has the simple form A_C+B_C^0/D. For the quadratic limiting problem this argument also applies to any fixed bounded positive scale law; exchanging the finite-tau optimization limit is asserted here for finite support.

For a desired limiting boundary b between mu²/{2*pi*e(1−e)} and1/{2*pi*e(1−e)},put u=sqrt(2*pi*e(1−e)b). The smallest covariance multiplier reaching that boundary is

$$C_{\min}(b)=\{\mu u+\sqrt{1-\mu^2}\sqrt{1-u^2}\}^{-2}. \qquad (V9)$$

For lower targets it is one; higher targets are unattainable in this model. To obtain the strict existence side at a slope-to-label ratio kappa,the designed boundary must exceed kappa. Equality at the phase boundary has not been characterized.

## 5 Attribution and use

Second-order cone constraints,KKT sufficiency and concave optimization are classical; see Boyd and Vandenberghe,Convex Optimization,Sections4.4.2 and5.5.3. Weighted PPI and covariance tuning also remain prior. The candidate result is the explicit compensated attainment objective under this covariance constraint,with the finite-support global curve and its limiting formula. Comparison with a new preference-ranking PPI preprint is recorded separately; general variance-budget allocation is already studied there. This derivation does not establish priority or a new generic optimization principle.

The budget is specified before fitting. Learned scales,learned scale laws and data-selected budgets require further analysis. Under conditional logit misspecification the row weights change the projection target. The calculation supplies an existence/local-variance trade-off whose empirical error cost must be evaluated separately.
