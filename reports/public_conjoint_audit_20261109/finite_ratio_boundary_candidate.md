# Candidate finite-auxiliary, power-dependent existence boundary

Research derivation,9 November2026. Independent numerical checks below are complete. Priority and independent proof review remain open before this enters the paper as a verified contribution. This addresses the same corrected-choice estimator as the existing derivation and includes a finite auxiliary/label ratio and a fixed power.

## Estimation problem and scope

Let m labelled and N auxiliary observations be independent, with X~N_p(0,I), Y~Bernoulli(1/2), and Z obtained by independently flipping Y with fixed probability ε in(0,1/2). The frozen oracle is g(Z)=ε+(1−2ε)Z. The model has p Gaussian slopes and fixes its intercept at zero. A free intercept or a nonzero utility signal is not covered by this finite-ratio argument.

For fixed0<λ≤1, minimize the published PPI objective

Q_λ(β)=L_m(Y;β)+λ{L_N(g;β)−L_m(g;β)}.

Assume N/m→τ in(0,∞) and p/m→κ. Define

f(a)=E[(G−a)_+²]=(1+a²)Φ(−a)−aφ(a), G~N(0,1),

and

F(t;τ,ε,λ)=
(1−ε)[f(−λεt)+f((1−λ+λε)t)]
+ε[f(−λ(1−ε)t)+f((1−λε)t)]
+τ[f(λ(1−ε)t/τ)+f(λεt/τ)].

The candidate boundary is

κ_c(τ,ε,λ)=1+τ−min_{t≥0}F(t;τ,ε,λ).

Below it the probability of a unique finite minimizer should tend to one; above it the probability should tend to zero. No equality-boundary limit is asserted. λ=0 has the separate classical ordinary-logit threshold1/2. Limits approaching zero power can be singular and should not be replaced by the zero-power formula.

## Exact cone reduction

The attainable weighted score moment is a classical exponential-family condition. Its feasibility is equivalent to a Gaussian linear map having a zero on a product of intervals. Orient each labelled X row by1−2Y and each auxiliary row by2Z−1. Conditional on all Y,Z, the oriented feature matrix A remains an independent standard Gaussian p by(m+N) matrix.

Multiplying the labelled score coefficients by their orientations gives these intervals:

- Labelled agreement Y=Z: [λε,1−λ+λε].
- Labelled disagreement Y≠Z: [λ(1−ε),1−λε].
- Auxiliary record: (λ/τ_m)[−(1−ε),ε], where τ_m=N/m.

Write D for their product and K=cone(D). The labelled lower endpoints are positive, so K contains no nonzero ray with all labelled coordinates zero. It is a closed convex cone. For0<λ<1 it is full dimensional; atλ=1 its span has dimension N+1. A nonzero point of ker(A)∩K can be rescaled into D, and conversely any zero of A on D gives a nonzero intersection.

For a generic Gaussian map, feasibility occurs away from tangency with probability one. When the positively weighted feature design has full column rank, this is the interior moment condition for a unique finite logit root. Atλ=1 the labelled coordinates are fixed, and conditional on their residual magnitudes their Gaussian linear combination is continuously distributed independently of the auxiliary design; hitting its moment boundary has probability zero. These qualifications must also be checked in the numerical cone/moment comparison.

## Statistical dimension and the scalar limit

For an independent standard Gaussian vector H in R^(m+N),

dist(H,K)²=min_{t≥0} Σ_i dist(H_i,tD_i)².

The classical statistical dimension identity is

δ(K)=(m+N)−E dist(H,K)².

The labelled disagreement fraction tends to ε. A uniform law on bounded t-intervals gives m⁻¹ dist(H,K)²→min F(t). Minimizers are tight: the positive labelled lower bounds, at leastλε, impose quadratic growth in t, while t=0 bounds the optimum by||H||². This also provides uniform integrability, giving δ(K)/m→κ_c.

The random-nullspace threshold is the established approximate conic kinematic theorem of [Amelunxen, Lotz, McCoy and Tropp2014](https://tropp.caltech.edu/papers/ALMT14-Living-Edge-preprint.pdf), not a new theorem about general cones. A codimension p separated from δ(K) by a positive multiple of m is larger than its order-sqrt(m+N) transition band. Conditional application followed by the labelled-count law gives the stated strict-side limit.

## Checkable reductions and practical meaning

Forλ=1, the primary contribution reduces to1+ε(1−ε)t². After rescaling t=τu,

κ_c(τ,ε,1)=τ[1−min_{u≥0}{τε(1−ε)u²+f((1−ε)u)+f(εu)}].

Asτ tends to infinity this recovers1/[2π ε(1−ε)], the earlier population-auxiliary candidate. A finite τ therefore cannot be replaced by the population limit without checking its effect.

For the informative conditional-mean oracle, the existing fixed-dimensional variance-optimal power isλ*=τ/(1+τ). Substituting that value yields a finite-ratio existence boundary at the relevant optimal weighting. The weight formula itself remains attributed prior work.

The boundary concerns an unpenalized root of a specified objective. It is not a lower bound on what every estimator can achieve. A classical convex-combination argument also shows that decreasing a positive power preserves coercivity when a larger power already has it; this monotonicity is not claimed as new. An exact quantitative phase curve could be useful for label and auxiliary budgets, but that use remains to be demonstrated and cannot be inferred from a computed curve alone.

## Numerical checks and remaining review

The100 same-draw comparisons of the original weighted-moment LP and oriented interval LP agree:43 interior and57 exterior cases, across positive powers includingλ=1. Independent quadrature checks f within8.9e-16; the unit-power reduction agrees within8.9e-16. Atτ=10000 the boundaries approach the independently derived population limits. Results are in [numeric checks](finite_ratio_boundary_numeric_checks.json).

The80 prespecified phase conditions,500 direct Gaussian draws each, retain all feasibility outcomes and have zero numerical-unresolved cases. They compare m=40,80, τ=1,4, ε=0.1,0.25, unit power and the existing population variance-optimal power. See [phase results](finite_ratio_phase_checks.csv). The finite-m curves are not claimed to equal an asymptotic step function.

Atλ=0 the intervals contain zero, so plain box feasibility becomes vacuous. Ordinary-logit existence requires the classical nonzero-kernel/interior criterion; its threshold1/2 is an attributed separate endpoint, not an output of this LP check.

The remaining review concerns equivalent results for weighted-logit separation, random-box feasibility and prediction-assisted high-dimensional inference. The cone machinery and generic feasibility calculations are established ingredients. Nonzero-signal finite-ratio extension and general discrete DCE validity have not been established by these checks.
