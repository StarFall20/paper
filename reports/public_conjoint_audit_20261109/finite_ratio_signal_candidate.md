# Finite auxiliary ratio with utility signal and an estimated intercept

Research derivation begun 9 November 2026, updated 10 November 2026. This extends the same corrected binary-choice estimator. It is a mathematical candidate pending independent proof and priority review. Numerical identities and phase experiments are recorded separately; their agreement does not prove novelty.

## Model and proposed limit

Let X=(S,W), where S~N(0,1), W~N_q(0,I), and W is independent of (S,Y,Z). Choices obey P(Y=1|S)=σ(γS) for fixed finite γ. A binary proxy independently flips Y with probability ε in (0,1/2). Let g(S,Z) be any fixed measurable prediction strictly between zero and one. The numerical specialization uses the conditional-mean oracle

g(S,Z)=σ(γS+(2Z−1)log((1−ε)/ε)).

Estimate an intercept, the slope of S, and q orthogonal slopes. The true intercept and orthogonal slopes are zero. Labelled and auxiliary rows are independent, N/m→τ>0, q/m→κ, and the PPI power λ is fixed in (0,1]. The objective is the auxiliary-only Q_λ stated in [the null derivation](finite_ratio_boundary_candidate.md). The following statement does not treat fitted or data-dependent predictors, a growing-dimensional signal subspace, or clustered choice rows.

Define primary and auxiliary signs s_L=1−2Y and s_U=2Z−1. The random intervals are

I_L=[λr,1−λ+λr], r=|Y−g(S,Z)|,

I_U=(λ/τ)[−g,1−g] if Z=1,
I_U=(λ/τ)[g−1,g] if Z=0.

For a standard normal H independent of (S,Y,Z), let

f(a)=E[(H−a)_+²]=(1+a²)Φ(−a)−aφ(a),

B(h,t,a,b)=1+h²−f(−h−ta)−f(h+tb).

For η=(η₀,η₁), set

Ψ(η,t)=E B(s_L(η₀+η₁S),t,I_L)
       +τ E B(s_U(η₀+η₁S),t,I_U).

The candidate critical ratio is

κ_c(γ,ε,τ,λ)=inf_{η∈R²} sup_{t≥0} Ψ(η,t).

The strict-side claim is a unique finite minimizer with probability tending to one if κ<κ_c, and no finite minimizer with probability tending to one if κ>κ_c. No equality-boundary probability is asserted. The difference between q orthogonal slopes and q+2 estimated coordinates is finite and disappears in the proportional limit.

For the oracle, joint sign symmetry makes Ψ convex and even in η₀ at fixed η₁,t, so the infimum has η₀=0. More generally this reduction holds when g(−S,1−Z)=1−g(S,Z). Under that symmetry, computation is a scalar convex minimization whose objective contains a scalar concave maximization. An asymmetric frozen prediction retains both multiplier coordinates. The proof below does not require oracle calibration: the cancellation in Section 3 uses the same prediction distribution in the two independent samples.

## 1. Exact finite-dimensional event

Orient each labelled row by s_L and each auxiliary row by s_U. The two non-Gaussian feature rows are V₀=s and V₁=sS. The oriented orthogonal rows remain an independent standard Gaussian q by (m+N) matrix A. Use τ_m=N/m in the finite auxiliary intervals.

Let D be the product of all observed intervals, K=cone(D), and

C=K∩ker(V'), V=[V₀,V₁].

Every primary lower endpoint is strictly positive almost surely. K is closed, pointed and polyhedral. At λ<1 its span is all of R^(m+N); at λ=1 its primary coordinates have a common scale and its span has dimension N+1. The corrected moment is feasible exactly when a nonzero point of ker(A) intersects C. This is the classical attainable-moment reduction applied to the specified objective.

A feasibility event must meet the relative interior of the attainable moments to give a finite root. Section 3 below establishes, with probability tending to one, that ker(V') meets the relative interior of D. Conditional on this event, a Gaussian nullspace that meets C nontrivially meets its relative interior almost surely; boundary-only tangency has probability zero by generic polyhedral position. Below the critical ratio the positively weighted feature matrix also has full column rank almost surely. These conditions connect the cone event to a unique finite root; a bare zero-feasibility LP is not sufficient at zero power.

## 2. Statistical dimension identity

For fixed observed S,Y,Z and independent H~N_(m+N)(0,I), polyhedral polar calculus gives

C°=K°+range(V),

δ(C)=E_H inf_η dist(H−Vη,K°)²
    =E_H inf_η ||Π_K(H−Vη)||².

The sum in the polar identity is closed because the cones are polyhedral. Also

||Π_K(w)||²=sup_{t≥0} Σ_i{w_i²−dist(w_i,tI_i)²}.

Each summand is convex in w and concave in t: its variational expression is sup_{v∈tI_i}(2w_i v−v²), and squared distance to the scaled interval is jointly convex in (w,t). Thus the exact identity is an infimum followed by a supremum. No exchange of those two operations is required.

Integrating H_i yields B above. Moving random extrema through that integration requires tightness and uniform convergence; pointwise convergence alone does not justify the candidate limit. The next two sections supply those steps.

## 3. Uniform control of the signal/intercept multiplier

For a unit vector u=(u₀,u₁), let h_u(S)=u₀+u₁S and define the normalized empirical support

ρ_m(u)=m⁻¹Σ_label support_(I_L)(−s_L h_u)
      +m⁻¹Σ_aux support_(I_U)(−s_U h_u).

The intervals are bounded at a fixed positive τ and λ. Their support functions are Lipschitz in u with an integrable envelope proportional to 1+|S|. A uniform law of large numbers holds on the unit circle. The population support simplifies by cancellation of the common proxy terms:

ρ(u)=E[(-h_u(S))_++Y h_u(S)].

This is the recession function of the ordinary population logistic risk in direction −u. It is strictly positive for every unit u: σ(γS) lies strictly between zero and one for finite γ, and h_u is nonzero with positive probability. Continuity and compactness give c=min_(||u||=1)ρ(u)>0. Thus with probability tending to one, inf_u ρ_m(u)≥c/2. This also places zero in the interior of V'D, giving the relative-interior feasibility used in Section 1.

For any η, choose t=1 and in each interval choose the endpoint maximizing (−V_iη)v. The independent normal terms in the variational expression can be bounded below by −2 M_m Σ_i|H_i|, where all interval endpoints have a deterministic bounded maximum M_m as τ_m→τ. Their squared terms are bounded by a deterministic multiple of m. Consequently, for the empirical normalized saddle function Ψ_m,

sup_t Ψ_m(η,t) ≥ 2ρ_m(η)−O_p(1)
                   ≥ c||η||−O_p(1),

uniformly in η on that event. At η=0 the minimized objective is at most ||H||²/m=O_p(1). Every empirical minimizing multiplier is therefore bounded in probability. The same argument gives a bounded population minimizing set. This control uses an actual population overlap condition implied by finite γ; it is not assumed from pointwise convergence.

## 4. Uniform control of the cone scale

Fix a compact multiplier set. Put w_i=H_i−V_iη and a_i=λ|Y_i−g_i| on primary rows. For all t≥0,

dist(w_i,tI_i)² ≥ (ta_i−w_i)_+² ≥ t²a_i²/2−w_i².

The squared lower endpoints have a strictly positive expectation. The empirical average m⁻¹Σ_label a_i² converges to that value. Uniformly over bounded η, m⁻¹Σ_all w_i²=O_p(1). It follows that

Ψ_m(η,t) ≤ 2m⁻¹Σ_all w_i²−(t²/2)m⁻¹Σ_label a_i².

At t=0, Ψ_m=0. Every maximizing scale on the compact multiplier set is bounded in probability. The population version has the same bound. This covers λ=1 despite its lower-dimensional cone; it uses the positive primary lower endpoints rather than a full-dimensional-box assumption.

## 5. Limit and the classical random-subspace theorem

On compact multiplier/scale sets, the summands are continuous with an integrable envelope formed from H², S² and bounded interval endpoints. Uniform convergence holds for the independent labelled and auxiliary arrays. Since τ_m→τ, the scaled auxiliary interval endpoints converge as well. The tightness results above localize both extrema, giving

inf_η sup_t Ψ_m(η,t) → inf_η sup_t Ψ(η,t)

in probability under the joint S,Y,Z,H experiment. This step preserves the inf-sup order. The minimized empirical value lies between zero and ||H||²/m, whose second moment stays bounded. Uniform integrability upgrades the convergence to L1. Conditional expectation over H and Jensen's inequality give

δ(C)/m → κ_c

in probability over the observed low-dimensional signal and proxy variables.

Conditional on those variables, A has an independent Gaussian nullspace. The [ALMT2014 approximate conic kinematic theorem](https://tropp.caltech.edu/papers/ALMT14-Living-Edge-preprint.pdf) gives the strict-side transition when its codimension q is separated from δ(C) by a positive multiple of m. The O(sqrt(m+N)) conic transition band is smaller than that separation. Generic position and the interior/full-rank qualifications from Section 1 connect this event to the corrected-logit minimizer.

The statistical-dimension identity and kinematic theorem are established prior results. The candidate increment is their explicit population value for this finite-ratio corrected-choice problem, with utility signal and a free intercept.

## 6. Diagnostics and scope

At γ=0, additional symmetry makes η₁=0 optimal, recovering the preceding balanced-null formula. At λ=0 the auxiliary box collapses and the primary box contains zero. That endpoint must use K=R_+^m×{0}, not ordinary zero feasibility. Its separate value is

inf_η E[(H−s_L(η₀+η₁S))_+²],

the attributed [Candès–Sur ordinary-logit boundary](https://arxiv.org/abs/1804.09753). Zero power is excluded from the positive-power tightness argument because its primary lower endpoints vanish.

The prespecified [signal plan](finite_ratio_signal_plan.json) tests the oracle specialization: null reduction, Gaussian quadrature, saddle derivatives, intercept symmetry, direct cone projection, original-score versus oriented LPs, and finite-m phase curves. Its120 original/interval LP checks,12 independent cone-projection checks and80 phase conditions are complete; see [numeric checks](finite_ratio_signal_numeric_checks.json) and [phase results](finite_ratio_signal_phase.csv). Non-oracle phase experiments have not been run in this plan. This derivation does not establish categorical-DCE sample sizes, validity of data-selected powers, fitted-predictor boundaries or confidence-interval coverage.
