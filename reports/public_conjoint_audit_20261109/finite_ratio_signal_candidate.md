# Nonzero-signal extension: derivation to verify next

Research outline only. The balanced-null finite-ratio candidate has a complete conditional cone argument; this extension still needs a proof of the limiting minimax step and direct numerical checks. It must not be reported as a proved theorem.

Let X=(S,X_perp), S~N(0,1), and P(Y=1|S)=sigmoid(γS). Z flips Y with probability ε and g(S,Z)=sigmoid(γS+(2Z−1)log((1−ε)/ε)). Let the true intercept be zero but permit estimation of an intercept. The remaining Gaussian coordinates are independent of S,Y,Z. The PPI power is fixed in(0,1] and N/m→τ.

After the same sign orientation as the null case, define primary sign s_L=1−2Y and auxiliary sign s_U=2Z−1. The intervals are

I_L=[λr,1−λ+λr], r=|Y−g|,

I_U=(λ/τ)[−g,1−g] if Z=1,
I_U=(λ/τ)[g−1,g] if Z=0.

The product-interval cone K now depends on S,Y,Z. The signal and intercept contribute two non-Gaussian rows V_0=s and V_1=sS, and the other p−1 rows remain independent standard Gaussian. The exact finite-dimensional event reduces to a random Gaussian nullspace intersecting

C=K∩{v:V_0'v=V_1'v=0}.

For a fixed realization, polyhedral polar calculus gives

δ(C)=E_H min_η ||Π_K(H−Vη)||²
     =E_H min_η sup_{t≥0} Σ_i{(H_i−V_iη)²−dist(H_i−V_iη,tI_i)²}.

The Gaussian tail function f from the null derivation integrates out H. For an interval[a,b] and shift h, the expected summand is

B(h,t,a,b)=1+h²−f(−h−ta)−f(h+tb).

The proposed limiting normalized statistical dimension is

inf_{η0,η1} sup_{t≥0} [
 E B(s_L(η0+η1S),t,I_L)
 +τ E B(s_U(η0+η1S),t,I_U)].

Here the notation B(h,t,I) uses I's two endpoints. Each expectation is over S,Y,Z under the specified model. Sign symmetry suggests η0=0 at the minimizer: the objective is convex and even in η0 for every fixedη1,t. This leaves a two-scalar saddle problem overη1 and t. The claimed critical ratio would be this value, by the established random-subspace theorem, after proving the limiting interchange and tightness.

Required checks are substantial: coercivity/tightness in the signal multiplier, uniform convergence over the saddle variables, the relative-interior condition, and handling the unit-power lower-dimensional cone. Pointwise convergence is insufficient to exchange random minima, expectations and limits.

Two reductions provide strong diagnostics. Atγ=0 the signal multiplier should minimize at zero and recover the verified null scalar formula. Atλ=0, treated as a separate endpoint, the auxiliary intervals collapse and the labelled intervals become[0,1]; the result should recover the established [Candès–Sur ordinary-logit separation boundary](https://arxiv.org/abs/1804.09753). Neither that baseline nor the [ALMT conic machinery](https://tropp.caltech.edu/papers/ALMT14-Living-Edge-preprint.pdf) is a proposed innovation.

This is a natural extension of the same estimator problem. Its status is an unverified mathematical candidate, not another contribution claim or a reason to defer reporting the completed finite-pool comparison.
