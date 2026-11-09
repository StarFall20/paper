# Finite estimability at matched prediction risk and asymptotic covariance

Research proposition, 10 November 2026. This is a consequence of the same corrected-choice boundary, not a second estimation framework. Priority and independent proof review remain open. Its numerical effect is small in the displayed cases with N/m≥1, which limits its present practical significance.

## A controlled comparison

Let Y~Bernoulli(1/2), independently of the Gaussian utility attributes. An ancillary confidence amplitude C lies in [0,c_max] for c_max<1/2. Conditional on C, the binary proxy Z flips Y with probability ε(C); the flip is independent of utility attributes. The fixed prediction is

g=1/2+C(2Z−1).

The model includes a free intercept. Labels and auxiliary rows are independent samples from this same law. Define

v=E C², K=E[C{1/2−ε(C)}], R=E(Y−g)²=1/4+v−2K.

Both E g=1/2 and Var(g)=v. The covariance Cov(Y,g)=K. Thus two predictors with the same v and K have the same Brier risk R.

At the true zero utility, the fixed-dimensional sandwich covariance for power λ is

16{1/4−2λK+λ²v(1+1/τ)} I / m.

This includes the intercept and all Gaussian slopes. The formula follows from the existing PPI score-covariance identity: the labelled score is X{−(Y−1/2)+λ(g−1/2)} and the auxiliary score is −λX(g−1/2), while the population Hessian is I/4. Therefore matching v,K also matches the whole fixed-dimensional asymptotic covariance for every fixed λ and matches its population optimal scalar weight. This covariance identity is prior methodology, not the proposition's proposed novelty.

## Candidate proposition for unit power

At λ=1, the finite-estimability threshold in dimensions/labels is

κ_c=τ[1−min_{u≥0}{τRu²+E h_u(C)}],

h_u(c)=f((1/2+c)u)+f((1/2−c)u),

where f is the normal positive-part second moment used in the main derivation. This follows from the primary singleton intervals, whose squared distance contributes only R, and the auxiliary intervals, whose contribution retains the distribution of C. Sign symmetry makes the intercept multiplier zero.

For fixed R and v, increasing C² in convex order strictly lowers κ_c whenever the two confidence distributions differ. In particular, a nonconstant C with E C²=v has a lower threshold than constant C=sqrt(v), when both comparisons can be constructed at the same R.

Proof: for u>0, define H_u(x)=h_u(sqrt(x)). Since

f''(a)=2Φ(−a), f'''(a)=−2φ(a),

h_u'''(c)=2u³{φ((1/2−c)u)−φ((1/2+c)u)}>0

for c>0. Also h_u'(0)=0. Writing c=sqrt(x),

H_u''(x)={c h_u''(c)−h_u'(c)}/(4c³)>0,

because the numerator is the integral from0 to c of h_u''(c)−h_u''(s). The positive continuous limit at x=0 follows by Taylor expansion. Hence H_u is strictly convex. The definition of convex order gives the pointwise strict comparison E h_u(C₂)>E h_u(C₁). The minimizing u is positive because the derivative of the bracket at u=0 is −2φ(0), and R>0 gives quadratic growth. Evaluate the strict comparison at the minimizer for C₂ to obtain κ_c(C₂)<κ_c(C₁).

Convex order and Jensen's inequality are established tools. The candidate statement concerns their explicit consequence for finite estimability of this corrected logistic objective at matched risk and covariance.

## Exact illustrative construction and effect size

Set c=0.4 and h=0.499. Predictor A uses C=c. Predictor B uses C=h with probability p=c²/h² and C=0 otherwise. Let

K=0.9 p h/2,
ε_A=1/2−K/c,
ε_B(h)=0.05,
ε_B(0)=1/2.

Both predictions are strictly inside (0,1). They have v=0.16, K≈0.14428858 and R≈0.12142285. Their full fixed-dimensional covariance functions agree for all fixed powers. B has greater dispersion of C², so its unit-power existence threshold is strictly smaller.

The deterministic table is [matched_moment_confidence_boundaries.csv](matched_moment_confidence_boundaries.csv). At τ=0.05 the thresholds are approximately0.03253 and0.03058, a relative reduction of about5.98%. At τ=1 the difference is about0.12%; at τ=4 it is much smaller. The example establishes that risk and fixed-dimensional covariance do not completely determine finite estimability. It does not establish a large operational advantage in the common regime N≥m.

As τ→∞, both thresholds approach1/(2πR). The matched-risk population-auxiliary law therefore coexists with distinct finite-auxiliary thresholds. This is a model-specific distinction between an infinite-pool summary and a finite-pool calculation.

The example was evaluated deterministically as a mathematical illustration. It is not a registered comparative simulation, a real respondent dataset or a claim to have identified the best predictor. General fitted-predictor, clustered-choice and multinomial extensions remain outside the proposition.
