# Technical appendices

## Appendix A Proof of the finite ratio boundary

### A1 Oriented intervals and the finite sample event

The gradient condition for (2) is $X^T(wp)-b_\lambda=0$. On a labelled row, multiply its residual numerator by $s_L$. For $Y=0$ the oriented residual ranges from $\lambda g$ to $1-\lambda+\lambda g$; for $Y=1$ it ranges from $\lambda(1-g)$ to $1-\lambda+\lambda(1-g)$. These are exactly $I_L$. Orienting an auxiliary residual by $s_U$ gives $I_U$, using the finite ratio $\tau_m=N/m$.

Let $D$ be the product of these observed intervals and let $K=\operatorname{cone}(D)$. Every labelled lower endpoint is strictly positive. The cone is closed, pointed and polyhedral. Define $V$ with columns $s$ and $sS$, stacked over the two samples, and

$$C=K\cap\ker(V^T). \qquad (A1)$$

Conditional on the low dimensional observations, the oriented remaining attributes form a $q$ by $(m+N)$ independent standard Gaussian matrix $A$. The corrected moment is feasible exactly when $\ker(A)$ meets $C$ at a nonzero point. A nonzero point of $K$ has positive scale and can be divided by that scale to obtain a point of $D$.

Interior feasibility must accompany the rank condition in Proposition 1. Section A3 shows, with probability tending to one, that $\ker(V^T)$ meets the relative interior of $D$. Conditional on this event, generic Gaussian position makes a nontrivial nullspace intersection meet the relative interior of $C$ almost surely. Boundary only tangency occurs on a null set of orientations for a fixed polyhedral cone. Relative interior is preserved under the corresponding linear images and preimages, connecting the cone event to an interior corrected moment. Below the strict critical ratio the positively weighted design has full column rank almost surely. At unit power the cone span has dimension $N+1$ before the two constraints, implying $\kappa_c\leq\tau$; the strict lower side has enough auxiliary rows for full rank. At smaller positive power, all $m+N$ rows carry curvature.

### A2 Statistical dimension identity

For independent $H\sim N_{m+N}(0,I)$, polyhedral polar calculus and the projection identity give

$$\delta(C)=E_H\inf_\eta\|\Pi_K(H-V\eta)\|^2. \qquad (A2)$$

Indeed $C^\circ=K^\circ+\operatorname{range}(V)$; the sum is closed for these polyhedral cones. For any vector $w$,

$$\|\Pi_K(w)\|^2=\sup_{t\geq0}\sum_i\{w_i^2-\operatorname{dist}(w_i,tI_i)^2\}. \qquad (A3)$$

Each summand equals $\sup_{v\in tI_i}(2w_iv-v^2)$. It is convex in $w_i$ and concave in $t$. Its normal expectation at $w_i=H_i-h$ is $B(h,t,I_i)$ from (8). We preserve the infimum followed by the supremum in (A2) and (A3). Passing the random extrema to population expectations requires the following uniform controls.

### A3 Tightness of the intercept and signal multiplier

For a unit vector $u=(u_0,u_1)$ put $h_u(S)=u_0+u_1S$. Let $\rho_m(u)$ be the sum of the labelled and auxiliary interval support functions at $-s h_u$, divided by $m$. Bounded interval endpoints and an envelope proportional to $1+|S|$ give a uniform law of large numbers on the unit circle. The common prediction contributions cancel between the samples, giving

$$\rho(u)=E\{(-h_u(S))_++Yh_u(S)\}. \qquad (A4)$$

For finite $\gamma$, the conditional choice probability lies strictly inside the unit interval. Every nonzero affine $h_u$ is nonzero with positive probability, so (A4) is positive in every unit direction. Continuity on the unit circle yields $c=\min\rho(u)>0$. With probability tending to one, $\inf_u\rho_m(u)\geq c/2$. Positive support in both directions also places zero in the interior of $V^TD$, establishing the relative interior event used in A1.

In (A3), set $t=1$ and choose each interval endpoint maximizing $(-V_i\eta)v_i$. Normal linear terms and bounded squared endpoints then give, uniformly in $\eta$,

$$\sup_t\Psi_m(\eta,t)\geq c\|\eta\|-O_P(1). \qquad (A5)$$

Here $\Psi_m$ is the empirical sum in (A3) divided by $m$. At zero multiplier the value is at most $\|H\|^2/m=O_P(1)$. Empirical minimizers are bounded in probability. The same support argument bounds the population minimizing set. This is the step that controls increasing multiplier values; pointwise convergence alone would not suffice.

### A4 Tightness of the cone scale

On a compact multiplier set write $w_i=H_i-V_i\eta$ and $a_i=\lambda|Y_i-g_i|$ on labelled rows. For $t\geq0$,

$$\operatorname{dist}(w_i,tI_i)^2\geq(ta_i-w_i)_+^2\geq t^2a_i^2/2-w_i^2.$$

The expectation of $a_i^2$ is strictly positive, and its empirical average converges to that value. Uniformly on the compact set the average squared $w_i$ is bounded in probability. Consequently

$$\Psi_m(\eta,t)\leq2m^{-1}\sum_iw_i^2-(t^2/2)m^{-1}\sum_{i\leq m}a_i^2. \qquad (A6)$$

At $t=0$ the objective is zero. Bound (A6) localizes all maximizing scales in probability, with a corresponding population bound. It remains valid at unit power, where the labelled intervals are singletons.

### A5 Limit and probability transition

On compact multiplier and scale sets, continuity and an integrable envelope formed from $H^2$, $S^2$ and bounded endpoints give uniform laws of large numbers for the two independent samples. The change from $\tau_m$ to $\tau$ is uniform there. Bounds (A5) and (A6) localize both extrema, so

$$\inf_\eta\sup_t\Psi_m(\eta,t)\ \to\ \inf_\eta\sup_t\Psi(\eta,t)$$

in probability. The empirical minimum lies between zero and $\|H\|^2/m$, whose second moments are bounded. Uniform integrability upgrades convergence to $L^1$. Taking conditional expectation over $H$ in (A2) yields $\delta(C)/m\to\kappa_c$ in probability.

The approximate conic kinematic theorem of Amelunxen et al. (2014) now applies conditionally to the Gaussian nullspace of $A$. A fixed positive gap between $q/m$ and $\kappa_c$ dominates its order $\sqrt{m+N}$ transition band. The interior and rank qualifications in A1 give the two strict sides of Theorem 1. No equality case follows from this argument.

### A6 Balanced reduction and numerical checks

In the balanced oracle model, sign symmetry makes the saddle objective even in both multiplier coordinates; convexity makes zero optimal. Normal squared distance to each interval gives the three terms in $F$ of (12). At unit power the labelled singleton contribution is $-t^2\epsilon(1-\epsilon)$. Substituting $u=t/\tau$ gives (13). Its large ratio limit follows by expanding $f$ at zero after writing the minimizer on its order $1/\tau$ scale. Since $f(0)=1/2$ and $f'(0)=-2\varphi(0)$, the limit is $2\varphi(0)^2/[\epsilon(1-\epsilon)]$.

Independent numerical formulations check the algebra. The balanced experiment has 100 original moment versus oriented interval LP matches and 80 phase conditions with 500 draws each. The signal experiment has 120 LP matches, 12 direct cone projection comparisons and 80 phase conditions with 300 draws. The imperfect prediction experiment adds 120 LP matches, 12 projection comparisons and 80 phase conditions with 300 draws. All classified phase draws are retained and no case is numerically unresolved. Independent projection errors are below $4.1\times10^{-14}$ after active face refinement; 100 versus 200 node quadrature agrees within $8.1\times10^{-9}$ in the imperfect grid.

One benchmark has an exact classical interpretation. At unit power with $g=1/2$ and a fixed zero intercept, put $Z_G=m^{-1/2}\sum_{i\leq m}(2Y_i-1)X_i$. This is independent standard Gaussian relative to the auxiliary rows. Finite existence is, almost surely, equivalent to

$$ (N/\sqrt{m})Z_G\in\{\sum_{j=1}^N c_jX_{Uj}: -1\leq c_j\leq1\}. \qquad (A7)$$

For $N\geq p$, Kabluchko and Seidel (2020, Theorem 3.3) give this Gaussian zonotope absorption probability, with squared scale $N^2/m$. This symmetric special case is established geometry. A separate 200 case signed cube LP comparison verifies the reduction; the signal, informative prediction and power dependence in the stated corrected model require the full calculation above.

### A7 Scalar affine reallocation

A scalar affine reallocation based only on the conditioned low dimensional observations cannot repair the labelled row infima while preserving the complete objective. Let $c$ be the stacked vector of added linear loss coefficients, measurable with respect to those observations. Objective equality for all fitted coefficients requires $W^Tc=0$. For $q\geq1$, conditional Gaussian independence gives

$$W^Tc\sim N(0,\|c\|^2 I_q). \qquad (A8)$$

A nonzero $c$ has zero probability of satisfying the requirement. Every labelled row at positive power needs a nonzero shift to make its infimum finite. A shift estimated from $W$ changes the conditional independence model. An objective preserving affine repair cannot satisfy the stated low dimensional conditioning.

## Appendix B Local score equivalence and predictor moments

### B1 Oracle prediction path

At $\beta_0$, put $A=X(\mu-Y)$ and $B_0=X(\mu-g_0)$. Their means are zero, and conditioning on $(X,Z)$ gives $E[AB_0^T]=M$. The prediction score for $g_s$ is $sB_0$. The covariance of the corrected score, scaled by $m$, is therefore

$$H_0-2\lambda sM+(1+1/\tau)\lambda^2s^2M.$$

The coefficient of $M$ is minimized at $\lambda s=a_\tau$. Inverting the population Hessian gives (16). This is the existing PPI++ variance criterion specialized to a conditional mean family. Each covariance quadratic form has the same minimizer.

Let $D(f;\beta)=L_N(f;\beta)-L_m(f;\beta)$. The loss is affine in its response, so for every coefficient and every finite sample,

$$Q_{\lambda_s,g_s}(\beta)=Q_{a_\tau,g_0}(\beta)+(\lambda_s-a_\tau)D(\mu;\beta). \qquad (B1)$$

Each individual loss in $D(\mu;\beta)$ has zero gradient at $\beta_0$. This proves the exact gradient identity in Proposition 2. If $H_m(\beta)$ and $H_N(\beta)$ are the empirical mean information matrices on the two samples, the Hessian difference is

$$\nabla^2Q_{\lambda_s,g_s}(\beta)-\nabla^2Q_{a_\tau,g_0}(\beta)=(\lambda_s-a_\tau)\{H_N(\beta)-H_m(\beta)\}. \qquad (B2)$$

PPI and GMM objectives can share first order laws and have different finite sample risks (Chen et al., 2026). Equation (B1) fixes the truth score on the same realization. Theorem 1 determines finite estimability for the resulting curvature choices. The oracle weights also separate this comparison from the finite sample cost of estimating PPI++ weights for mean estimation studied by Mani et al. (2026).

### B2 Strict existence inequalities

For $\gamma=0$, $\epsilon=0.1$ and $\tau=1$, the half-shrunk prediction takes values $0.3$ and $0.7$. Its Brier risk is $0.13$. The unit power boundary reduces to

$$\kappa_{\operatorname{sh}}=1-\min_{u\geq0}J(u),\qquad J(u)=0.13u^2+f(0.7u)+f(0.3u). \qquad (B3)$$

Convexity of $f$ gives $J''(u)\geq0.26$, hence $\inf J\geq J(1)-J'(1)^2/0.52$. Rational interval bounds give $J(1)>0.57$ and $|J'(1)|<0.11$. It follows that $\kappa_{\operatorname{sh}}<0.454$.

For the unshrunk oracle at half power, substituting the rescaled value $v=1$ into (12) gives

$$F_*(1)=0.9\{f(-0.1)+f(1.1)\}+0.1\{f(-0.9)+f(1.9)\}+f(0.9)+f(0.1).$$

The same rational bounds give $F_*(1)<1.28$, so its boundary satisfies $\kappa_{\operatorname{or}}\geq2-F_*(1)>0.72$. The two strict bounds establish

$$\kappa_{\operatorname{sh}}<0.46<0.47<0.5<0.72<\kappa_{\operatorname{or}}. \qquad (B4)$$

The normal density and distribution values are bounded by alternating Taylor series with rational arguments. Machin's identity bounds $\pi$, and integer square root comparisons bound the normalization constant. The reproduction code retains the exact rational endpoints. These bounds prove the displayed scalar inequalities using Theorem 1. Direct minimization gives boundaries $0.435067$ and $0.740125$. Ordinary logit has boundary $0.5$, and (16) is $17/25$ of its covariance in both references.

The multiplier support bound and the positive residual second moment in Appendix A hold uniformly near $\gamma=0$. Dominated convergence on the resulting compact sets gives continuous boundaries in that neighborhood. The strict separation consequently persists for nonzero utility signals near zero. This argument does not specify the neighborhood's size.

### B3 Matched finite sample evaluation

The nonzero signal example fixes $\gamma=1.5$, $\epsilon=0.1$ and $\tau=1$ after the population evaluation. Its sampling plan precedes simulation. Each labelled budget supplies 500 independent draws; the four dimensions and four methods share each draw. The complete run has 1000 model draws and 16000 classifications.

Table B1  Matched finite root frequencies along the local score equivalence path

| Labels | Noise slopes per label | Human | Oracle variance | Shrunk variance | Shrunk fixed half |
|---|---|---|---|---|---|
| 80 | 0.2500 | 0.972 | 0.998 | 0.828 | 0.998 |
| 80 | 0.3625 | 0.534 | 0.952 | 0.276 | 0.928 |
| 80 | 0.4500 | 0.122 | 0.792 | 0.034 | 0.732 |
| 80 | 0.6500 | 0.000 | 0.140 | 0.000 | 0.060 |
| 160 | 0.2500 | 1.000 | 1.000 | 0.952 | 1.000 |
| 160 | 0.3625 | 0.606 | 0.994 | 0.264 | 0.992 |
| 160 | 0.4500 | 0.056 | 0.924 | 0.012 | 0.814 |
| 160 | 0.6500 | 0.000 | 0.084 | 0.000 | 0.020 |

Labels are simulated choice rows. The variance reference powers are $0.5$ for the unshrunk conditional mean and $1$ for the half-shrunk prediction. The final column fixes power $0.5$ on that same half-shrunk prediction. Ratios count noise slopes; the fitted intercept and signal slope are additional parameters.

All 4000 matched truth score comparisons agree within $8\times10^{-16}$. Original moment and independent residual formulations agree in 240 checks. Separate directions verify all 2355 ordinary separation cases. An independent interior point feasibility program verifies the unshrunk interior moments in 24 retained discordant designs, while direct recession calculations verify their half-shrunk exterior moments. No classification or independent check remains unresolved. These checks evaluate availability; they do not rank point error or establish growing dimensional interval validity.

### B4 Predictor moments

Let $Y$ be balanced and independent of Gaussian attributes. Conditional on an ancillary confidence amplitude $C\in[0,c_{max}]$, $c_{max}<1/2$, a binary proxy flips $Y$ with probability $\epsilon(C)$. Set $g=1/2+C(2Z-1)$ and define

$$v=EC^2,\qquad K=E[C\{1/2-\epsilon(C)\}],\qquad R=1/4+v-2K.$$

The prediction mean is one half, its variance is $v$, and its covariance with $Y$ is $K$. Its Brier risk is $R$. The established fixed dimensional PPI covariance, including the intercept, is

$$16\{1/4-2\lambda K+\lambda^2v(1+1/\tau)\}I/m. \qquad (B5)$$

Matching $v$ and $K$ matches this entire covariance function and its optimal scalar power. Unit power in Theorem 1 instead gives

$$\kappa_c=\tau[1-\min_{u\geq0}\{\tau Ru^2+E h_u(C)\}],$$

$$h_u(c)=f((1/2+c)u)+f((1/2-c)u). \qquad (B6)$$

Proposition 4. At fixed $R$ and $v$, a strictly larger distribution of $C^2$ in convex order gives a strictly smaller boundary in (B6), provided the two distributions differ.

For $u>0$, define $H_u(x)=h_u(\sqrt{x})$. The identities $f''(a)=2\Phi(-a)$ and $f'''(a)=-2\varphi(a)$ imply

$$h_u'''(c)=2u^3\{\varphi((1/2-c)u)-\varphi((1/2+c)u)\}>0.$$

Also $h_u'(0)=0$, so

$$H_u''(x)=\{c h_u''(c)-h_u'(c)\}/(4c^3)>0,\qquad c=\sqrt{x}.$$

The numerator is the integral from zero to $c$ of $h_u''(c)-h_u''(s)$. A Taylor expansion gives its positive continuous limit at zero. Strict convex order increases $E h_u(C)$ for every positive $u$. The bracket in (B6) has negative derivative at zero and grows quadratically, so its minimizer is positive and finite. Evaluating the strict comparison at the larger dispersion minimizer proves the boundary inequality.

A calibrated specialization takes $\epsilon(C)=1/2-C$, so $E(Y\mid g)=g$ and $K=v$. Set $c=0.4$, $c_{max}=0.499$ and $p=c^2/c_{max}^2$. Prediction A has constant $C=c$; prediction B has $C=c_{max}$ with probability $p$ and $C=0$ otherwise. Both have $v=K=0.16$ and Brier risk $R=0.09$. At $\tau=0.1$ their unit power boundaries are 0.0616361 and 0.0592898. Proposition 4 gives the strict ordering analytically. Coupling them on the same $X,Y$ holds ordinary human separation identical on every sample. Calibration and the matched full covariance function coexist with opposite corrected attainment limits at any ratio between these boundaries. The calibration identity is established (van der Laan and van der Laan, 2026); the example sharpens its boundary consequence for (2).

For a nonzero signal, let $g_e=E(Y\mid S,Z_e)$ use a binary symmetric signal with error $e$, and put $V_e(S)=\operatorname{Var}(g_e\mid S)$. Prediction A returns $g_e$. Prediction B returns a stronger signal posterior $g_h$, $0<h<e<1/2$, with probability $V_e(S)/V_h(S)$, and returns $\mu(S)$ otherwise. Conditional on $S$ the selector is independent of $Y$. Garbling the stronger signal into the weaker one proves $V_h\geq V_e$. Both predictions satisfy $E(Y\mid S,g)=g$, have conditional variance $V_e(S)$ and have conditional Brier risk $\mu(1-\mu)-V_e(S)$. The identities $E[(\mu-Y)(\mu-g)\mid S]=E[(\mu-g)^2\mid S]=V_e(S)$ match every score covariance entry. With $M=E[XX^TV_e(S)]$, their full regular covariance functions are

$$V_\lambda=H_0^{-1}[H_0-\{2\lambda-\lambda^2(1+1/\tau)\}M]H_0^{-1}/m. \qquad (B6a)$$

Their common variance reference power is $\tau/(1+\tau)$. For $\gamma=1$, $e=0.05$, $h=0.001$ and $\tau=4$, the computed boundaries at power 0.8 are 1.1886304 and 1.2192432, with Brier risk 0.0451020 for both. The signals use the known generating law. Independent adaptive integration and 60 original support programs check the calculation. The retained finite size experiment has 200 paired draws per condition; at 80 labels and noise dimension ratio 1.2 the finite counts are 106 and 107. At ratio 1.35 they are 62 and 51. All 36 exploratory pairs and all 2400 classifications accompany the reproduction package. The result is a calibrated matched information counterexample; these small sample frequencies establish no uniform practical predictor ordering. Classification accuracy and logarithmic prediction loss are separate, unmatched properties.

### B5 An observed reference coefficient

Fix a finite coefficient vector $b$ and write $\mu_b(X)=\sigma(X^Tb)$. Freeze this reference and the original prediction during the comparison. For $a>0$ and $a\leq\lambda\leq1$, put

$$g_\lambda=\mu_b+(a/\lambda)(g-\mu_b). \qquad (B7)$$

Affinity of the loss in its response gives, at every coefficient $\beta$,

$$Q_{\lambda,g_\lambda}(\beta)=Q_{a,g}(\beta)+(\lambda-a)\{L_N(\mu_b;\beta)-L_m(\mu_b;\beta)\}. \qquad (B8)$$

Each loss in braces has zero gradient at $b$. More strongly, its separate labelled and auxiliary score rows at $b$ are

$$X\{\mu_b-Y+a(g-\mu_b)\},\qquad aX(\mu_b-g),$$

independently of $\lambda$. Their empirical covariance middle is identical along the path. Hessians differ by $(\lambda-a)\{H_N(\beta)-H_m(\beta)\}$. A sandwich that uses the actual path-dependent Hessian can vary. The equality concerns the score middle, or a variance reference with a common pooled-information bread.

The prediction score at $b$ is scaled by $a/\lambda$ relative to its base value. In the existing pooled-information rule of Appendix C1, its labelled cross covariance scales by $a/\lambda$ and its pooled variance by $(a/\lambda)^2$. If the original untruncated weight is $0<a<1$, the recomputed weight is exactly $\lambda$. At zero base power the positive path instead uses $g_\lambda=\mu_b$ and has zero prediction-score variance; the weight ratio can be undefined. The samplewise identity holds for fitted references. A sampling or truth-covariance guarantee requires additional assumptions on that reference. Prediction recalibration and its relation to PPI++ are established (van der Laan and van der Laan, 2026); the present operation fixes a realized reference score for the existence comparison.

For an exact finite example, use labelled attributes $(2,4)$, choices $(1,1)$, auxiliary attributes $(1,1)$, base predictions $(0.9,0.9,0.5,0.5)$ and reference $b=0$. The scalar rule gives $a=15/88$: the labelled cross covariance with divisor $m$ is $1/5$, and the pooled prediction-score variance with divisor $m+N-1$ is $44/75$. The symmetrized factors in the numerator and denominator cancel. The common objective gradient at the reference is $-57/44$. Since all attributes are positive, the upper attainable moment is $3-2\lambda$, and the positive-direction recession cost is

$$Q_{\lambda,g_\lambda}^{\infty}(1)=9/44-\lambda. \qquad (B9)$$

The corrected moment is positive throughout the path. The scalar log-partition gradient spans the open interval from zero to that upper moment, giving a unique finite root for $15/88\leq\lambda<9/44$. The moment is on the boundary at $9/44$ and exterior above it. All transformed predictions are strictly inside $(0,1)$. Exact fraction arithmetic and separate numerical support checks retain the base recession gap $3/88$ and unit power cost $-35/44$. This deliberately selected finite example illustrates the same loss geometry without a known true probability function.

Table B2  Finite targets at equal observed reference scores with strictly interior base weights

| Labels | Prediction | Allocations | Base finite | End finite | Base only | End only |
|---|---|---|---|---|---|---|
| 50 | AI aided | 201 | 201 | 201 | 0 | 0 |
| 50 | X only | 176 | 176 | 176 | 0 | 0 |
| 20 | AI aided | 80 | 76 | 79 | 1 | 4 |
| 20 | X only | 65 | 62 | 64 | 1 | 3 |

The base weight is strictly between zero and one in every row. End denotes unit power on the transformed prediction. Base only and end only are paired discordant counts. Denominators condition on this base-weight stratum. The same 500 block permutations supply both label budgets and prediction routines. The zero-weight strata and all-allocation counts are reported below.

The complete calculation retains 10000 path targets, with no unresolved state. Verification reconstructs 870 original support classifications and checks them against an independently formulated residual interior LP. It verifies all 490 exterior directions and all 391 ordinary human weak separators. Row-score and covariance-middle discrepancies are below $2.3\times10^{-16}$ and $2.2\times10^{-18}$; recomputed weights agree within $5.2\times10^{-13}$. The initial scalar covariance interface failure and its corrected direct scalar calculation are retained. Neither this fixed-design audit nor the equal reference scores assert that a learned prediction satisfies the Gaussian phase assumptions.

Zero base powers require separate accounting. There are 420 such AI aided allocations and 435 attributes only allocations. At positive path powers the prediction equals the ridge reference probability, so its reference prediction-score variance vanishes and the recomputed variance ratio can be undefined. These cases give 356 and 371 finite unit power endpoints, respectively; each stratum has 64 exterior endpoints. Including the positive base power strata, either prediction routine returns 435 finite unit power endpoints among all 500 allocations. Ordinary human logit is finite in 109 allocations; the ridge reference is finite in all 500. The comparison establishes a finite-design difference at equal observed reference scores. It does not evaluate coefficient error or interval coverage for the transformed predictions.

### B6 Complete moments on a separated human design

The complete GMM criterion uses the human score $m_1$ and the auxiliary minus labelled prediction score $m_2$. Its covariance construction and global sufficient certificate are established in Chen et al. (2026, Sections 2.1-2.3 and Supplement S3.10). We compare that criterion with (2) on the endpoint of B5.

The labelled attributes are $(2,4)$ with choices $(1,1)$; auxiliary attributes are $(1,1)$. Predictions are $(25/44,25/44)$ on labelled rows and $(1/2,1/2)$ on auxiliary rows. Freeze the reference at zero. Center score covariances with sample size divisors and add the existing covariance ridge $m^{-2}=1/4$. The covariance entries are $\Sigma_{11}=1/2$, $\Sigma_{12}=-3/88$ and $\Sigma_{22}=493/1936$. The matrix is positive definite. Its inverse has entries $W_{11}=1972/977$, $W_{12}=264/977$ and $W_{22}=3872/977$.

For a coefficient $\beta$ the two moments are

$$m_1(\beta)=\sigma(2\beta)+2\sigma(4\beta)-3,$$

$$m_2(\beta)=\sigma(\beta)-\sigma(2\beta)-2\sigma(4\beta)+53/44. \qquad (B10)$$

Write $m_j=m_j(\beta)$. The complete criterion with this frozen weight is

$$K(\beta)=(986m_1^2+264m_1m_2+1936m_2^2)/977. \qquad (B11)$$

Its limits at positive and negative infinite coefficients are $1225/977$ and $10729/977$. Exact fraction arithmetic gives

$$K(\log(3/2))=43618624129/38838705425<1225/977. \qquad (B12)$$

A finite point below both limits gives a nonempty compact sublevel set. Continuity proves a finite global minimum. Uniqueness is unspecified. In the same design the human score is strictly negative at every finite coefficient, and the unit power compensated loss has recession cost $-35/44$. The three objectives have distinct existence conclusions on identical observations and predictions.

The calculation fixes the reference at zero and evaluates the uncalibrated complete criterion. CC-PPI additionally specifies a supervised pilot, a protected covariance update and a final fitting procedure. The oracle covariance equality in Proposition 2 concerns (2); comparative predictive risk is not evaluated in this example.

The scope of the prior global certificate can be checked analytically. Write $A$ for the human Hessian. Its displayed conditions include a lower bound $v^TA(c+tv)v\geq a_r$ for $0\leq t\leq r$ and every unit $v$, with $a_r r>\|m_1(c)\|$. Integrating the human Hessian gives $v^Tm_1(c+rv)>0$. The human loss minimized over the closed ball has an interior minimizer and zero score. Positive curvature gives full labelled rank. Thus those sufficient conditions also guarantee a finite human MLE and cannot hold on this separated design. This implication leaves open other sufficient certificates for complete moments.

An independent direct array implementation reconstructs the covariance, finite witness and both limits. Seven numerical starts report optimizer success with small gradients; four return criteria above the exact finite witness and cannot be global minima. Global attainment follows from (B12) and the two limits. The archive retains every numerical return and separates this diagnostic from the fraction proof.

## Appendix C Estimation conventions and reproducibility

### C1 Existing power tuning and covariance

For power tuning at an initializer, put $\mu_i=\sigma(X_i^T\beta)$, $T_i=X_i(\mu_i-Y_i)$ on labelled rows and $P_i=X_i(\mu_i-g_i)$ on all rows. The implementation estimates the pooled information matrix $J$ and uses the PPI++ trace rule

$$A=\operatorname{tr}(J^{-1}\widehat{\operatorname{Cov}}(T_L,P_L)_{sym}J^{-1}),$$

$$D=2(1+m/N)\operatorname{tr}(J^{-1}\widehat{\operatorname{Var}}(P)J^{-1}),$$

$$\widehat\lambda=\operatorname{clip}_{[0,1]}(A/D). \qquad (C1)$$

Here the symmetric covariance is the sum of the two cross covariance matrices; its displayed factor two agrees with that definition. The source style pooled prediction variance uses its sample covariance convention. Missing initializers and nonpositive or unresolved denominators remain explicit statuses.

For a solved fixed power objective, labelled scores are $X\{(1-\lambda)\mu-Y+\lambda g\}$ and auxiliary scores are $\lambda X(\mu-g)$. Their separate sample covariance estimates divided by $m$ and $N$, respectively, form the sandwich middle matrix. The bread is the Hessian of (2). Penalized methods add $I/m$ to that Hessian and use the same random score middle matrix. They do not receive a penalty bias correction. The one step diagnostic evaluates the corresponding sandwich at its returned point. It is not an exact root method.

For soft pooling, the middle matrix is the labelled score covariance weighted by $m/(m+N)^2$ plus the auxiliary score covariance weighted by $N/(m+N)^2$, and the bread is the pooled Hessian. A closed form intercept calculation independently verifies both covariance implementations within $4.5\times10^{-16}$. An exterior one dimensional example verifies the ridge root against a separate scalar solver within $1.9\times10^{-10}.$

### C2 DY prior comparator and diagnostic covariance

The DY estimator minimizes the labelled mean logistic loss with $y_i^*=\alpha Y_i+(1-\alpha)/2$. Strictly interior pseudo-responses and full labelled rank give a unique finite solution. Define $s_i=X_i\{\sigma(X_i^T\beta)-y_i^*\}$ and let $J$ be the mean labelled logistic Hessian. The diagnostic covariance is

$$\widehat{\operatorname{Var}}(\beta)=J^{-1}\widehat{\operatorname{Cov}}(s)J^{-1}/m. \qquad (C2)$$

The sample score covariance uses denominator $m-1$. A separate intercept calculation verifies this formula and the analytic pseudo-response root. The loss multiplier $1/\alpha$ in the published prior expression cancels from the sandwich. Shrinkage bias is retained when assessing coverage of the human parameter. These raw estimates are distinct from the authors' high dimensional rescaling. The recorded code comparison pins their repository commit; no external R code is executed.

### C3 Reproduction contents

The compact reproduction package includes the manuscript source, publication figure builder, complete aggregate tables for the reported grids, independently written estimator and boundary code, a finite objective audit with a synthetic CSV example, and pinned source metadata. Public numeric inputs are downloaded separately and verified against Git blob hashes. The loader permits only specified NumPy numeric pickle globals; downloaded research scripts are not executed. Original research inputs and participant level questionnaire files are excluded.

The quick check runs analytic roots, boundary and rank cases, alternative LP formulations, finite dimensional covariance identities, and mathematical reductions. Full commands regenerate the public allocations and every simulation grid. Source hashes distinguish the completed original finite pool run from a later one dimensional covariance compatibility fix that leaves the original 11 dimensional calculations unchanged. The package documents that difference and preserves the completed aggregate outputs. Counts in the paper refer to choice rows, allocations or generated repetitions as specified, and never convert simulated responses into human respondents.

### C4 Independent training budget diagnostic

Before simulation we fixed two total label budgets, two dimension ratios, two training fractions, three correction powers, the seed and 100 repetitions per condition. Training, correction and auxiliary Gaussian rows are independent. Only the training and correction rows receive simulated labels. The training predictor and direct human ridge both minimize mean logistic loss plus $\rho\|\beta\|^2/2$ with fixed $\rho=0.1$, including the intercept in the penalty. Human maximum likelihood and corrected fits have no penalty. Every returned root has maximum score residual below $10^{-9}$.

A common independent design of $M=4096$ Gaussian rows supplies integration points $X^\circ_j$. With $\mu_j=\sigma((X^\circ_j)^T\beta_0)$, the excess prediction loss is

$$R(\beta)=M^{-1}\sum_{j=1}^M\{\ell((X^\circ_j)^T\beta,\mu_j)-\ell((X^\circ_j)^T\beta_0,\mu_j)\}. \qquad (C3)$$

The soft response notation in (1) gives the conditional expected Bernoulli loss. This integration requires no additional training labels. Paired differences and their Monte Carlo standard errors use only repetitions in which both methods return verified finite roots; full availability counts accompany them. The 4800 method records, 48 summary cells and 80 paired comparisons reconcile independently. Separate positive margin LPs certify all 397 ordinary logit separation cases. Reconstruction of eight designs and an independent log probability expression verify the recorded losses. Training parameter convergence and interval coverage are outside this experiment.

## Appendix D Powers approaching zero

### D1 Existence on human separating directions

Let $h=H^\infty$ and $a=A^\infty$ for the fixed sample losses in Section 3.6. Define the human weak separation cone $S=\{d:h(d)=0\}$. Assume the stacked design has full column rank. For powers strictly below one this is the curvature rank condition; statements of uniqueness at unit power additionally require full auxiliary rank.

Proposition 5. A finite root exists at some sufficiently small positive power exactly when $a(d)>0$ for every nonzero $d\in S$. When a larger positive power already gives a coercive objective, every smaller positive power gives a coercive objective as well.

Equation (17) proves the nesting statement. On $S$, the identity $Q_\lambda^\infty(d)=\lambda a(d)$ gives necessity. For sufficiency, if $S$ meets the unit sphere, continuity and compactness give a positive lower bound for $a$ on that intersection. It remains positive in a neighborhood. Away from this neighborhood $h$ has a positive minimum, while $a$ is bounded below on the sphere. Every sufficiently small positive power makes $(1-\lambda)h+\lambda a$ positive throughout the sphere. If $S=\{0\}$, $h$ is already positive there. Coercivity and full rank then establish the finite root.

For comparison, proper pseudo-label augmentation $H+cB$ has $h\geq0$ and $B^\infty\geq0$. Its zero recession directions and positively weighted design rank are invariant to every $c>0$. Its finite existence event is weight invariant. Synthetic likelihood regularization, including the existence results of Li and Huang (2026), builds on that positive augmentation structure. The subtracted labelled prediction loss in (2) changes the compensation problem. This comparison is an application of classical separation geometry.

### D2 Proof of the right limit

Substitute $v=\lambda t$ in (12). Its minimized expression becomes $G(v)+R_\lambda(v)$, where

$$R_\lambda(v)=(1-\epsilon)f((\lambda^{-1}-1+\epsilon)v)$$

$$\hphantom{R_\lambda(v)}+\epsilon f((\lambda^{-1}-\epsilon)v). \qquad (D1)$$

For $v>0$ this remainder strictly increases with power. The derivative of the full expression at zero is $-2\varphi(0)/\lambda$, so its minimizer is positive. Comparing the minima at two powers proves strict ordering of their boundaries.

The function $G$ is coercive and has $G(0)=1/2+\tau$ and derivative $2\varphi(0)\{2\epsilon(1-\epsilon)-1\}<0$ at zero. Its minimizer is positive and finite. Nonnegativity of $R_\lambda$ bounds the minimum below by $\min G$. Evaluating at a positive minimizer of $G$, where $R_\lambda$ tends to zero, gives the reverse limiting bound. This proves (19), including its strict gap from one half. The proof concerns the stated iterated limit.

### D3 Coefficient growth on a partially separated sample

If $H$ has no finite stationary point and $Q_\lambda$ has finite roots for all sufficiently small positive powers, the root norm tends to infinity. A bounded subsequence would imply a finite zero of $\nabla H$ by continuity of $(1-\lambda)\nabla H+\lambda\nabla A$.

For the coefficient direction, retain Proposition 5's positive recession condition and full stacked rank. Put $u_i=(2Y_i-1)X_i$. Let $I_+$ contain each labelled row with $u_i^Tv>0$ for some $v\in S$, and let $I_0$ contain the remaining rows. Every vector in $S$ has zero margins on $I_0$. Adding exposing directions supplies one vector with margins at least one on all $I_+$. This is the classical maximal separable face (Albert and Anderson, 1984; Ji and Telgarsky, 2019). It can be checked by maximizing $\sum_i s_i$ subject to $u_i^Tv\geq s_i$ and $0\leq s_i\leq1$. A nonnegative dual annihilator with positive entries on $I_0$ certifies the neutral rows.

Let $M$ consist of vectors satisfying $u_i^Tv\geq1$ on $I_+$ and $u_i^Tv=0$ on $I_0$. Define

$$V^*=\operatorname{argmin}_{v\in M}a(v). \qquad (D2)$$

The set is nonempty and compact. The distance from $\beta_\lambda/\log(1/\lambda)$ to $V^*$ tends to zero. A unique minimizer gives point convergence. The signed cost $a$ can be negative away from $S$. The result adapts established margin arguments to the corrected objective (Rosset et al., 2003, 2004). It conditions on the realized design and frozen predictions; it requires no Gaussian attribute law.

To prove the statement, write $L=\log(1/\lambda)$ and let $H_0$ retain the $I_0$ human loss terms with their original divisor $m$. Its minimum $h_0$ is attained at some $b_0$ on its identified row space. A weak separator of this neutral design with any positive margin, added to a sufficiently large exposing direction on $I_+$, would expose a neutral row and contradict its definition. Thus the neutral rows have overlap. Also $\inf H=h_0$. For each $v\in M$, the candidate $Lv+b_0$ has neutral loss $h_0$ and separable loss $O(\lambda)$. The auxiliary softplus remainder gives $A(Lv+b_0)=La(v)+O(1)$. Its criterion excess above $(1-\lambda)h_0$, divided by $\lambda L$, tends to $a(v)$.

Positivity of $a$ on the unit slice of $S$, continuity and homogeneity give constants $c>0$ and $K<\infty$ with $a(v)\geq c\|v\|-Kh(v)$ everywhere. Near $S$ the cost has a positive lower bound; on the remaining unit sphere $h$ is bounded away from zero. Since $H\geq h$ and $H\geq h_0$, for small powers

$$Q_\lambda(\beta)-(1-\lambda)h_0\geq\lambda c\|\beta\|-\lambda Kh_0. \qquad (D2a)$$

Comparison with $Lv+b_0$ gives $\|\beta_\lambda\|=O(L)$. Every scaled limit lies in $S$: a negative human margin would make its loss grow linearly in $L$. Hence its neutral margins are zero. If any $I_+$ margin is strictly below one, that row's loss divided by $\lambda L$ diverges. The neutral loss exceeds $h_0$ and the scaled auxiliary cost remains bounded, contradicting the recovery bound. Each limit is feasible in (D2). The nonnegative excess human loss supplies the lower bound $a(v)$; recovery at a minimizer supplies the reverse bound. Compactness of $V^*$ completes the set convergence.

Under complete separation $I_0$ is empty, recovering the strict-margin formula. With overlap and full labelled rank, $V^*=\{0\}$. A nonunique limiting set requires set-valued comparison; an arbitrary LP vertex need not be the path limit. For example, labelled row $(1,0)$ with choice one and prediction $0.75$, auxiliary rows $(1,1),(1,-1),(-1,1),(-1,-1)$ with predictions one half, gives $a(v)=\max(|v_1|,|v_2|)/2-v_1/4$. Its limiting set is $\{(1,t):|t|\leq1\}$, while the finite path is $(\log(4/\lambda-1),0)$.

The earlier scalar checks remain: all-one labelled features and choices, with $g_L=0.75$ and $g_U=0.25$, have root $\log(2/\lambda-1)$; equal predictions give a boundary target, and reversed predictions give an exterior target. The existing two-coordinate strict-separation calculation approaches $(1,1)$. A quasi-separated example with opposing neutral rows additionally verifies (D2) and a recession that is negative away from $S$. All exact examples and retained paths accompany the reproduction package.

### D4 All powers with signal and estimated intercept

Use the signs, remaining Gaussian matrix $A$ and multiplier columns $V$ of Appendix A. Divide every positive power interval by $\lambda$. Labelled intervals become $[r_i,r_i+1/\lambda-1]$, with $r_i=|Y_i-g_i|>0$. Auxiliary intervals become $J_{Uj}$, using the observed ratio $\tau_m=N/m$. Define $K_\infty$ by the existence of a scale $v\geq0$ satisfying

$$u_{Li}\geq vr_i,\qquad u_{Uj}\in vJ_{Uj}. \qquad (D3)$$

This is a closed pointed polyhedral cone. At zero scale it includes nonnegative labelled vectors with zero auxiliary coordinates. It is the closure of the increasing union of finite positive power cones. For positive scale, all finite labelled coordinates fit in a sufficiently expanded box; zero scale points follow by approximation. Put $C_\infty=K_\infty\cap\ker(V^T)$. A finite root at any positive power supplies a nonzero point of $\ker(A)\cap C_\infty$. An ordinary logit root supplies such a point on the zero scale face. One constrained cone therefore supplies an upper bound simultaneously for all powers, including zero.

Polyhedral polar calculus gives the identity (A2) with $K_\infty$ and $C_\infty$. For $w=H-V\eta$, its projection norm is

$$\|\Pi_{K_\infty}(w)\|^2=\sup_{v\geq0}[\sum_{i\leq m}\{w_i^2-(vr_i-w_i)_+^2\}$$

$$\hphantom{\|\Pi_{K_\infty}(w)\|^2=}+\sum_{j>m}\{w_j^2-\operatorname{dist}(w_j,vJ_{Uj})^2\}]. \qquad (D4)$$

At zero scale the labelled contribution is $(w_i)_+^2$, and its normal expectation is $f(h_L)$. The full normal expectation divided by $m$ is (20).

Both optimization variables need uniform control. Fix reference power $1/2$. The limiting cone contains every smaller positive power cone, each of which contains the reference cone. Projection norm onto a larger closed convex cone is at least that onto the smaller cone. The reference version of (A5) therefore bounds the empirical limiting and all smaller power objectives below by $c\|\eta\|-O_P(1)$. Their values at zero multiplier are at most $\|H\|^2/m$. This localizes their minimizing multipliers in probability. The population argument places all corresponding minimizing sets in a common compact set.

On a compact multiplier set, the labelled distance obeys $(vr_i-w_i)_+^2\geq v^2r_i^2/2-w_i^2$. Its empirical quadratic coefficient converges to $Er^2>0$. This bounds every maximizing scale in probability, with a corresponding population bound. Strictly interior predictions give positive $Er^2$; no deterministic lower bound on $g$ or $1-g$ is required. Compact uniform laws use the integrable $H^2,S^2$ envelope. The empirical minimum is between zero and $\|H\|^2/m$, with bounded second moments. Uniform integrability and conditional expectation over $H$ yield

$$\delta(C_\infty)/m\to\kappa_\infty. \qquad (D5)$$

The conditional conic kinematic bound now excludes a nonzero intersection when $q/m>\kappa_\infty$. Cone inclusion proves the simultaneous upper side of Theorem 2 without a union over powers.

For the lower side, write $v=\lambda t$ in the finite power objective. Its population expression is

$$\Psi_\lambda(\eta,v)=\Psi_\infty(\eta,v)-E f(h_L+v(r+1/\lambda-1)). \qquad (D6)$$

For every positive $v$, the remainder tends to zero by dominated convergence. Positive scales approximate a zero scale by continuity. Consequently $F_\lambda(\eta)=\sup_v\Psi_\lambda(\eta,v)$ increases pointwise to $F_\infty(\eta)=\sup_v\Psi_\infty(\eta,v)$ as positive power decreases. These functions are finite, convex and continuous in $\eta$. On the common compact set containing the minimizers, monotone convergence is uniform by Dini's theorem. This proves $\kappa_c(\lambda)\to\kappa_\infty$ while retaining the order of the two extrema.

At zero multiplier the labelled contribution is at most $1/2$ and the auxiliary contribution at most $\tau$, giving $\kappa_\infty\leq1/2+\tau<1+\tau$. The stacked design has full column rank below the strict boundary. Choose a fixed $\lambda_0\in(0,1)$ with $\kappa<\kappa_c(\lambda_0)$. If $E_m(\lambda)$ denotes a unique finite root, nesting gives

$$P\{E_m(\lambda_m)\}\geq P\{E_m(\lambda_0)\}-P\{\lambda_m>\lambda_0\}. \qquad (D7)$$

The right side tends to one for strictly positive $\lambda_m\to0$ in probability, including data dependent powers. This proves the lower side without a rate requirement.

It remains to establish the strict ordinary logit gap. Let $M(h)=E(H+h)_+=\varphi(h)+h\Phi(h)>0$ and $h=\eta_0+\eta_1S$. Differentiating (20) at zero scale and cancelling the common prediction population terms gives

$$\partial_v\Psi_\infty(\eta,0)=2E\{YgM(h)+(1-Y)(1-g)M(-h)\}>0. \qquad (D8)$$

Every finite multiplier thus gains strictly over its zero scale value $Ef(h_L)$. The coercive bound above ensures a finite minimizer $\eta_\infty$ of $F_\infty$. Evaluating at that minimizer gives $\kappa_\infty=F_\infty(\eta_\infty)>Ef(h_L(\eta_\infty))\geq\kappa_0$, completing the proof. In the balanced specialization, (20) reduces to $1+\tau-G(v)$ and recovers (19).

### D5 Numerical checks and exact zero selection

The general calculation includes 30 signal, sample ratio and prediction conditions. Two hundred versus 400 node quadrature agrees within $1.7\times10^{-9}$. Independent finite differences check its gradient and Hessian, and the zero scale derivative identity agrees within $1.2\times10^{-16}$. All 120 finite power comparisons lie below the corresponding limiting boundary. All 120 original moment and rescaled interval LPs agree. Twenty-four limiting cone projections, retaining the intercept and signal constraints, match an independent full variable quadratic program within $1.6\times10^{-13}$ in squared norm. Feasibility, orthogonality and an independent dual bound verify the values even when a dual optimizer reports precision loss. Both zero scale cases are retained.

For 120 separate finite designs, the limiting LP has 56 interior moments and 64 exterior moments. Each interior case has a positive power witness checked by the original corrected moment LP. Each exterior case has an independent human weak separator with negative corrected auxiliary recession cost. The identity $Q_\lambda^\infty(d)=\lambda a(d)$ agrees within $1.3\times10^{-13}$ on those directions at three powers. These calculations separately check the two structural possibilities of Proposition 5.

A separate analytic scope check uses the excluded hard predictor $g=Z=1-Y$. Here $r=1$, $s_U=s_L$ and $J_U=[-1/\tau,0]$. The scale derivative is $E\{f'(-h_L-v)-f'(-h_L+v/\tau)\}\leq0$, so the limiting supremum is attained at zero and gives $\kappa_0$. The same cone argument applies in this example because $r=1$ retains scale tightness. The uniform strict gain cannot include this hard endpoint.

The general phase experiment generates 6000 independent Gaussian model draws across 12 conditions, with 40 or 80 labels, auxiliary ratios one or four, utility signal 1.5 and three prediction calibrations. Each draw compares three dimensions and five power events, giving 180 aggregate cells with 500 repetitions and 90000 feasibility calls. The raw table retains 12 unknown solver statuses. Reconstructing their exact draws and checking the original corrected moment LP resolves every case with an explicit certificate. Raw and verified tables are separate; no solver status remains unresolved. One positive power call retains its boundary or numerical proximity classification, separately from interior and exterior calls. The paired nesting checks show no violation.

A selector returning exact zero requires a separate statement. In the balanced fixed zero intercept example with $1/2<\kappa<\kappa_c(0+)$, independently choose zero with probability one half and $1/m$ otherwise. Its finite root probability tends to one half; the strictly positive sequence has probability tending to one. Coefficient error and interval behavior also remain separate from the existence theorem.

The earlier balanced validation is retained: 48 independent constrained projections and 6000 original Gaussian draws replayed at powers $1/\sqrt{m}$ and $1/m$, with 12000 new feasibility calculations. All 12 original limiting aggregates agree, and all 36 new report cells retain 500 draws without unresolved states.

### D6 Coefficient size and new-row probability loss

Write $Q_\lambda=H+\lambda D$, where $H=L_m(Y)$ and $D=L_N(g)-L_m(g)$. The design includes every fitted coefficient. For a unit vector $v$, define $z_i=(2Y_i-1)X_i^Tv$, $s_v=m^{-1}\sum_i(z_i)_+>0$, $t_v=m^{-1}\sum_i(-z_i)_+$ and $r_v^2=m^{-1}\sum_i(z_i)_+^2$. Put $C_L=\|X_L\|_{\operatorname{op}}/\sqrt{m}$, $d_v=m^{-1}\sum_L|X_i^Tv|+N^{-1}\sum_U|X_i^Tv|$ and $e=\|\nabla Q_\lambda(\beta)\|_2$. Predictions lie in $[0,1]$. Every finite coefficient satisfies

$$\|\beta\|_2\geq\frac{s_v}{r_vC_L}[\log\frac{s_v}{2(\lambda d_v+e+t_v)}]_+. \qquad (D9)$$

Here $[x]_+=\max(x,0)$. For a strict separator with minimum human margin $a>0$, the sharper bound is

$$\|\beta\|_2\geq C_L^{-1}[\log\frac{a}{2(\lambda d_v+e)}]_+. \qquad (D10)$$

For (D9), the negative human directional score is the mean of $z_i\sigma(-(2Y_i-1)X_i^T\beta)$. Its negative-margin part has magnitude at most $t_v$. Its positive part is at least $(s_v/2)\exp(-r_vC_L\|\beta\|_2/s_v)$: use $\sigma(-u)\geq e^{-|u|}/2$, Jensen with weights proportional to $(z_i)_+$, and Cauchy-Schwarz. The directional prediction difference has magnitude at most $d_v$. The gradient residual supplies (D9). When every margin is at least $a$, unweighted Jensen gives (D10). These bounds use the actual design, including approximately separating directions and approximate roots.

Proposition 6. Under Theorem 2, suppose $\kappa_0<\kappa<\kappa_\infty$. For any possibly data dependent sequence $0<\lambda_m\leq1$ tending to zero in probability, a finite unique root exists with probability tending to one. For constants $c>0$ and finite $C$, on an event of probability tending to one its norm satisfies

$$\|\widehat\beta_{\lambda_m}\|_2\geq c\log(1/\lambda_m)-C. \qquad (D11)$$

Its excess expected human log loss on an independent new Gaussian row diverges in probability. At zero utility signal its Brier loss tends to $1/2$, compared with truth loss $1/4$, while its classification error equals $1/2$ for every fit.

The root-existence conclusion follows from Theorem 2. The human Gaussian maximum margin has a positive limit above the ordinary separation threshold (Montanari et al., 2025; preprint v3, Theorem 3 and Corollary L.1). To match their isotropic threshold, put $T=(2Y-1)S$. The zero-intercept logistic model makes $2Y-1$ symmetric and independent of $T$. Conditional Jensen minimizes the fitted-intercept threshold at zero, giving the same $\kappa_0$ as the no-intercept isotropic calculation. Reflect $S$ for negative signal. Rotational invariance permits application of the isotropic margin theorem despite the distinguished signal coordinate. Its general continuous nondegenerate link assumptions include the null link $1/2$.

Choose a unit zero-intercept human separator with minimum margin bounded away from zero. Gaussian spectral bounds give deterministic upper bounds for $C_L$ and $C_U=\|X_U\|_{\operatorname{op}}/\sqrt{N}$ with probability tending to one, uniformly over the separator; $d_v\leq C_L+C_U$. At an exact root $e=0$, (D10) gives (D11) uniformly over the selected power. Intersect with the finite-root event from Theorem 2. No rate or independence condition on the selected power is introduced.

For population log loss, write a unit parameter direction as $(b,a,r)$ with $b^2+a^2+r^2=1$ and $r\geq0$. The fresh logit is $u=b+aS+rG$ for independent standard normal $G$. Its human recession cost is $E\{u_+-\sigma(\gamma S)u\}$. Every nonzero affine Gaussian logit has strictly positive cost at finite $\gamma$. An integrable bound proportional to $1+|S|+|G|$ gives continuity. Compactness of this unit half-sphere supplies $c_\gamma>0$, independent of the noise dimension. The inequality $\log(1+e^u)\geq u_+$ gives population loss at least $c_\gamma\|\beta\|_2$. Subtracting the finite truth risk proves divergence. Define the loss arbitrarily when a root is absent; that event has probability tending to zero.

For null signal let the new fitted logit be $b+rG$. Its expected log loss and Brier loss satisfy

$$R(\beta)=E\log\{2\operatorname{cosh}((b+rG)/2)\},$$

$$R(\beta)-\log2\geq\|\beta\|_2/\sqrt{2\pi}-\log2, \qquad (D12)$$

$$B(\beta)=1/2-E\sigma'(b+rG)\to1/2. \qquad (D13)$$

For (D12), $E|b+rG|\geq\sqrt{2/\pi}\sqrt{b^2+r^2}$; differentiating $E|a+G|/\sqrt{1+a^2}$ gives its minimum at zero. For (D13), normalize $(b,r)$ by its diverging norm and take convergent subsequences. A positive limiting slope gives a Gaussian score with no atom at zero; a zero limiting slope leaves intercept plus or minus one. Dominated convergence gives $E\sigma'\to0$. Under the null every deterministic or fitted classification rule has conditional error $1/2$.

The diagnostic in Section 5.5 retains 1200 simulated path records from 200 independent model draws. Table D1 reports conditional numerical-fit losses with each finite denominator explicit. The true probability is one half, so the optimal log loss is $\log2$ and Brier loss is $1/4$. The endpoint Brier values use corrected Gaussian quadrature. Numerical gradient error divided by power is at most $9.51\times10^{-8}$; this verifies stationarity at the retained coefficients without a uniform coefficient-distance claim.

Table D1  New-row probability loss in the null Gaussian diagnostic

|Labels|Power|Finite|Mean log loss|Median log loss|Mean Brier|
|---|---|---|---|---|---|
|80|0.5|39/100|4.2880|2.4345|0.431250|
|80|0.1|67/100|6.2155|3.6955|0.452295|
|80|0.0002|67/100|50.8524|34.4706|0.495591|
|160|0.5|43/100|3.1808|2.6972|0.438039|
|160|0.1|82/100|5.8849|3.6687|0.455246|
|160|0.0002|82/100|44.9153|35.4378|0.495132|

Each label budget has 100 independent draws; the six-power plan also includes 0.02, 0.005 and 0.001. Means and medians condition on the displayed finite fits. On the 39 and 43 draws finite at both 0.5 and 0.0002, mean log-loss increases are 36.5512 and 34.2667, with Monte Carlo standard errors 4.2569 and 2.4899. The same paired comparison from 0.5 to 0.1 gives mean differences -1.2676 and -0.1604. Heavy tails make medians and explicit paired denominators consequential. No monotone finite-power risk law is asserted. All raw coefficients, six resolved support flags, six integral repairs and both integration implementations remain available. The derivation concerns the stated compensated objective; existing maximum-margin theory and generic vanishing-regularization mechanisms retain their attribution.

## Appendix E Prediction indices in the fitted design

### E1 Fixed dimensional index extension

For Proposition 3 replace the multiplier matrix in (A1) by the signed columns $(1,S,T^T)$. Conditional on $(S,T,Y,Z,U)$ the remaining oriented Gaussian matrix remains independent standard normal. Every interval identity and the nonzero cone intersection event in A1 are unchanged. The fixed number of additional columns changes the curvature rank requirement by a fixed amount; the strict proportional sides retain full rank. The unit power cone span bound remains $\kappa_c\leq\tau$.

For a unit multiplier $u$ define $h_u=u_0+u_1S+u_T^TT$. The interval support cancellation in A3 gives $E\{(-h_u)_++Yh_u\}$. Conditional on $(S,T)$, both responses have strictly positive probability, and every nonzero affine $h_u$ is nonzero with positive probability. This support expectation is positive on the compact unit sphere in $\mathbb R^{k+2}$. Its minimum is positive. The integrable envelope becomes $1+|S|+\|T\|$. Thus the empirical and population minimizing multipliers have a common bounded set, and zero is interior to the low dimensional interval image with probability tending to one.

The scale bound in A4 still uses $Er^2>0$. Uniform laws and uniform integrability now use the integrable envelope $H^2+S^2+\|T\|^2$. The statistical dimension limit and conditional kinematic bound prove Theorem 1 with (22). For Theorem 2 use the same closed cone (D3), with the enlarged multiplier matrix. The common multiplier bound, scale bound and monotone compact convergence in D4 remain valid. Identity (D8) holds with $h=h_\eta(S,T)$ and is strictly positive. Finally, conditioning on $(S,Y)$ and applying convexity of $f$ to the zero mean $\eta_T^TT$ shows that the ordinary boundary minimizes at $\eta_T=0$. The strict ordinary gap and all positive power quantifiers follow. At zero utility signal an unused signal coordinate can be left among the independent Gaussian columns; its population multiplier is zero.

### E2 Independently trained linear predictions

Let an independent training sample produce a slope vector $v_m$ and intercept $c_m$. Orient the Gaussian contrasts so their true utility direction is the first unit vector $e_S$. Write

$$a_m=v_m^Te_S,\qquad b_m=\|v_m-a_me_S\|. \qquad (E1)$$

Condition on training. When $b_m>0$, take the second rotation direction to be $(v_m-a_me_S)/b_m$; when it is zero take any orthogonal unit direction. Isotropic audit rows give independent standard normal indices $S,T$ and independent remaining Gaussian columns. The fitted prediction is exactly

$$g_m=\sigma(c_m+a_mS+b_mT). \qquad (E2)$$

Corollary 2. Suppose $(a_m,b_m,c_m)$ converges in probability to finite deterministic $(a,b,c)$. For fixed positive power, the strict existence sides are given by Proposition 3 with $k=1$ and $g=\sigma(c+aS+bT)$. Theorem 2 also holds for strictly positive powers converging to zero in probability, with this limiting predictor and its constrained indices. No rate of convergence of the training parameters is required. The training sample must be independent of all audit rows.

To justify passage from conditional training to a deterministic threshold, restrict the training parameters to a compact set whose probability tends to one. The support lower bound in E1 is independent of these parameters because the prediction terms cancel. Conditional Bernoulli variance gives $E(Y-g_m)^2\geq E\{\sigma(\gamma S)[1-\sigma(\gamma S)]\}>0$, uniformly over the set. It provides a uniform scale bound at fixed positive power and in the closed limiting cone. The logistic map is continuous in $(a,b,c)$ with derivative bounded by a constant times $1+|S|+|T|$. Compact uniform laws and localization therefore give continuous population thresholds and uniform convergence of the empirical extrema. Conditional kinematic bounds hold on these same sets. Parameter convergence then gives the strict probability limits; nesting supplies the data dependent positive power lower side as in (D7). This argument assumes convergence of the three training quantities; it does not prove that condition for an arbitrary learning algorithm.

Sixty independent simulated logistic training fits check the rotation directly. Prediction, objective, rotated score and rotated Hessian identities have maximum errors below $5.6\times10^{-16}$. All 180 original and rotated support statuses agree, including 12 exterior cases. These finite algebra checks do not establish training parameter convergence or cross fitting validity.

### E3 Identical calibrated predictions with distinct boundaries

Let $Y$ be a fair binary response independent of standard normals $T,R$ and the remaining fitted columns $W$. For $0<\epsilon<1/2$ define a proxy by

$$Z=1\{T>\Phi^{-1}(\epsilon)\}\quad\text{if }Y=1,$$

$$Z=1\{T>\Phi^{-1}(1-\epsilon)\}\quad\text{if }Y=0,\qquad g=\epsilon+(1-2\epsilon)Z. \qquad (E3)$$

This is a joint generative model of the proxy and outcome. The predictor observes $Z$. The proxy flips each response with probability $\epsilon$, and $E(Y\mid g)=g$. Compare fitted contrasts $(1,R,W)$ and $(1,T,W)$, sharing the identical response and prediction in each pair. The marginal Gaussian contrast law, complete $(Y,Z,g)$ law, calibration and Brier risk $\epsilon(1-\epsilon)$ agree exactly. Both human population targets are zero. In the first model $T$ is external prediction information; in the second it is a fitted attribute.

At unit power, let $t_*>0$ be the external model's optimal cone scale. The external boundary minimizes its index multiplier at zero. The derivative of the aligned envelope there is

$$2\varphi(\Phi^{-1}(1-\epsilon))[-t_*(1-2\epsilon)$$

$$\hphantom{2\varphi(\Phi^{-1}(1-\epsilon))[}+\tau\{f'(t_*(1-\epsilon)/\tau)-f'(t_*\epsilon/\tau)\}]. \qquad (E4)$$

For $x>0$, $f''(x)=2\Phi(-x)<1$. Integrating between the two positive arguments makes the bracket in (E4) strictly negative. The scale maximum is unique because the labelled second derivative is $-2\epsilon(1-\epsilon)<0$. The envelope derivative is thus (E4), and a positive index multiplier lowers the boundary strictly. Calibration and the complete response–prediction law cannot determine existence without their dependence on fitted attributes. The asymptotic ordering does not impose a samplewise ordering on paired finite designs.

The calibrated grid retains 12 conditions. An independent 18 condition grid uses the uninformative prediction $g=1/2+c\operatorname{sign}(T)$, keeping its marginals identical between models. Its largest computed gap is 0.005971.

### E4 Numerical and finite size checks

Integration on each side of the proxy thresholds uses 80 and 160 Legendre nodes, with normal truncation at 12. Boundary values agree within $5.6\times10^{-15}$; omitted normal mass is below $3.6\times10^{-33}$. Independent scalar formulas, analytic derivatives, finite differences and intercept symmetry agree within $1.7\times10^{-11}$.

For 120 separate paired finite designs, 234 original corrected moment and oriented box LPs agree. Six additional unit power cases have independently verified auxiliary rank deficiency and retain their separate infeasible box states. Sixteen independent constrained cone projections agree in squared norm within $2.2\times10^{-14}$.

The illustrative finite size check uses $\epsilon=0.1$, $\tau=1$ and unit power, selected after the population grid for its largest calibrated gap. Each label budget has 500 paired draws shared across dimensions. All 8000 support calculations resolve, and Table E1 retains every paired result. Both models fit $d$ slopes; their conditional Gaussian remainder counts differ by one.

Table E1  Paired finite solution frequencies with identical calibrated predictions

| Labels | Slopes | External | Aligned | Difference | MCSE |
| --- | --- | --- | --- | --- | --- |
| 80 | 24 | 0.916 | 0.862 | -0.054 | 0.0112 |
| 80 | 31 | 0.550 | 0.500 | -0.050 | 0.0135 |
| 80 | 32 | 0.486 | 0.430 | -0.056 | 0.0130 |
| 80 | 38 | 0.128 | 0.138 | 0.010 | 0.0082 |
| 160 | 49 | 0.974 | 0.918 | -0.056 | 0.0103 |
| 160 | 62 | 0.588 | 0.522 | -0.066 | 0.0154 |
| 160 | 63 | 0.536 | 0.446 | -0.090 | 0.0148 |
| 160 | 76 | 0.066 | 0.060 | -0.006 | 0.0066 |

The difference is aligned minus external. Monte Carlo standard errors use repetition level paired indicators; rows share draws. The last 80 label row has the reverse ordering. These probabilities do not establish general prediction superiority or interval validity.

## Appendix F Whole menu boundary and hard predictions

### F0 Model and boundary

Fix $J$ and work in the contrast space $E=\{u\in\mathbb R^J:\sum_j u_j=0\}$. Center independent standard normal signal profiles across alternatives to obtain $S\in E$. True choice probabilities are

$$P(Y=j\mid S)=e^{\gamma S_j}/\sum_k e^{\gamma S_k},\qquad |\gamma|<\infty. \qquad (F1)$$

Estimate $J-1$ alternative constants, one common signal slope and $q$ common noise slopes. Each remaining profile column is independent standard Gaussian across menus and alternatives, independent of the fixed dimensional tuple supplying $(S,Y,g)$. The two samples are independent draws from this joint law. The frozen prediction satisfies $g\in\Delta_J$, where $\Delta_J$ is the probability simplex. Assume $a_*=E\|g-e_Y\|^2>0$, $N/m\to\tau>0$ and $q/m\to\kappa$. Prediction vectors can lie on simplex faces, including hard category predictions.

The oriented binary intervals become zero sum residual sets

$$D_L=(1-\lambda)\Delta_J-e_Y+\lambda g,$$

$$D_U=(\lambda/\tau)(\Delta_J-g). \qquad (F2)$$

For an independent standard Gaussian $H$ in $E$, let $B_J(w,D)=\|w\|^2-\operatorname{dist}^2(w,D)$. A multiplier $\eta=(a,b)\in E\times\mathbb R$ gives $h_\eta=a+bS$. Define

$$\Psi_J(\eta,t)=E B_J(H-h_\eta,tD_L)+\tau E B_J(H-h_\eta,tD_U). \qquad (F3)$$

Theorem 3. Under these assumptions, for every fixed $0<\lambda\leq1$ the boundary is

$$\kappa_{J,c}(\lambda)=\inf_{\eta\in E\times\mathbb R}\ \sup_{t\geq0}\Psi_J(\eta,t). \qquad (F4)$$

Below this ratio, the probability of a unique finite minimizer of (2) tends to one; above it, the probability of a finite minimizer tends to zero. The dimension ratio counts common noise slopes per labelled menu. The fitted constants and signal slope are additional fixed dimensional parameters. The equality case is unspecified.

The positive power limit retains the chosen vertex tangent cone $C_Y=\operatorname{cone}\{e_j-e_Y:j≠Y\}$. At a common scale $v\geq0$, its residual sets are

$$D_{L,\infty}(v)=v(g-e_Y)+C_Y,$$

$$D_{U,\infty}(v)=(v/\tau)(\Delta_J-g). \qquad (F5)$$

Let $\Psi_{J,\infty}(\eta,v)$ replace the scaled sets in (F3) by (F5), and put

$$\kappa_{J,\infty}=\inf_\eta\ \sup_{v\geq0}\Psi_{J,\infty}(\eta,v). \qquad (F6)$$

The ordinary boundary is $\kappa_{J,0}=\inf_\eta E\|\Pi_{C_Y}(H-h_\eta)\|^2$.

Corollary 3. Under Theorem 3, $\kappa_{J,c}(\lambda)$ increases to $\kappa_{J,\infty}$ as positive power decreases to zero. This limit bounds the existence region for every power in $[0,1]$ simultaneously. Every sequence of possibly data dependent powers with $0<\lambda_m\leq1$ almost surely and $\lambda_m\to0$ in probability has this same strict two sided boundary, without a rate requirement. For $J\geq3$, $\kappa_{J,\infty}>\kappa_{J,0}$. For $J=2$, this strict inequality holds exactly when $P(g_Y>0)>0$.

### F1 Finite residual cone and probability boundary

Choose an orthonormal basis for $E$, whose dimension is $r=J-1$. In these coordinates each remaining profile column is standard Gaussian on $E$. Stack the $n=m+N$ menus. The remaining design is an $nr$ by $q$ independent standard Gaussian matrix, independent of the fixed dimensional observations on which we condition. Let $V$ contain the $r$ fitted constant columns and the common signal column. The finite moment equation is a Gaussian nullspace intersection with

$$K=\operatorname{cone}(D),\qquad D=\prod_iD_i,\qquad C=K\cap\ker(V^T). \qquad (F7)$$

Each menu retains one probability simplex. The moment support is $\{\sum_iw_iF_i^Tp_i:p_i\in\Delta_J\}$. Under full rank of the positively weighted attribute contrasts, its interior is the finite root condition, as in Proposition 1. This is ordinary convex support geometry. In particular, a feasible boundary moment and an exterior corrected moment describe distinct events.

Assumption $a_*>0$ implies $P(g_Y<1)>0$. With probability tending to one, at least one labelled menu has a strictly negative chosen residual coordinate before common scaling. That coordinate bounds the scale of any bounded residual sequence, making $K$ closed and pointed. This argument allows other labelled menus to have exactly correct hard predictions. The exceptional sample with no such anchor has probability tending to zero and is not silently assumed impossible.

For independent standard Gaussian menu normals $H_i$ in $E$, polyhedral polar calculus gives

$$\delta(C)=E_H\inf_\eta\sup_{t\geq0}\sum_i B_J(H_i-V_i\eta,tD_i). \qquad (F8)$$

Indeed $C^\circ=K^\circ+\operatorname{range}(V)$ is closed, and projection onto $K$ maximizes $2w^Tu-\|u\|^2$ over $u\in K$. The order of infimum and supremum is preserved.

The support functions of labelled and auxiliary residuals cancel their prediction terms in expectation. For a unit nuisance direction they sum, per labelled menu, to

$$\rho(\eta)=E[\max_j(-h_{\eta,j})+h_{\eta,Y}]. \qquad (F9)$$

Finite signal gives every true category positive conditional probability. Every nonzero multiplier produces a nonconstant contrast with positive probability. Continuity and compactness of the multiplier unit sphere give a strictly positive minimum of (F9). The support envelope is integrable, bounded by a constant times $1+\|S\|$. Its uniform law places zero inside the low dimensional residual image and gives the relative interior qualification used to link a nonzero generic cone intersection to an interior moment. Evaluating a support maximizing residual at fixed scale bounds minimizing multipliers, exactly as in Appendix A3.

For a labelled menu, the minimum residual norm is $\lambda\|g-e_Y\|$. To check this, write the unchosen components as $u_j\geq\lambda g_j$ and the chosen component as minus their sum. Increasing any nonnegative unchosen component increases the squared norm. The minimum occurs at the translated chosen vertex. Thus, on a compact multiplier set,

$$\operatorname{dist}^2(w,tD_L)\geq t^2\lambda^2\|g-e_Y\|^2/2-\|w\|^2. \qquad (F10)$$

The positive mean $a_*$ localizes maximizing scales. Uniform laws on the resulting compact sets use the squared Gaussian and signal envelopes. Empirical extrema converge to (F4); uniform integrability follows from the upper bound $\sum_i\|H_i\|^2/m$. Conditional expectation over the Gaussian normals therefore gives $\delta(C)/m\to\kappa_{J,c}(\lambda)$. Generic Gaussian position and the relative interior event supply the finite moment implication. At unit power the cone span has dimension at most $Nr+1$ before the fixed constraints, so $\kappa_{J,c}(1)\leq\tau r$. The strict lower region has sufficient auxiliary contrast rank. At smaller positive power all menus supply curvature. Conditional conic kinematics now gives both strict sides of Theorem 3. No equality case follows.

### F2 All positive powers and the zero face

Rescale $v=\lambda t$. The labelled residual set is $v(g-e_Y)+v(1-\lambda)(\Delta_J-e_Y)/\lambda$, and the auxiliary set is unchanged at $(v/\tau)(\Delta_J-g)$. As power decreases the labelled sets increase. The closed limiting cone retains all the labelled tangent cones at $v=0$, with zero auxiliary components. This is the ordinary likelihood face.

The bounds can be chosen jointly across powers. After rescaling, every finite positive power cone contains the unit power residual section. Its support lower bound (F9) makes all minimizing multipliers lie in one compact set. The labelled minimum norm in the rescaled sets, including the limit, is $v\|g-e_Y\|$; (F10) therefore localizes $v$ uniformly on that multiplier set.

For each fixed $v>0$, the sets increase to (F5), and their expected projection values converge monotonically. The limiting value is continuous down to $v=0$. Taking suprema over positive scales therefore recovers the retained zero face as well. This argument does not assert pointwise convergence of the finite power integrand at zero scale: its finite power value there is zero, while the limiting labelled projection can be nonzero. The maximized functions of the multiplier are continuous and increase to their continuous limit on the common compact set. Monotone compact convergence passes their minima to (F6).

The limiting constrained cone contains every positive power cone and the ordinary likelihood face. Conic kinematics excludes a nonzero intersection above (F6), giving an upper bound for all powers without a union over selected weights. Below it, choose a fixed positive power whose boundary exceeds the dimension ratio. Identity (17) nests the finite coercive events at smaller powers. For any almost surely positive $\lambda_m\to0$ in probability, the chance of exceeding that fixed power tends to zero. This proves the two strict sides of Corollary 3 without a rate or independence requirement on the selected powers.

### F3 Strict gap and number of alternatives

Fix a finite multiplier and put $w=H-h_\eta$, $p=\Pi_{C_Y}w$ and $z=w-p$. The normal component satisfies $z_Y=\max_jz_j$, with equality at every active unchosen coordinate of $p$. Consequently

$$\max_jw_j-w_Y-(g-e_Y)^Tp=\max_jp_j-g^Tp.$$

Differentiation of the shifted labelled tangent cone and auxiliary simplex at zero gives

$$\tfrac{1}{2}\partial_v\Psi_{J,\infty}(\eta,0+)=E[\max_jp_j-g^Tp]. \qquad (F11)$$

The integrand is nonnegative. For two alternatives, $p=c(e_{1-Y}-e_Y)$ with $c\geq0$, so it equals $2cg_Y$. Conditional Gaussian full support gives $c>0$ with positive probability at every finite multiplier. The derivative is strictly positive exactly when $P(g_Y>0)>0$. If $g=e_{1-Y}$ almost surely, it is zero for every multiplier. Concavity in scale then maximizes at zero, making (F6) equal to the ordinary boundary.

For $J\geq3$, fix $(S,Y,g)$. If $g_Y>0$, every nonzero tangent projection makes the integrand positive. If $g_Y=0$, take an unchosen category $a$ with $g_a>0$ and a different unchosen category $b$. An open Gaussian projection cell has $p=c(e_b-e_Y)$ for $c>0$: the normal has equal chosen and $b$ coordinates, while the other coordinates are strictly smaller. Strict inactive inequalities and a positive ray coefficient define an open cell in the contrast space. On it the integrand is $c(1-g_b)\geq cg_a>0$. Thus the conditional expectation is positive for every fixed prediction vector. Its integrated derivative is positive at every finite multiplier.

The limiting multiplier minimum is attained by the common coercivity bound. At its minimizer, a positive scale gives a value strictly larger than the zero scale value, which is at least the ordinary minimum. This proves the strict gap claimed for larger menus and for the specified binary prediction class. It is a consequence of the compensated objective and its menu constraints. Generic conic projection is an established tool.

If $a_*=0$, then $g=e_Y$ almost surely. The objective instead reduces exactly to $(1-\lambda)L_m(Y)+\lambda L_N(Y)$. For $0<\lambda<1$ all likelihood rows have positive weights, so the existence event is ordinary pooled separation and has boundary $(1+\tau)\kappa_{J,0}$. At unit power only auxiliary rows remain, giving $\tau\kappa_{J,0}$; zero power gives $\kappa_{J,0}$. Every positive sequence strictly below one has the same pooled event, whatever its approach to zero. This treats the zero error endpoint by its exact ordinary likelihood reduction.

### F4 Variance reference and numerical evidence

The numerical hard proxy chooses the true alternative with probability $1-\epsilon$ and each other alternative with probability $\epsilon/(J-1)$. Its fixed prediction is $g=e_Z$. For one orthogonal Gaussian common slope in a fixed dimensional model, let $\mu$ be the true menu probabilities and put

$$R=E(1-\|\mu\|^2),\qquad a=1-\epsilon J/(J-1),$$

$$K=aR,\qquad V_g=2\epsilon+(2a-1)R. \qquad (F12)$$

Exact conditional summation of $(Y,Z)$ gives human score variance $R$, covariance with the surrogate score $K$ and surrogate score variance $V_g$. The usual coordinate sandwich is $[R-2\lambda K+(1+1/\tau)\lambda^2V_g]/R^2$. The existing PPI++ control variate criterion therefore gives

$$\lambda_{\text{var}}=\operatorname{clip}_{[0,1]}\{\tau K/[(1+\tau)V_g]\}. \qquad (F13)$$

This oracle scalar reference uses known simulation quantities. It is a fixed dimensional criterion for the stated coordinate, not an optimum for every utility coefficient or a high dimensional risk guarantee. The posterior oracle value $\tau/(1+\tau)$ generally differs. At error one, $K<0$ and (F13) selects exactly zero; the menu strict gap still concerns the distinct positive power limit. No accuracy benefit from an always wrong predictor is inferred.

The complete population calculation retains 320 evaluations across 80 event conditions, two Sobol resolutions and two scramble seeds. The largest range across resolutions and seeds is 0.001436; the largest difference between the two fine resolution seeds is 0.000567. These are stability diagnostics, without a rigorous integration error bound. Table F1 displays the fine primary values for always wrong hard predictions. Signal follows (F1), so its binary rows have contrast signal $\sqrt2\gamma$.

Table F1  Ordinary and positive power limits for always wrong hard predictions

|Alternatives|Signal|Auxiliary ratio|Ordinary|Positive limit|Difference|
|---|---|---|---|---|---|
|2|0|1|0.499962|0.499962|0.000000|
|2|0|4|0.499962|0.499962|0.000000|
|2|1.5|1|0.334899|0.334899|0.000000|
|2|1.5|4|0.334899|0.334899|0.000000|
|3|0|1|0.833448|0.846956|0.013508|
|3|0|4|0.833448|0.856080|0.022632|
|3|1.5|1|0.559383|0.565067|0.005684|
|3|1.5|4|0.559383|0.568725|0.009343|

For a separate check, the two alternative reduction matches the binary signal convention $\sqrt2\gamma$ in 12 comparisons, with maximum numerical difference below 0.000186. At zero signal and three alternatives, the ordinary tangent cone is a planar wedge of angle $\pi/3$. Its classical statistical dimension is $1/2+1/3=5/6$, providing another independent check. The source records distinguish this existing geometric reduction from the compensated calculation.

The matched finite experiment was selected after the population and variance reference calculations. All three methods and every displayed dimension share each draw. It retains 1000 independent model draws, 18 cells of 500 repetitions, 9000 classifications and 120 independent residual LP crosschecks. All 1090 ordinary boundary moments have separately verified weak separating directions. No classification or independent crosscheck remains unresolved. Table F2 includes every displayed frequency; the complete paired differences and Monte Carlo errors are in the archive.

Table F2  Matched finite fit frequencies with hard predicted categories

|Labelled menus|Slopes per menu|Ordinary|Unit|Variance reference|
|---|---|---|---|---|
|40|0.15|0.996|0.536|0.998|
|40|0.40|0.762|0.086|0.894|
|40|0.65|0.082|0.002|0.520|
|80|0.15|1.000|0.622|1.000|
|80|0.40|0.916|0.040|0.980|
|80|0.65|0.064|0.000|0.538|

Independent finite support and NNLS checks include hard predictions and exact boundary cases. One residual LP initially reported an unknown solver status. A separately formulated nonnegative variable program with an interior point algorithm confirms infeasibility; direct evaluation of its saved direction gives recession cost $-0.002705354$. Original outputs and the interrupted checkpoint are preserved. The completed run retains every planned draw. These checks establish numerical consistency for the displayed calculations, without asserting an independent proof review or priority certificate.

## Appendix G Positive utility scales and auxiliary budgets

### G1 Scaled intervals and the boundary

Observe $X=D(1,S,W^T)^T$, where $S$ is standard normal, $0<d_l\leq D\leq d_u<\infty$ almost surely, and the $q$ coordinates of $W$ are independent standard normal. The bounds and law are fixed as sample size grows. Take $D$ independent of $S$ and $W$, and

$$P(Y=1\mid D,S)=\sigma(\gamma DS),\qquad |\gamma|<\infty.\qquad (G1)$$

The true coefficient is $(0,\gamma,0^T)^T$ and the fitted intercept is multiplied by the specified scale. An additional fixed-dimensional tuple $(Z,U)$ supplies a frozen prediction $g(D,S,Z,U)\in(0,1)$. Conditional on $(D,S,Y,Z,U)$, $W$ remains independent standard normal. Here $Z$ is binary and can be an external proxy. Both labelled and auxiliary samples are independent draws from this law. The scale is known input, not an unknown parameter inferred from these choices.

Use the original compensated mean loss

$$Q_\lambda=L_m(Y)+\lambda\{L_N(g)-L_m(g)\},\qquad 0<\lambda\leq1.\qquad (G2)$$

Let $N/m\to\tau>0$ and $q/m\to\kappa\in(0,\infty)$. Put $s_L=1-2Y$, $s_U=2Z-1$, $r=|Y-g|$, and define the unscaled residual intervals

$$I_L=[\lambda r,1-\lambda+\lambda r],\qquad I_U=(\lambda/\tau)\{s_U(p-g):0\leq p\leq1\}.\qquad (G3)$$

For an interval $I$ let

$$B(h,t,I)=E_H\{(H-h)^2-\operatorname{dist}^2(H-h,tI)\},\qquad H\sim N(0,1).\qquad (G4)$$

For $h_\eta(S)=\eta_0+\eta_1S$, set

$$\kappa_D=\inf_\eta\sup_{t\geq0}\{E B(s_Lh_\eta(S),t,DI_L)+\tau E B(s_Uh_\eta(S),t,DI_U)\}.\qquad (G5)$$

The positive scale assertion in Proposition 7 follows. At every fixed positive power, the probability of a unique finite minimizer tends to one if $\kappa<\kappa_D$. The probability of any finite minimizer tends to zero if $\kappa>\kappa_D$. The equality case is unspecified. Setting $D=1$ recovers the independent Gaussian statement. No joint limit with $d_l$ tending to zero is included.

### G2 Finite design transformation

Use the actual ratio $\tau_m=N/m$ in finite intervals. Orient Gaussian noise row $i$ by $s_i$, and move $D_i$ into the residual variable. Write $\mathcal B_m$ for the product of the intervals $D_i I_i$, $K_m=\operatorname{cone}(\mathcal B_m)$, $V_i=s_i(1,S_i)$, and

$$C_m=K_m\cap\ker(V^T).\qquad (G6)$$

Conditional on the low-dimensional observations, the oriented noise matrix $G$ is a $q$ by $m+N$ independent standard Gaussian matrix, independent of $C_m$. The gradient feasibility equation is equivalent to a nonzero point in $\ker(G)\cap C_m$. Each labelled interval has a strictly positive lower endpoint in a finite sample. Thus $\mathcal B_m$ is compact and excludes zero, $K_m$ is closed and pointed, and every nonzero conic point has a positive common scale. Dividing by it gives interval feasibility. Allowing a different conic scale for each row would change the event.

For uniqueness the rows with positive curvature weight must span all fitted parameters. Below the strict boundary this holds almost surely eventually. At unit power the labelled coordinates of $\mathcal B_m$ are fixed, so $\dim\operatorname{span}(K_m)\leq N+1$ and $\kappa_D\leq\tau$. A strict lower-side ratio consequently has enough independent auxiliary rows for rank. At smaller powers all $m+N$ rows contribute curvature and $\kappa_D\leq1+\tau$.

### G3 Relative interior and qualification

Gaussian absorption and relative-interior calculus are established (Götze et al., 2020, Theorem 3.1 and Lemma 5.1). For clarity, the polyhedral transversality step used here is recorded explicitly. For a fixed pointed polyhedral cone $C$ in a linear span of dimension $r$, a Gaussian kernel independent of $C$ almost surely has the following property: a nonzero intersection with $C$ meets its relative interior. If $q\geq r$, the intersection is zero almost surely. If $q<r$, work inside the span, where the random kernel has dimension $r-q$. For every proper face $F$ of dimension $f$, generic rank gives an intersection with its span of dimension $\max(f-q,0)$. A nonzero intersection with $F$ requires $f>q$. In that case the kernel projects surjectively onto the quotient by the face span. At a relative-interior point of $F$, choose a kernel direction with the same quotient as an interior direction of $C$. A sufficiently small step makes every active face inequality strict while preserving the other strict inequalities. The step lies in the kernel and in the relative interior of $C$. Finitely many faces allow these generic-rank statements simultaneously.

It remains to qualify the nuisance intersection. For unit $u$, let $h_u=u_0+u_1S$. The sum of labelled and auxiliary support functions at $-s h_u$, divided by $m$, converges uniformly on the unit circle. After cancellation of the prediction contributions its limit is

$$\rho_D(u)=E[D\{(-h_u(S))_++Yh_u(S)\}].\qquad (G7)$$

The factor $D$ is necessary. Conditional choice probabilities are strictly between zero and one, and every nonzero affine $h_u$ is nonzero with positive probability. Therefore $\rho_D(u)>0$. Continuity and compactness give $\min_{\|u\|=1}\rho_D(u)>0$. Bounded interval endpoints and the envelope $d_u(1+|S|)$ give the empirical uniform law. With probability tending to one, zero is interior to $V^T\mathcal B_m$, so $\ker(V^T)$ meets the relative interior of $\mathcal B_m$.

Relative interiors commute with the relevant linear image, and the qualified subspace intersection preserves relative interiors. Hence $\operatorname{ri}(C_m)=\operatorname{ri}(K_m)\cap\ker(V^T)$. The Gaussian kernel result then yields a residual witness in the relative interior of the original product intervals. This gives an interior attainable moment and a finite optimum under the curvature-rank condition. A failure of feasibility gives an exterior moment, whose separating direction has negative recession cost. This argument keeps feasibility, interiority and rank as distinct steps.

### G4 Statistical dimension and probability limit

For an independent standard Gaussian $H$ in $\mathbb R^{m+N}$, polyhedral polar calculus gives

$$\delta(C_m)=E_H\inf_\eta\sup_{t\geq0}\sum_i\{(H_i-V_i\eta)^2-\operatorname{dist}^2(H_i-V_i\eta,tD_iI_i)\}.\qquad (G8)$$

The order is expectation, infimum, supremum. Conditional expectation over $H$ cannot be interchanged with the finite-sample infimum without an argument.

The support bound in Appendix G3 localizes empirical minimizing multipliers. At fixed $t=1$, choosing endpoints maximizing $-(V_i\eta)z_i$ bounds the normalized objective below by $c\|\eta\|-O_P(1)$, uniformly in $\eta$. The value at zero multiplier is at most $\|H\|^2/m=O_P(1)$. The same argument bounds the population minimizing set.

On any compact multiplier set, put $w_i=H_i-V_i\eta$ and $a_i=\lambda D_i|Y_i-g_i|$ on labelled rows. Then

$$\operatorname{dist}^2(w_i,tD_iI_i)\geq(ta_i-w_i)_+^2\geq t^2a_i^2/2-w_i^2.\qquad (G9)$$

The empirical average of $a_i^2$ tends to $\lambda^2E[D^2|Y-g|^2]>0$. The normalized saddle objective is consequently bounded above by a tight random constant minus a positive multiple of $t^2$. Its value at $t=0$ is zero. This localizes maximizing scales, including at unit power. The constants may depend on the fixed scale law; no uniform bound as $d_l$ tends to zero is assumed.

On compact multiplier and scale sets, the integrable envelope from $1+H^2+S^2$, with bounded $D$ and interval endpoints, gives a uniform law for both samples. The change from $\tau_m$ to $\tau$ is uniform there. Localization passes the random extrema to the population extrema. The minimum is between zero and $\|H\|^2/m$, with uniformly bounded second moment. Uniform integrability then gives $L^1$ convergence. Conditional expectation over $H$ yields $\delta(C_m)/m\to\kappa_D$ in probability.

The established approximate conic kinematic theorem applies to the independent Gaussian kernel. A fixed nonzero gap between $q/m$ and $\kappa_D$ dominates the order-$\sqrt{m+N}$ transition window. The qualified interior and rank arguments give the stated strict sides. This proves the positive-scale extension under its specified sampling model.

### G5 Balanced oracle profile

Now put $\gamma=0$, let $Z$ flip $Y$ independently with probability $e\in(0,1/2)$, and use $g=e+(1-2e)Z$. Take $D$ independent of $(S,Y,Z)$. On labelled rows, $s_L$ is independent of $r$, $D$ and $S$. On auxiliary rows the oriented interval is always $(\lambda/\tau)[-(1-e),e]$ and $s_U$ is an independent fair sign. For each scale $t$, averaging these signs makes the convex multiplier objective minimize at zero. This establishes the zero multiplier before taking the supremum.

Set $v=\lambda t$, $b=(1-\lambda)/\lambda$, and $f(x)=E(H-x)_+^2$. Then

$$J_{\lambda,\tau}(v)=1+\tau-(1-e)\{f(-ev)+f((b+e)v)\}$$

$$\hphantom{J_{\lambda,\tau}(v)}-e\{f(-(1-e)v)+f((b+1-e)v)\}$$

$$\hphantom{J_{\lambda,\tau}(v)}-\tau\{f((1-e)v/\tau)+f(ev/\tau)\}. \qquad (G10)$$

and the boundary is

$$\kappa_D(\lambda,\tau)=\sup_{v\geq0}E J_{\lambda,\tau}(Dv).\qquad (G11)$$

Write $c=e(1-e)$ and $c_H=\sqrt{2/\pi}$. We have $J(0)=0$, $J'(0)=c_H/\lambda$, and $J''(v)\leq-c$ for $v\geq0$. Indeed $f''(x)=2\Phi(-x)$, and the negative labelled arguments alone contribute at least $E r^2=c$ to $-J''$. Every other contribution is nonnegative. Thus both the unscaled and mixed profiles have unique positive maximizers.

At the mixed maximizer, nonconstant $D$ cannot put $Dv$ at the unique unscaled maximizer almost surely. Therefore every nonconstant independent positive scale law strictly reduces this balanced fixed-power boundary. Strong concavity also gives

$$\kappa_D(\lambda,\tau)\leq\frac{(ED)^2}{\pi\lambda^2c\,ED^2}.\qquad (G12)$$

At unit power the labelled contribution has curvature exactly $-2c$, giving the sharper upper bound $(ED)^2/(2\pi c ED^2)$. These are analytic bounds, independent of numerical optimization.

### G6 Matched information and exact counterexample

For $ED^2=1$, the full fixed-dimensional population Hessian is $I/4$. Human and prediction score covariance blocks are $I/4$, $(1-2e)^2I/4$ and cross covariance $(1-2e)^2I/4$. The existing scalar variance rule selects $\lambda_\tau=\tau/(1+\tau)$, giving

$$mV_{\lambda_\tau}=4\{1-\lambda_\tau(1-2e)^2\}I.\qquad (G13)$$

Positive whole-row scaling leaves ordinary human separation identical in every paired sample. The ordinary null threshold is $1/2$. These are established separation and PPI covariance facts. The compensated profile retains the scale law beyond these moments.

At $e=1/4$, $\tau=1$ and $\lambda=1/2$, constant scale gives $F''\leq47/16$. Hence $\kappa_1\geq J(1/2)\geq\sqrt{2/\pi}-47/128>213/640>1/4$. A second law has probability $1/100$ at $\sqrt{99.999901}$ and otherwise scale $1/1000$. It has $ED^2=1$ and $ED<101/1000$. Since $3<\pi<4$, the mixed boundary is below $10201/140625<1/4$. Thus the two corrected objectives have opposite existence limits at ratio $1/4$, while their full regular covariance is $(7/2m)I$ and their human separation events are exactly equal. The finite simulation uses a separately stated milder scale law.

For a two-point law with probability $p$ at $b_a=\sqrt{[1-(1-p)a^2]/p}$ and otherwise at $a$, changing variable $u=b_av$ gives $pJ(u)+(1-p)J(au/b_a)$. Strong concavity bounds maximizers on a common compact set for fixed $p$. As $a$ decreases to zero the second term tends uniformly to zero there, so the boundary tends to $p\kappa_1$. This permits arbitrarily small boundaries at fixed covariance. The distribution is chosen first and kept fixed in the proportional sample limit.

### G7 Auxiliary ratio limits and fixed power

For each fixed scale law, take the proportional limit at each fixed $\tau$ before the following auxiliary-ratio limits. Along $\lambda_\tau=\tau/(1+\tau)$,

$$\lim_{\tau\downarrow0}\kappa_D(\lambda_\tau,\tau)=1/2,\qquad\lim_{\tau\to\infty}\kappa_D(\lambda_\tau,\tau)=\frac{(ED)^2}{2\pi c ED^2}.\qquad (G14)$$

For the first limit, the negative labelled $f$ terms are at least $1/2$, so $\kappa_D\leq1/2+\tau$. At $v=\tau u$, the profile tends to $1/2-Ef(Du)$. Sending $u$ to infinity after $\tau$ tends to zero gives the matching lower bound, by positivity of $D$ and dominated convergence.

For the second limit, $b=1/\tau$. The identity $f(-x)+f(x)=1+x^2$ and a Taylor expansion of the auxiliary terms at zero give, uniformly on compact $v$ sets,

$$EJ_{\lambda_\tau,\tau}(Dv)\to c_H(ED)v-c(ED^2)v^2.\qquad (G15)$$

Strong concavity bounds maximizing scales uniformly for $\tau\geq1$, because $J'(0)=c_H/\lambda_\tau\leq2c_H$. Maximizing the limiting quadratic proves the second formula. It is also the infinite-auxiliary limit at fixed unit power.

If $(ED)^2/(ED^2)<\pi e(1-e)$, the second limit is below $1/2$. For any dimension ratio between these limits, there are two finite auxiliary ratios, the second larger, with opposite existence limits along the existing variance rule. Meanwhile the complete regular covariance in Appendix G6 decreases strictly as $\tau$ grows. The construction uses an oracle rule and a calibrated predictor; it introduces no estimated-weight noise.

For comparison, at any held positive power the balanced boundary is nondecreasing with the auxiliary ratio. The auxiliary term is the perspective $\tau K(Dv/\tau)$ of the concave function $K(x)=1-f((1-e)x)-f(ex)$ with $K(0)=0$. Concavity makes this perspective nondecreasing in $\tau$, and the labelled terms are unchanged. Taking a supremum preserves the ordering. Thus the additional-data comparison specifically concerns moving along the variance-selected powers. It does not assert a finite-sample monotonicity law for arbitrary new rows or a general ranking of estimators.

