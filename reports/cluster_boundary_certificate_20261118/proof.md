# Finite estimation with fixed dependent task blocks

## Model and boundary

Each of m labelled people and N auxiliary people supplies T tasks, with fixed T. People and the two samples are independent. Fix a positive definite unit-diagonal matrix Sigma and a factor L satisfying LL^T=Sigma. Within a person, each fitted Gaussian attribute column is L times an independent standard Gaussian T-vector. Different columns are independent. Fit a common intercept, one unused Gaussian signal coefficient and q remaining coefficients. All true coefficients are zero.

With probability omega the T outcomes are one shared fair binary draw; otherwise they are independent fair binary draws. For every task an independent proxy flip has probability epsilon in (0,1/2), and g=epsilon+(1-2epsilon)Z. Attributes are independent of the outcomes and proxy flips. This gives a frozen predictor on both samples and a correctly specified null marginal logit target. The fitted criterion is the existing compensated average of marginal task losses. It is distinct from a joint random-coefficient panel likelihood.

Let N/m tend to tau>0 and q/m tend to kappa>0. Hold power lambda in (0,1] fixed. Define the task residual boxes

D_L=(1-lambda)[0,1]^T-Y+lambda g,

D_U=(lambda/tau)([0,1]^T-g).

For a standard Gaussian T-vector H independent of the discrete block, put

F(t)=E[||H||^2-dist^2(H,t L^T D_L)]
     +tau E[||H||^2-dist^2(H,t L^T D_U)].

The critical number of noise coefficients per labelled person is sup_{t>=0} F(t). Below this value the probability of a unique finite minimizer tends to one; above it the target is exterior and the probability of a finite minimizer tends to zero. Equality is left unspecified. Sigma and T are fixed throughout this proportional limit.

This is the null, fixed-block specialization of the same conditional-cone argument used in the current manuscript. General conic kinematics and statistical dimension are established results. The initial block formula and numerical screening were already retained on 12 November; this report closes its stated argument and numerical-gap obligations within the present internal review.

## Residual event and conditional Gaussian matrix

Multiplying the task-averaged score equation by mT gives residuals in the displayed boxes, with the observed ratio tau_m=N/m. Let B be blockdiag(L^T). Write all remaining attribute columns as blockdiag(L)G. Conditional on outcomes, predictions and the fitted low-dimensional columns, G has independent standard Gaussian entries. Thus the remaining score equations are G^T u=0 for u in the cone of the transformed residual product B D.

Let V contain the fitted constant and unused signal. The low-dimensional equations become V_tilde^T u=0, with V_tilde=B^{-T}V. The relevant cone is

C=cone(B D) intersect ker(V_tilde^T).

Every labelled residual coordinate has a fixed sign and magnitude at least lambda epsilon. Since L is invertible, any transformed labelled block has norm at least s_min(L)lambda epsilon sqrt(T). A bounded sequence of cone points therefore has bounded scale. The cone is closed, pointed and polyhedral, and each nonzero feasible point has positive scale. Dividing by this scale returns a residual in the original boxes.

## Low-dimensional qualification and localization

For h_t=eta_0+eta_1 S_t, cancellation of the common prediction terms between the two populations gives the support sum

sum_t E[(-h_t)_+ + Y_t h_t] = (1/2)sum_t E|h_t|.

On the unit multiplier circle this is bounded below by T/sqrt(2pi). The bound follows from E|a+bG|>=sqrt(2/pi)sqrt(a^2+b^2). All marginal S_t are standard Gaussian. Independence across people gives a uniform law on that circle. With probability tending to one, zero lies in the interior of the low-dimensional residual image. The same support bound makes the empirical multiplier objective at scale one grow at least linearly in ||eta||, up to a bounded-in-probability term. Its value at zero multiplier is bounded by the normalized Gaussian squared norm. Empirical and population minimizing multipliers are consequently localized to compact sets in probability.

On such a set, the positive labelled anchor supplies a negative quadratic bound in the cone scale. For any transformed labelled residual set E_L and w,

dist^2(w,t E_L)>=t^2 dist^2(0,E_L)/2-||w||^2.

The labelled anchor is uniformly positive in this model, so maximizing scales are also localized. Bounded boxes and Gaussian signal moments give integrable compact-set envelopes. Uniform laws therefore apply to the multiplier and scale extrema. Normalized projection values lie between zero and the normalized Gaussian squared norm, whose second moments are bounded. Uniform integrability allows conditional Gaussian expectation. The change from tau_m to tau is uniform on the localized sets.

## Removing the two population multipliers

For a fixed transformed box E, the function ||w||^2-dist^2(w,tE) is convex in w: it is the supremum over u in tE of 2w^T u-||u||^2. The transformed signal column is an independent standard Gaussian T-vector. Conditional Jensen removes its multiplier at every fixed intercept multiplier and scale.

The joint law is invariant under complementing every Y and g in a block. Together with H becoming -H, this makes the intercept objective even. Convexity places its minimum at zero for every fixed scale. The infimum over both multipliers followed by the scale supremum therefore equals the displayed supremum at zero multiplier. This preserves the original order of extrema.

## Interior, rank and the probability transition

The support qualification ensures the fixed-dimensional kernel intersects the relative interior of the transformed residual product. Conditional Gaussian position then makes a nonzero remaining-kernel intersection meet the cone's relative interior almost surely; boundary-only tangency has probability zero. Relative-interior linear-image calculus connects this event to an interior corrected moment.

At unit power the labelled blocks are a single stacked vector times the common scale, so the cone span has dimension at most NT+1 before the two constraints. The critical ratio is at most tau T. On its strict lower side the NT positively weighted Gaussian auxiliary rows have full fitted-column rank almost surely. At smaller positive power all (m+N)T rows contribute curvature; the corresponding strict side again supplies rank. The existing interior-moment criterion then yields a unique finite minimizer.

The preceding localization argument gives delta(C)/m tending to sup F. Apply the established approximate conic kinematic theorem conditionally to G. A fixed gap in q/m dominates the order sqrt(m) kinematic band because T and tau are fixed. Below the critical ratio the relative-interior and rank arguments give a finite unique root with probability tending to one. Above it there is no nonzero cone intersection, hence no feasible corrected moment. The closed finite support set then has a strict separating recession direction. This proves the two strict sides.

## Matched ordinary boundary and full cluster covariance

For T=3, set the first off-diagonal correlation a to 4/5 or 9/10. Set the remaining correlations to b and 3/2-a-b, where b is the unique ordinary-cone root in the retained bracket. Its ordinary cone dimension is

delta(Sigma)=3/2+[sum asin(corr(Sigma^{-1})_ij)-sum asin(Sigma_ij)]/(2pi)=1.

The root brackets, monotonicity, positive Cholesky pivots and endpoint signs are enclosed with Arb. The matrices have diagonal sum three and total off-diagonal sum three identically. The common omega=3/4 ordinary-human critical ratio is omega*1+(1-omega)*3/2=9/8. The independent-outcome component uses the classical random-label orthant average T/2. These ordinary correlated-pattern results retain their prior attribution.

At epsilon=1/4 and the true parameter zero, the joint human/prediction cluster score covariance for the intercept is

[[5/24,5/96],[5/96,11/384]],

and that for each Gaussian coefficient is

[[7/48,7/192],[7/192,19/768]].

Different Gaussian coordinates have zero cross blocks, and their cross blocks with the intercept vanish. The population Hessian is I/4. Thus the two models match the entire regular fixed-dimensional cluster covariance of the human and prediction scores, and the sandwich reference at every fixed power and auxiliary ratio. This also matches any effective-sample-size summary that is solely a function of those covariance blocks. It does not match the realized samplewise separation event.

At unit power and tau=1, the regular covariance has intercept entry 31/(12m) and Gaussian-coordinate entries 47/(24m) in both models. The exact labelled quadratic coefficient in the boundary is 225/256. The four antipodal auxiliary states have probabilities 25/64,13/64,13/64,13/64.

The deterministic enclosure gives a critical-ratio lower bound above 0.719879 for model A and an upper bound below 0.718838 for model B. At q/m tending to 719/1000, A has a finite compensated estimate with probability tending to one and B has an exterior target with probability tending to one. Both ordinary human MLEs are on the finite side of their common 9/8 threshold. These strict assertions depend on the stated block model and the explicit numerical error certificate.

## Interpretation and remaining scope

The calculation separates a compensation-specific global event after matching a classical dependence capacity and full cluster score information. The raw screening gap is approximately 0.0025 coefficients per labelled person. This is a theoretical distinguishing example with a small magnitude and highly correlated Gaussian task profiles. It does not establish a large practical effect, a new cluster variance estimator, a partially labelled-person method, a mixed-logit theorem or sufficient JCM contribution. The current article's positive-scale construction already supplies the stronger samplewise human-separation control. This block result should enter the manuscript only if its added panel scope justifies its assumptions and length.

Primary established inputs: [Amelunxen et al.](https://arxiv.org/abs/1303.6672), conditional conic kinematics; [Lopez, Schroeder and Opper](https://arxiv.org/abs/cond-mat/9505093), correlated ordinary-pattern capacity; [Broska and Howes](https://arxiv.org/abs/2610.09601), cluster-robust PPI covariance and power tuning. This is internal proof review, with no external expert or global priority certification.
