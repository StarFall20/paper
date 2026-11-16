# Vanishing power: finite roots and probability loss

12 November 2026. This supplement studies the binary compensated-logit objective already defined in the manuscript. It proves a finite-design coefficient bound and derives a probability-loss consequence in the manuscript's Gaussian model. Classical margin divergence and Gaussian separability remain attributed. Mathematical derivation and the assessment of priority are separate tasks; the latter remains incomplete.

## A finite-design bound, including approximate roots

Let H be the labelled mean logistic loss and D the auxiliary-minus-labelled frozen-prediction loss, so Q_lambda=H+lambda D. Let X_L and X_U be the actual parameter design matrices, including any intercept. Fix a unit direction v. Put z_i=(2Y_i-1)X_i'v, z_i^+=max(z_i,0), z_i^-=max(-z_i,0), s=mean(z^+)>0, t=mean(z^-), nu=sqrt(mean((z^+)^2)), C_L=||X_L||_op/sqrt(m), and d_v=mean_L|X_i'v|+mean_U|X_i'v|. Predictions lie in[0,1]. For any finite beta with gradient residual e=||grad Q_lambda(beta)||_2,

$$\|\beta\|_2\geq\frac{s}{\nu C_L}\left[\log\frac{s}{2(\lambda d_v+e+t)}\right]_+. \tag{R1}$$

This version remains valid when a computed separator has small negative margins. When v is an exact weak separator, t=0. A strict separator with minimum margin gamma>0 gives the simpler bound

$$\|\beta\|_2\geq C_L^{-1}\left[\log\frac{\gamma}{2(\lambda d_v+e)}\right]_+. \tag{R2}$$

To prove the general bound, write the directional human score as minus the mean of z_i sigma(-(2Y_i-1)X_i'beta). Its negative-margin part has absolute value at most t. The positive part is at least (s/2) times the z^+-weighted mean of exp(-|X_i'beta|). Jensen and Cauchy-Schwarz bound this below by (s/2)exp(-nu C_L||beta||/s). The directional prediction difference has absolute value at most d_v. The stationarity residual then gives the displayed inequality. The strict-margin version uses z_i>=gamma and an unweighted Jensen bound. This argument concerns the existing empirical objective and requires no independent-predictor assumption.

## A joint increasing-dimension consequence

Retain the binary Gaussian model and frozen strictly interior predictor of manuscript Theorem2. Suppose kappa_0<kappa<kappa_infinity. For every possibly data-dependent sequence 0<lambda_m<=1 with lambda_m->0 in probability, Theorem2 gives a finite unique root with probability tending to one. The root's coefficient norm also diverges, and is at least c log(1/lambda_m)-C on an event with probability tending to one, for constants c>0 and finite C depending on the dimension and sample ratios and the human signal.

The human Gaussian design has a strictly positive limiting normalized separation margin above its ordinary threshold. This is existing theory: [Montanari, Ruan, Sohn and Yan, arXiv 1911.01544v3, Theorem 3 and Corollary L.1](https://arxiv.org/html/1911.01544v3). We use the general margin result with its continuous nondegenerate link assumptions. Its isotropic constant-link case has threshold 1/2. With finite logistic signal the threshold is kappa_0. A zero-intercept separating direction suffices, even though the fitted choice model estimates an intercept.

To identify the thresholds, write T=(2Y-1)S. The zero-intercept isotropic threshold is inf_c E(-cT-G)_+^2, for independent standard normal G. Reflect S when gamma is negative. In the zero-intercept human model, (2Y-1) and T are independent and the first is symmetric. Conditional Jensen in the intercept multiplier then minimizes the fitted-intercept expression at zero. This gives the same kappa_0. At gamma=0, T is standard normal, and the minimum is 1/2. Rotational invariance permits a spread signal direction in applying the isotropic corollary; no empirical signal-coordinate distribution assumption is imposed on the fixed distinguished coordinate.

Choose a unit human separator with minimum margin bounded below by a deterministic positive constant on an event of probability tending to one. Gaussian spectral bounds give deterministic upper bounds on C_L and d_v on the same event, uniformly over the selected direction; specifically d_v<=C_L+C_U, with C_U=||X_U||_op/sqrt(N). At an exact root e=0, (R2) proves the logarithmic norm bound uniformly over the selected power. Intersect this event with the uniform finite-existence event in manuscript Theorem 2. No independence or rate requirement is added to lambda_m.

For a fresh Gaussian row, let mu(S)=sigma(gamma S), and represent a unit parameter direction by (a,b,r) with a^2+b^2+r^2=1 and r>=0. Let U=a+bS+rG, where G is independent standard normal. The human population logistic recession cost is E[U_+-mu(S)U]. Every true conditional choice probability is strictly interior, and every nonzero affine Gaussian score is nonzero with positive probability. This cost is strictly positive for every such unit direction. It is continuous by an integrable bound proportional to 1+|S|+|G|. Compactness supplies c_gamma>0 independent of the number of noise slopes.

The inequality log(1+exp(u))>=u_+ gives expected new-row human log loss at least c_gamma||beta||. Subtracting the finite truth risk shows that excess log loss diverges in probability along the fitted vanishing-power roots. Define this loss arbitrarily on the event where a root is absent; that event has probability tending to zero. This is a consequence for the specified compensated estimator, not a new Gaussian maximum-margin theorem or a claim about penalized or complete-GMM alternatives.

## Null-signal probability interpretation

At gamma=0 write a fresh fitted logit as b+rG, G standard normal. The conditional expected log loss is E log(2cosh((b+rG)/2)). The inequality E|b+rG|>=sqrt(2/pi)*sqrt(b^2+r^2) gives excess log loss at least ||beta||/sqrt(2*pi)-log2. As ||beta|| diverges, predicted probabilities saturate: E sigma'(b+rG)->0. Thus Brier loss=1/2-E sigma'(b+rG)->1/2, compared with the truth value1/4. Classification error remains exactly1/2 for every fitted decision rule. The result concerns probability quality; it does not assert a changing null classification accuracy.

For saturation, normalize(b,r) by its diverging norm and take convergent subsequences. If the limiting r component is positive, the normalized Gaussian score has no atom at zero. If it is zero, the limiting intercept is plus or minus one. Dominated convergence yields the result. The absolute-Gaussian inequality follows by differentiating E|a+G|/sqrt(1+a^2); its minimum is at a=0. No parameter-orientation limit is required.

## Planned empirical interpretation

The public-data follow-up fixes predictions within each allocation and evaluates all specified powers on the same released evaluation rows. Its finite-design bound applies directly. Its repeated allocation rows are not independent new participants, and it does not empirically establish the increasing-dimension Gaussian law. The known-truth simulation checks probability loss in the null example. The experiment is a finite grid: neither every-draw monotonicity nor a predictive ranking of all power selection rules follows from the asymptotic consequence.
