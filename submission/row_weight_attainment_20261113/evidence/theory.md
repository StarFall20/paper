# Known scales: finite attainment and regular variance respond to different row weights

13 November 2026. This is a derived consequence of the existing compensated objective and its known-scale boundary. Positive weighted likelihood, support geometry, weighted estimating equations, envelope differentiation and Cauchy–Schwarz are established tools. The result supplies a specific comparison within that model; its priority and journal contribution remain under assessment. The reviewed 57-page manuscript is unchanged at this stage.

## 1 Fixed weighted objective and target

Write X_i=D_i R_i with D_i known and strictly positive. The fitted intercept is part of R_i and receives the same multiplier. Let w(D) be fixed, measurable, bounded and bounded away from zero on the fixed compact scale support. Define weighted losses using the original sample counts:

$$L_m^w(Y;\beta)=m^{-1}\sum_{i\in L}w(D_i)\{A(X_i^T\beta)-Y_iX_i^T\beta\},\quad A(u)=\log(1+e^u).$$

Use the same weight function in all three terms of

$$Q_{\lambda,w}=L_m^w(Y)+\lambda\{L_N^w(g)-L_m^w(g)\}. \qquad (W1)$$

The two samples are independent draws from the same joint law; the predictor and weights are frozen. The expected prediction terms cancel. If the conditional human logit model is correct, the weighted population score vanishes at the same true coefficient. Its Hessian is positive definite under the usual weighted full-rank population condition. Under misspecification this is a weighted logistic projection and can differ from the original target.

The definition divides by m and N, not by the two random sums of weights. Multiplying the complete objective by one common positive constant leaves its minimizer unchanged. Separately normalizing the labelled and auxiliary weights changes the finite objective. The numerical implementation uses only one common normalization.

## 2 Exact finite-design support identity

Put T_i=w(D_i)D_i. For every direction b, positive homogeneity gives

$$w(D_i)(X_i^Tb)_+=(T_iR_i^Tb)_+.$$

Consequently the recession function of (W1) equals the recession function of the original unweighted compensated objective on the virtual design T_iR_i, with the SAME choices and predictions. Curvature rank also agrees because both designs differ by positive row multipliers. The weighted attainable moment and its zonotope are exactly the virtual-design moment and zonotope.

Classical convex support geometry now gives identical interior, boundary, exterior and curvature-rank classifications. In particular, w(D)=1/D restores the support event on R on every fixed sample. This holds for arbitrary finite data and positive powers, subject to the same rank distinction as the original audit.

The two losses generally have different finite coefficients: A(D_iR_i^T\beta)/D_i differs from A(R_i^T\beta). This is an event identity, not an equality of likelihoods or estimates. It also preserves no finite-dimensional score identity between the weighted and unweighted fits.

## 3 Balanced model and two distinct optima

Now use the preceding balanced model: R=(1,S,W), independent standard Gaussian S and W, fair binary Y, and a proxy Z that flips Y with probability e in (0,1/2). The calibrated frozen prediction is g=e+(1−2e)Z. D is independent of (R,Y,Z), has a fixed bounded positive law, and E D²=1. Let q/m tend to kappa and N/m tend to tau. The entire fitted intercept is scaled.

Let J=J_(lambda,tau) be the strongly concave scalar profile from Appendix G5. It has J(0)=0 and a unique positive maximizer v_* for every fixed positive lambda. The finite-design identity and the already derived known-scale boundary imply

$$K(w)=\sup_{v\geq0}E J(vw(D)D). \qquad (W2)$$

Both strict probability sides follow from that existing result. Equality at the phase boundary remains unspecified. This step introduces no new Gaussian absorption theorem.

For fixed dimension, put a=1−2e. The population Hessian of (W1) is E[wD²]I/4. Its corrected-score covariance gives

$$mV_{\lambda,w}=4\{1-2\lambda a^2+(1+1/\tau)\lambda^2a^2\}\frac{E[w^2D^2]}{(E[wD^2])^2}I. \qquad (W3)$$

This is a regular, fixed-dimensional oracle covariance. It does not describe proportional-dimensional MSE or interval coverage. The established variance power remains lambda_tau=tau/(1+tau) for every such w.

**Proposition.** In this balanced model with nonconstant D:

1. The largest existence boundary over positive bounded row weights is K(1/D)=max J. Equality holds only when w(D)D is constant almost surely, up to a common multiplier.
2. The smallest covariance multiplier in (W3) is one. Equality holds only when w(D) is constant almost surely, up to a common multiplier.
3. These two optima differ. The inverse-scale weight has covariance multiplier 1/(E D)²=1+CV(D)², while the unweighted loss has multiplier one.

For part 1, E J(vT) cannot exceed max J. The scaled profile is strongly concave, with a unique finite positive maximizing v. Equality requires vT=v_* almost surely. For part 2, Cauchy–Schwarz gives (E[wD²])²≤E[w²D²]E[D²]. Its equality condition requires wD proportional to D. Nonconstant D prevents both equality conditions from holding simultaneously. Part 3 follows by substitution. The inequalities are elementary once the compensated profile is available; generic variance–robustness trade-offs and weighted PPI are not new claims.

## 4 A continuous power-weight path

Take fixed weights w_r(D)=D^(−r), with r in [0,1]. Its effective support scale is D^(1−r). Let M(k)=E D^k and L(k)=log M(k). The covariance multiplier is

$$F(r)=\frac{M(2-2r)}{M(2-r)^2}. \qquad (W4)$$

Differentiation gives

$$\frac{d}{dr}\log F(r)=2\{L'(2-r)-L'(2-2r)\}. \qquad (W5)$$

Convexity of L makes F strictly increasing for r>0 when D is nonconstant. At zero the derivative vanishes. If Q_2 gives D density proportional to D² relative to its original law, then

$$F(r)=1+\operatorname{Var}_{Q_2}(\log D)r^2+o(r^2). \qquad (W6)$$

For attainment, put t=1−r and let v_t uniquely maximize E J(vD^t). Bounded positive scale support and strict concavity justify the implicit-function and envelope steps. The first-order condition is E[D^t J'(v_tD^t)]=0. Therefore

$$K'(r)=-v_t E[D^t\log D\,J'(v_tD^t)]. \qquad (W7)$$

Under the probability measure Q_t with density D^t/M(t), the expectation in (W7) is M(t) times Cov_(Q_t)(log D,J'(v_tD^t)). For t>0 the first argument increases strictly with D and the second decreases strictly. Their covariance is strictly negative for nonconstant D. Thus K'(r)>0 for r<1 and K'(1)=0. Small positive row-weight exponents improve the population existence boundary to first order while costing regular covariance only to second order.

This is a monotone trade-off along a specified path. It does not establish Pareto optimality of every intermediate weight among all measurable weights. It also does not make finite-sample support events nested: all observed reverse transitions must remain reported.

## 5 What the comparison allows an analyst to decide

The same variance-optimal correction power can be used throughout. A specified scale-weight change then moves existence and regular covariance in opposite directions. An analyst using a known-scale model can check the actual weighted objective and quantify this cost instead of treating a recovered finite root as an efficiency gain. Column standardization is an invertible coefficient transformation and leaves the event unchanged; the row-weight change modifies the criterion and has the explicit cost above.

Estimated scales and learned weights require a separate sampling analysis. This calculation conditions on a specified scale law; the balanced choices supply no identification of an unknown error-scale distribution. The result gives an implication of the existing boundary calculation whose empirical value and specific-result priority remain to be assessed.
