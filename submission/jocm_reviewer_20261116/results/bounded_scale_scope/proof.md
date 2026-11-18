# How much known scale variation permits the endpoint reversal?

This calculation stays within Appendix G of the existing manuscript. It does not introduce another estimator or claim a new moment inequality. Consider its balanced binary model with independent known positive whole-row scale D, calibrated proxy error e in (0,1/2), and existing population power lambda=tau/(1+tau). The intercept is multiplied by the specified scale. Let rho=(E D)^2/E D^2. The existing result gives the iterated auxiliary-ratio endpoint limits

$$\lim_{\tau\downarrow0}\kappa_D(\lambda_\tau,\tau)=1/2,\qquad \lim_{\tau\to\infty}\kappa_D(\lambda_\tau,\tau)=\frac{\rho}{2\pi e(1-e)}.$$

The proportional sample limit is taken first at every fixed scale law and fixed positive tau. These endpoints concern existence, not prediction risk.

## Classical moment bound

Suppose a <= D <= b almost surely, with a>0, and r=b/a. The classical bounded-variable variance bound, equivalently a reverse Cauchy bound, implies

$$\rho\geq\frac{4ab}{(a+b)^2}=\frac{4r}{(1+r)^2}.$$

A self-contained proof follows from (D-a)(D-b)<=0. Writing mu=E D gives E D^2 <= (a+b)mu-ab. Moreover,

$$\mu^2-\frac{4ab}{(a+b)^2}\{(a+b)\mu-ab\}=\left(\mu-\frac{2ab}{a+b}\right)^2\geq0.$$

Equality requires support on the two endpoints with P(D=b)=a/(a+b). After normalization to E D^2=1, this law has a=1/sqrt(r), b=sqrt(r), endpoint probabilities r/(1+r) and 1/(1+r). Its mean is 2 sqrt(r)/(1+r). This elementary inequality is established prior; see Bhatia and Davis (2000), *A Better Bound on the Variance*, American Mathematical Monthly 107(4), 353–357, DOI 10.1080/00029890.2000.12005203. The publisher metadata was checked; the algebra above supplies the complete proof used here.

## Consequence for the existing boundary

Among all positive scale laws supported in an interval of ratio at most r, a strict decline between the two iterated auxiliary endpoints is possible exactly when

$$\frac{4r}{(1+r)^2}<c_e,\qquad c_e=\pi e(1-e).$$

Equivalently it is possible exactly when

$$r>r_*(e)=\frac{1+\sqrt{1-c_e}}{1-\sqrt{1-c_e}}.$$

Necessity follows from the classical lower bound for rho. Sufficiency uses its equality law: its large-auxiliary boundary is strictly below 1/2. Any fixed dimension ratio strictly between those endpoints then lies below the boundary at a sufficiently small positive tau and above it at a sufficiently large finite tau. The existing proportional-dimensional theorem supplies opposite existence limits. The fixed-dimensional covariance factor 4{1-lambda_tau(1-2e)^2} decreases strictly with tau. Human separation is identical on each coupled pair because positive whole-row scaling does not change margin signs.

This is a sharp characterization of the endpoint mechanism for the stated model. If r<=r_*(e), it excludes this strict endpoint reversal. It does **not** prove that the boundary is monotone, or above 1/2, at every finite tau. The minimizing law for the endpoint ratio need not minimize a finite-tau boundary. Proxy accuracy is 1-e; covariance gains can be small for weak proxies. The result is a subordinate robustness calculation using an established inequality and the manuscript's existing boundary, not an independent innovation claim.

## Verified finite-ratio evaluations

For a positive power write h=(1-lambda)/lambda and

$$J(v)=1+\tau-(1-e)\{f(-ev)+f((h+e)v)\}-e\{f(-(1-e)v)+f((h+1-e)v)\}-\tau\{f((1-e)v/\tau)+f(ev/\tau)\},$$

where f(x)=E(H-x)_+^2 and H is standard normal. Then kappa_D=sup_{v>=0} E J(Dv). Using f''(x)=2 Phi(-x), the negative arguments alone give J''(v)<=-e(1-e), for v>=0. Normalization E D^2=1 yields the same strong concavity constant for the mixture. Its derivative at zero is positive and the derivative becomes negative, so the optimizer is unique. A feasible value at v and the upper bound

$$\sup_{u\geq0} E J(Du)\leq E J(Dv)+\frac{\{E[D J'(Dv)]\}^2}{2e(1-e)}$$

give global enclosures. Arb evaluations include all normal tails. The independent check reconstructs the original squared-distance formula by adaptive integration, without importing the closed-form profile implementation. Its integration diagnostic is numerical, separately identified from the Arb enclosure.
