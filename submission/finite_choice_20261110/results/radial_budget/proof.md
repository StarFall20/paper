# Positive utility scales and auxiliary sample size

12 November 2026. This note completes the mathematical argument for the stated positive-scale model. It separates that argument from priority and publication assessment. The Gaussian nullspace and statistical-dimension tools are established. The specified compensated objective, explicit boundary and its consequences are the subject of this calculation.

## 1 Fixed positive scales

Observe $X=D(1,S,W^T)^T$, where $S$ is standard normal, $0<d_l\leq D\leq d_u<\infty$ almost surely, and the $q$ coordinates of $W$ are independent standard normal. The bounds and law are fixed as sample size grows. Take $D$ independent of $S$ and $W$, and

$$P(Y=1\mid D,S)=\sigma(\gamma DS),\qquad |\gamma|<\infty.$$

The true coefficient is $(0,\gamma,0^T)^T$ and the fitted intercept is multiplied by the specified scale. An additional fixed-dimensional tuple $(Z,U)$ supplies a frozen prediction $g(D,S,Z,U)\in(0,1)$. Conditional on $(D,S,Y,Z,U)$, $W$ remains independent standard normal. Here $Z$ is binary and can be an external proxy. Both labelled and auxiliary samples are independent draws from this law. The scale is known input, not an unknown parameter inferred from these choices.

Use the original compensated mean loss

$$Q_\lambda=L_m(Y)+\lambda\{L_N(g)-L_m(g)\},\qquad 0<\lambda\leq1.$$

Let $N/m\to\tau>0$ and $q/m\to\kappa\in(0,\infty)$. Put $s_L=1-2Y$, $s_U=2Z-1$, $r=|Y-g|$, and define the unscaled residual intervals

$$I_L=[\lambda r,1-\lambda+\lambda r],\qquad I_U=(\lambda/\tau)\{s_U(p-g):0\leq p\leq1\}.$$

For an interval $I$ let

$$B(h,t,I)=E_H\{(H-h)^2-\operatorname{dist}^2(H-h,tI)\},\qquad H\sim N(0,1).$$

For $h_\eta(S)=\eta_0+\eta_1S$, set

$$\kappa_D=\inf_\eta\sup_{t\geq0}\{E B(s_Lh_\eta(S),t,DI_L)+\tau E B(s_Uh_\eta(S),t,DI_U)\}.$$

**Positive-scale extension.** At every fixed positive power, the probability of a unique finite minimizer tends to one if $\kappa<\kappa_D$. The probability of any finite minimizer tends to zero if $\kappa>\kappa_D$. The equality case is unspecified. Setting $D=1$ recovers the independent Gaussian statement. No joint limit with $d_l$ tending to zero is included.

## 2 Exact finite-design transformation

Use the actual ratio $\tau_m=N/m$ in finite intervals. Orient Gaussian noise row $i$ by $s_i$, and move $D_i$ into the residual variable. Write $\mathcal B_m$ for the product of the intervals $D_i I_i$, $K_m=\operatorname{cone}(\mathcal B_m)$, $V_i=s_i(1,S_i)$, and

$$C_m=K_m\cap\ker(V^T).$$

Conditional on the low-dimensional observations, the oriented noise matrix $G$ is a $q$ by $m+N$ independent standard Gaussian matrix, independent of $C_m$. The gradient feasibility equation is equivalent to a nonzero point in $\ker(G)\cap C_m$. Each labelled interval has a strictly positive lower endpoint in a finite sample. Thus $\mathcal B_m$ is compact and excludes zero, $K_m$ is closed and pointed, and every nonzero conic point has a positive common scale. Dividing by it gives interval feasibility. Allowing a different conic scale for each row would change the event.

For uniqueness the rows with positive curvature weight must span all fitted parameters. Below the strict boundary this holds almost surely eventually. At unit power the labelled coordinates of $\mathcal B_m$ are fixed, so $\dim\operatorname{span}(K_m)\leq N+1$ and $\kappa_D\leq\tau$. A strict lower-side ratio consequently has enough independent auxiliary rows for rank. At smaller powers all $m+N$ rows contribute curvature and $\kappa_D\leq1+\tau$.

## 3 Relative interior and Gaussian transversality

For a fixed pointed polyhedral cone $C$ in a linear span of dimension $r$, a Gaussian kernel independent of $C$ almost surely has the following property: a nonzero intersection with $C$ meets its relative interior. If $q\geq r$, the intersection is zero almost surely. If $q<r$, work inside the span, where the random kernel has dimension $r-q$. For every proper face $F$ of dimension $f$, generic rank gives an intersection with its span of dimension $\max(f-q,0)$. A nonzero intersection with $F$ requires $f>q$. In that case the kernel projects surjectively onto the quotient by the face span. At a relative-interior point of $F$, choose a kernel direction with the same quotient as an interior direction of $C$. A sufficiently small step makes every active face inequality strict while preserving the other strict inequalities. The step lies in the kernel and in the relative interior of $C$. Finitely many faces allow these generic-rank statements simultaneously.

It remains to qualify the nuisance intersection. For unit $u$, let $h_u=u_0+u_1S$. The sum of labelled and auxiliary support functions at $-s h_u$, divided by $m$, converges uniformly on the unit circle. After cancellation of the prediction contributions its limit is

$$\rho_D(u)=E[D\{(-h_u(S))_++Yh_u(S)\}].$$

The factor $D$ is necessary. Conditional choice probabilities are strictly between zero and one, and every nonzero affine $h_u$ is nonzero with positive probability. Therefore $\rho_D(u)>0$. Continuity and compactness give $\min_{\|u\|=1}\rho_D(u)>0$. Bounded interval endpoints and the envelope $d_u(1+|S|)$ give the empirical uniform law. With probability tending to one, zero is interior to $V^T\mathcal B_m$, so $\ker(V^T)$ meets the relative interior of $\mathcal B_m$.

Relative interiors commute with the relevant linear image, and the qualified subspace intersection preserves relative interiors. Hence $\operatorname{ri}(C_m)=\operatorname{ri}(K_m)\cap\ker(V^T)$. The Gaussian kernel result then yields a residual witness in the relative interior of the original product intervals. This gives an interior attainable moment and a finite optimum under the curvature-rank condition. A failure of feasibility gives an exterior moment, whose separating direction has negative recession cost. This argument keeps feasibility, interiority and rank as distinct steps.

## 4 Statistical dimension and uniform control

For an independent standard Gaussian $H$ in $\mathbb R^{m+N}$, polyhedral polar calculus gives

$$\delta(C_m)=E_H\inf_\eta\sup_{t\geq0}\sum_i\{(H_i-V_i\eta)^2-\operatorname{dist}^2(H_i-V_i\eta,tD_iI_i)\}.$$

The order is expectation, infimum, supremum. Conditional expectation over $H$ cannot be interchanged with the finite-sample infimum without an argument.

The support bound in Section3 localizes empirical minimizing multipliers. At fixed $t=1$, choosing endpoints maximizing $-(V_i\eta)z_i$ bounds the normalized objective below by $c\|\eta\|-O_P(1)$, uniformly in $\eta$. The value at zero multiplier is at most $\|H\|^2/m=O_P(1)$. The same argument bounds the population minimizing set.

On any compact multiplier set, put $w_i=H_i-V_i\eta$ and $a_i=\lambda D_i|Y_i-g_i|$ on labelled rows. Then

$$\operatorname{dist}^2(w_i,tD_iI_i)\geq(ta_i-w_i)_+^2\geq t^2a_i^2/2-w_i^2.$$

The empirical average of $a_i^2$ tends to $\lambda^2E[D^2|Y-g|^2]>0$. The normalized saddle objective is consequently bounded above by a tight random constant minus a positive multiple of $t^2$. Its value at $t=0$ is zero. This localizes maximizing scales, including at unit power. The constants may depend on the fixed scale law; no uniform bound as $d_l$ tends to zero is assumed.

On compact multiplier and scale sets, the integrable envelope from $1+H^2+S^2$, with bounded $D$ and interval endpoints, gives a uniform law for both samples. The change from $\tau_m$ to $\tau$ is uniform there. Localization passes the random extrema to the population extrema. The minimum is between zero and $\|H\|^2/m$, with uniformly bounded second moment. Uniform integrability then gives $L^1$ convergence. Conditional expectation over $H$ yields $\delta(C_m)/m\to\kappa_D$ in probability.

The established approximate conic kinematic theorem applies to the independent Gaussian kernel. A fixed nonzero gap between $q/m$ and $\kappa_D$ dominates the order-$\sqrt{m+N}$ transition window. The qualified interior and rank arguments give the stated strict sides. This proves the positive-scale extension under its specified sampling model.

## 5 Balanced oracle specialization

Now put $\gamma=0$, let $Z$ flip $Y$ independently with probability $e\in(0,1/2)$, and use $g=e+(1-2e)Z$. Take $D$ independent of $(S,Y,Z)$. On labelled rows, $s_L$ is independent of $r$, $D$ and $S$. On auxiliary rows the oriented interval is always $(\lambda/\tau)[-(1-e),e]$ and $s_U$ is an independent fair sign. For each scale $t$, averaging these signs makes the convex multiplier objective minimize at zero. This establishes the zero multiplier before taking the supremum.

Set $v=\lambda t$, $b=(1-\lambda)/\lambda$, and $f(x)=E(H-x)_+^2$. Then

$$J_{\lambda,\tau}(v)=1+\tau-(1-e)\{f(-ev)+f((b+e)v)\}-e\{f(-(1-e)v)+f((b+1-e)v)\}-\tau\{f((1-e)v/\tau)+f(ev/\tau)\},$$

and the boundary is

$$\kappa_D(\lambda,\tau)=\sup_{v\geq0}E J_{\lambda,\tau}(Dv).$$

Write $c=e(1-e)$ and $c_H=\sqrt{2/\pi}$. We have $J(0)=0$, $J'(0)=c_H/\lambda$, and $J''(v)\leq-c$ for $v\geq0$. Indeed $f''(x)=2\Phi(-x)$, and the negative labelled arguments alone contribute at least $E r^2=c$ to $-J''$. Every other contribution is nonnegative. Thus both the unscaled and mixed profiles have unique positive maximizers.

At the mixed maximizer, nonconstant $D$ cannot put $Dv$ at the unique unscaled maximizer almost surely. Therefore every nonconstant independent positive scale law strictly reduces this balanced fixed-power boundary. Strong concavity also gives

$$\kappa_D(\lambda,\tau)\leq\frac{(ED)^2}{\pi\lambda^2c\,ED^2}.$$

At unit power the labelled contribution has curvature exactly $-2c$, giving the sharper upper bound $(ED)^2/(2\pi c ED^2)$. These are analytic bounds, independent of numerical optimization.

## 6 Exact same-information comparison

For $ED^2=1$, the full fixed-dimensional population Hessian is $I/4$. Human and prediction score covariance blocks are $I/4$, $(1-2e)^2I/4$ and cross covariance $(1-2e)^2I/4$. The existing scalar variance rule selects $\lambda_\tau=\tau/(1+\tau)$, giving

$$mV_{\lambda_\tau}=4\{1-\lambda_\tau(1-2e)^2\}I.$$

Positive whole-row scaling leaves ordinary human separation identical in every paired sample. The ordinary null threshold is $1/2$. These are established separation and PPI covariance facts. The compensated profile retains the scale law beyond these moments.

At $e=1/4$, $\tau=1$ and $\lambda=1/2$, constant scale gives $F''\leq47/16$. Hence $\kappa_1\geq J(1/2)\geq\sqrt{2/\pi}-47/128>213/640>1/4$. A second law has probability $1/100$ at $\sqrt{99.999901}$ and otherwise scale $1/1000$. It has $ED^2=1$ and $ED<101/1000$. Since $3<\pi<4$, the mixed boundary is below $10201/140625<1/4$. Thus the two corrected objectives have opposite existence limits at ratio $1/4$, while their full regular covariance is $(7/2m)I$ and their human separation events are exactly equal. The finite simulation uses a separately stated milder scale law.

For a two-point law with probability $p$ at $b_a=\sqrt{[1-(1-p)a^2]/p}$ and otherwise at $a$, changing variable $u=b_av$ gives $pJ(u)+(1-p)J(au/b_a)$. Strong concavity bounds maximizers on a common compact set for fixed $p$. As $a$ decreases to zero the second term tends uniformly to zero there, so the boundary tends to $p\kappa_1$. This permits arbitrarily small boundaries at fixed covariance. The distribution is chosen first and kept fixed in the proportional sample limit.

## 7 Additional auxiliary observations along the variance rule

For each fixed scale law, take the proportional limit at each fixed $\tau$ before the following auxiliary-ratio limits. Along $\lambda_\tau=\tau/(1+\tau)$,

$$\lim_{\tau\downarrow0}\kappa_D(\lambda_\tau,\tau)=1/2,\qquad\lim_{\tau\to\infty}\kappa_D(\lambda_\tau,\tau)=\frac{(ED)^2}{2\pi c ED^2}.$$

For the first limit, the negative labelled $f$ terms are at least $1/2$, so $\kappa_D\leq1/2+\tau$. At $v=\tau u$, the profile tends to $1/2-Ef(Du)$. Sending $u$ to infinity after $\tau$ tends to zero gives the matching lower bound, by positivity of $D$ and dominated convergence.

For the second limit, $b=1/\tau$. The identity $f(-x)+f(x)=1+x^2$ and a Taylor expansion of the auxiliary terms at zero give, uniformly on compact $v$ sets,

$$EJ_{\lambda_\tau,\tau}(Dv)\to c_H(ED)v-c(ED^2)v^2.$$

Strong concavity bounds maximizing scales uniformly for $\tau\geq1$, because $J'(0)=c_H/\lambda_\tau\leq2c_H$. Maximizing the limiting quadratic proves the second formula. It is also the infinite-auxiliary limit at fixed unit power.

If $(ED)^2/(ED^2)<\pi e(1-e)$, the second limit is below $1/2$. For any dimension ratio between these limits, there are two finite auxiliary ratios, the second larger, with opposite existence limits along the existing variance rule. Meanwhile the complete regular covariance in Section6 decreases strictly as $\tau$ grows. The construction uses an oracle rule and a calibrated predictor; it introduces no estimated-weight noise.

For comparison, at any held positive power the balanced boundary is nondecreasing with the auxiliary ratio. The auxiliary term is the perspective $\tau K(Dv/\tau)$ of the concave function $K(x)=1-f((1-e)x)-f(ex)$ with $K(0)=0$. Concavity makes this perspective nondecreasing in $\tau$, and the labelled terms are unchanged. Taking a supremum preserves the ordering. Thus the additional-data comparison specifically concerns moving along the variance-selected powers. It does not assert a finite-sample monotonicity law for arbitrary new rows or a general ranking of estimators.

## 8 Literature and audit scope

Gaussian conic kinematics and polyhedral interior calculus provide the proof machinery. [Tang and Ye](https://arxiv.org/html/1908.06208) establish ordinary elliptical binary-response separation boundaries. [Hess and Train](https://eml.berkeley.edu/~train/papers/scale.pdf) explain whole-utility scaling and its identification limits. The known scale here multiplies every fitted coefficient, including the intercept.

[Bellec and Koriyama](https://arxiv.org/html/2501.03163v2) give a general Gaussian M-estimator phase transition. Their Assumption2(2) requires an integrable absolute row infimum. The corrected labelled row in the present loss can have infimum minus infinity, which prevents direct use of that theorem. Their conic machinery remains relevant; the assumption difference alone establishes no priority claim.

[Shoham et al.](https://arxiv.org/html/2510.22586v1) use the same correction algebra for online gradient tuning. Their Section3 studies fresh batches and a population stationarity target; Theorem3.5 adds bounded-objective and bounded-gradient assumptions. Here the event concerns attainment of one frozen empirical objective. The present calculation supplies no comparison with their online algorithm. Their author repository is inspected at commit75bf0ef55090393450489d1067b57b59f891e364 without executing or copying its implementation.

[Xu and Sun](https://arxiv.org/pdf/2602.10464) already study harmful prediction aggregation and filtering. Their GLM estimator in Section3.3 minimizes over a compact parameter set; Theorem6 concerns asymptotic covariance. The present example has an oracle predictor and improving regular covariance, while an unrestricted empirical minimum disappears. A broad claim that additional predictions can hurt is already occupied by prior work.

The scalar PPI variance rule and these general tools remain established work. The candidate contribution is the explicit compensated boundary and its same-information collapse and auxiliary-ratio consequences. Specific-theorem priority and sufficient choice-modelling value require assessment beyond the mathematical and numerical checks in this note.
