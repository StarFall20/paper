# General simultaneous power boundary: derivation before checks

10 November2026. Candidate extension within the existing corrected binary-logit problem. This document records the mathematical argument before its planned numerical checks. Classical conic geometry is retained as prior machinery. The current delivered manuscript remains the verified balanced result until this extension is checked and integrated.

## Objects and proposed result

Keep all assumptions of existing Theorem1. There are q independent Gaussian remaining slopes,one Gaussian utility signal S,an estimated intercept,finite logistic signal gamma,a frozen predictor g(S,Z,U) strictly in(0,1),and N/m->tau>0. Put s_L=1-2Y,s_U=2Z-1,r=|Y-g| and h_L=s_L(eta_0+eta_1 S),h_U=s_U(eta_0+eta_1 S). Let J_U be the existing auxiliary interval divided by power:at Z=1 it is[-g,1-g]/tau,and at Z=0 it is[g-1,g]/tau.

For f(a)=E(H-a)_+^2 and the existing B(h,v,a,b),define

Psi_infty(eta,v)=E[1+h_L^2-f(-h_L-vr)]+tau E B(h_U,v,J_U),v>=0.

Let F_infty(eta)=sup_v>=0 Psi_infty(eta,v),kappa_infty=inf_eta F_infty(eta). Let kappa_ordinary=inf_eta E f(h_L),the existing ordinary-logit Gaussian existence threshold with intercept and signal fitted.

Proposed theorem:kappa_infty>kappa_ordinary. Below kappa_infty,a fixed strictly positive power gives a unique finite root with probability tending to one. Above kappa_infty,the probability that ANY power in[0,1] gives a finite root tends to zero. Any possibly data-dependent powers0<lambda_m<=1 almost surely with lambda_m->0 in probability have this same strict two-sided boundary. The equality case is unspecified. This is an exact-existence statement,without a coefficient error,normal approximation or interval guarantee.

## Closed cone and projection formula

Divide finite positive-power intervals by lambda. Labelled intervals become[r_i,r_i+1/lambda-1],and auxiliary intervals are J_U at the observed tau_m. Let K_infty be the CLOSED polyhedral cone of vectors u for which some v>=0 satisfies u_Li>=v r_i and u_Uj in v J_Uj. At v=0 include nonnegative labelled vectors and zero auxiliary coordinates. This is the closure of the increasing union of finite-positive-power cones. Every observed r_i is strictly positive,so the explicit polyhedral representation is closed and pointed. Let V have columns s and sS,and C_infty=K_infty intersect ker(V^T).

Conditional on all low-index data,the oriented remaining design matrix A is independent standard Gaussian. Finite feasibility at any positive power implies a nonzero intersection of ker(A) and C_infty. An ordinary-logit stationary point also supplies a nonzero vector on its v=0 labelled face. Thus one cone controls the upper side for ALL powers,including zero,without an uncountable union of probability bounds.

Polyhedral polar calculus gives delta(C_infty)=E_H inf_eta ||Pi_K_infty(H-V eta)||^2. The projection norm squared equals the supremum over v>=0 of

sum_label[(H_i-h_Li)^2-(v r_i-H_i+h_Li)_+^2]
+sum_aux[(H_j-h_Uj)^2-dist(H_j-h_Uj,v J_Uj)^2].

Its normal expectation divided by m is Psi_infty. At zero scale its label contribution is f(h_L),not zero.

## Localization and statistical-dimension limit

The rescaled finite cones K_lambda increase as positive lambda decreases. Fix lambda_ref=1/2. For every0<lambda<=lambda_ref and every shifted vector w,projection-norm monotonicity gives

||Pi_K_infty(w)||^2 >= ||Pi_K_lambda(w)||^2 >= ||Pi_K_ref(w)||^2.

The existing AppendixA3 lower bound for the reference cone is c||eta||-O_P(1),uniformly in eta,with c>0. At eta=0 every projection norm squared is at most ||H||^2. This localizes empirical minimizing multipliers in probability for the limiting cone and uniformly for all small finite powers. The analogous population argument puts their minimizing sets in a common compact set. No assumption of uniformly interior g is needed;bounded original reference intervals and finite logistic signal supply the existing lower bound.

On a compact multiplier set,w_i=H_i-V_i eta has a uniformly bounded empirical second moment. Labelled distances satisfy (v r_i-w_i)_+^2>=v^2 r_i^2/2-w_i^2. The empirical mean of r_i^2 tends to E r^2>0. Hence all maximizing scales are bounded in probability,with a corresponding population bound. This uses positive expectation,not a deterministic positive lower bound on g.

On these compact sets,uniform laws hold with an integrable envelope from H^2,S^2 and bounded interval endpoints. The empirical minimum lies between0and||H||^2/m;bounded second moments give uniform integrability. Taking conditional expectation over H yields delta(C_infty)/m->kappa_infty in probability. The classical approximate conic kinematic theorem then gives the simultaneous upper side when q/m>kappa_infty.

## Fixed-power thresholds approach the same bound

After writing v=lambda*t,the finite population objective is

Psi_lambda(eta,v)=Psi_infty(eta,v)-E f(h_L+v(r+1/lambda-1)).

For each positive v,the final term tends to zero by dominated convergence. Continuity in v lets positive scales approximate a zero scale as well. Thus F_lambda(eta)=sup_v Psi_lambda(eta,v) increases pointwise to F_infty(eta). These functions are finite convex continuous functions of eta. The common compact minimizing set from the reference-cone argument and Dini's theorem give uniform convergence there. Consequently kappa_c(lambda)->kappa_infty. The order inf_eta sup_v is preserved throughout.

The bound at eta=0 is at most1/2+tau,strictly below1+tau. Below kappa_infty the stacked Gaussian design therefore has full column rank asymptotically. Choose a fixed lambda_0 in(0,1) with q/m limiting below kappa_c(lambda_0). The existing fixed-power theorem supplies its coercive finite root. Fixed-sample nesting gives

P(E_m(lambda_m))>=P(E_m(lambda_0))-P(lambda_m>lambda_0).

This tends to one for strictly positive lambda_m->0 in probability,without independence from the data or a rate condition. This completes the lower side.

## Strict gain over ordinary existence

Write M(x)=E(H+x)_+=phi(x)+x Phi(x) and h=eta_0+eta_1 S. Direct differentiation and cancellation between the two identical prediction populations give

d_v Psi_infty(eta,0)=2E[Y g M(h)+(1-Y)(1-g)M(-h)]>0.

This identity is central and requires independent numerical verification. All factors are nonnegative;M is strictly positive at every finite argument and g is strictly interior. The expression is strictly positive for every finite eta. At zero scale Psi_infty(eta,0)=E f(h_L). At a finite minimizer eta_infty of F_infty,its maximizing scale improves that value strictly. Therefore

kappa_infty=F_infty(eta_infty)>Psi_infty(eta_infty,0)>=kappa_ordinary.

The gain is an existence/stabilization property,including imperfect or uninformative frozen predictions. Prediction quality still changes the size of the region and the estimation cost. This is not a gain in choice information or a uniformly better estimator. The assumption of a single fixed strictly interior predictor must not be silently extended to a changing sequence of learned or hard predictors.

## Verification and prior boundaries

The plan checks signal,intercept asymmetry,prediction reversal and constant predictions;population derivatives,quadrature and Hessians;independent full-variable cone projection;and finite corrected-moment feasibility. The special unit-power constant-one-half Gaussian-zonotope case remains classical and attributed. General conic machinery and ordinary-logit existence are not innovation claims. The existing fixed-sample coefficient path and expected choice-loss comparison remain distinct from this increasing-dimension existence theorem.
