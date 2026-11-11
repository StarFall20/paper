# A menu extension of the compensated choice objective

11 November 2026. This is a derivation and validation record for the existing research question. The 32-page binary manuscript has not yet been rewritten around this extension. Priority,proof review and sufficient importance remain separate assessments.

## Model and loss

Fix the number of alternatives $J\geq2$ and put $r=J-1$. Write $E=\{u\in\mathbb R^J:\sum_j u_j=0\}$ and let $P_E$ be orthogonal projection onto $E$. Each menu contains independent Gaussian profile attributes $(S_j,W_j)$,with $q$ common noise slopes. True choice probabilities are

$$\Pr(Y=j\mid S)=\frac{\exp(\gamma S_j)}{\sum_k\exp(\gamma S_k)},\qquad |\gamma|<\infty.$$

The fitted utilities have $J-1$ alternative constants,one common signal slope and $q$ common noise slopes. The true constants and noise slopes are zero. Each noise-attribute column has independent standard Gaussian entries across alternatives and menus,independent of the fixed-dimensional tuple $(S,Y,Z,U)$. A frozen prediction $g(S,Z,U)$ lies in the relative interior of the probability simplex almost surely. Both independent samples follow this joint law. Assume $N/m\to\tau>0$ and $q/m\to\kappa$.

This is conditional choice logit with alternative-specific profiles and common attribute coefficients. It does not cover a general class-specific coefficient matrix acting on one shared covariate vector. The author vignette of [Tan and Bellec2023](https://bellecp.github.io/multinomial.html) explicitly uses that different matrix model and studies inference in an existence region. Only the vignette and proceedings abstract were inspected here.

For utility vector $u$ and a simplex response $y$,use $\ell(u,y)=\log\sum_j\exp(u_j)-y^Tu$. Retain exactly the current objective

$$Q_\lambda=L_m(Y)+\lambda\{L_N(g)-L_m(g)\},\qquad 0<\lambda\leq1.$$

For observed menu features $F_{ij}$,its finite moment support is the weighted Minkowski sum of the menu convex hulls. A single simplex variable per menu preserves the probability-sum constraint. Full contrast rank and a moment in the support interior give a unique finite root. This is classical support geometry;neither this criterion nor its LP implementation is proposed as a new contribution. [Sablica,Hornik and Rusch2026](https://arxiv.org/html/2610.08100v1) supplies recent categorical-likelihood separation context already added to the manuscript.

## The candidate finite-power boundary

The oriented binary intervals become zero-sum menu residual sets

$$D_L=(1-\lambda)\Delta_J-e_Y+\lambda g,\qquad D_U=(\lambda/\tau)(\Delta_J-g).$$

For standard Gaussian $H$ on $E$ and a multiplier $\eta=(a,b)\in E\times\mathbb R$,put $h_\eta=a+bP_ES$ and

$$F(w,tD)=\|w\|^2-\operatorname{dist}^2(w,tD).$$

The extension gives

$$\kappa_c(\lambda)=\inf_{a\in E,b\in\mathbb R}\sup_{t\geq0}\left[E F(H-h_\eta,tD_L)+\tau E F(H-h_\eta,tD_U)\right].$$

Under the stated assumptions,the probability of a unique finite minimizer tends to one when $\kappa<\kappa_c(\lambda)$,and the probability of a finite minimizer tends to zero when $\kappa>\kappa_c(\lambda)$. Equality is left unspecified. The growing dimension is the number of common noise slopes. The signal slope and $J-1$ constants remain fitted but fixed dimensional.

To see the reduction,choose an orthonormal basis of $E$. Project each Gaussian profile column onto this basis. The stacked remaining design is an $nr$ by $q$ independent standard Gaussian matrix. The score event is a nullspace intersection with

$$C=\operatorname{cone}\left(\prod_iD_i\right)\cap\ker(V^T),$$

where $V$ imposes the $r$ constant constraints and the single signal constraint. The product cone is closed and pointed:each labelled chosen coordinate is at most $-\lambda(1-g_Y)<0$ before scaling. The Gaussian nullspace is independent of the conditioned residual sets and multiplier columns. Classical conic kinematics locate the transition through $\delta(C)$,as in the binary proof.

Polyhedral polar calculus gives

$$\delta(C)=E_H\inf_\eta\sup_{t\geq0}\sum_i F(H_i-V_i\eta,tD_i).$$

The multiplier is controlled by a support identity. The labelled and auxiliary prediction terms cancel in their population sum,leaving

$$\rho(\eta)=E\left[\max_j(-h_{\eta,j})+h_{\eta,Y}\right].$$

Conditional on $S$,all true categories have positive probability. A nonzero $(a,b)$ gives a nonconstant utility contrast with positive probability. Thus $\rho$ is strictly positive on the compact unit sphere. An integrable $1+\|S\|$ envelope gives uniform support bounds and a common bounded set for minimizing multipliers. This also places zero in the interior of the low-dimensional residual image with probability tending to one.

The closest labelled residual has norm $\lambda\|g-e_Y\|$:all unchosen components have lower bounds $\lambda g_j$,and their smallest values occur together at the vertex $e_Y$. Its squared mean is positive. On a compact multiplier set,

$$\operatorname{dist}^2(w,tD_L)\geq t^2\lambda^2\|g-e_Y\|^2/2-\|w\|^2.$$

This localizes the maximizing scale. Uniform laws use the integrable squared Gaussian and signal envelopes. They pass the empirical extrema to the population expression without exchanging infimum and supremum. Uniform integrability then passes conditional Gaussian expectation to the statistical dimension. Relative-interior qualification and generic Gaussian position connect the cone intersection to interior moment feasibility. At unit power,the cone span has at most $Nr+1$ dimensions before the fixed constraints,so the strict lower region supplies enough auxiliary contrast rank. At smaller powers all menus supply curvature.

In the symmetric proxy model used for computation,the joint law is invariant under alternative permutations. The objective is convex in $a$;averaging permutations makes $a=0$ optimal. Computation then has one signal multiplier and one scale. The proxy mislabels $Y$ with probability $\epsilon$,uniformly over the other alternatives,and the frozen oracle is its exact conditional posterior given $(S,Z)$.

## The positive-power limit and strict ordinary-logit gap

Put $C_Y=\operatorname{cone}\{e_j-e_Y:j\ne Y\}$. Dividing residuals by positive power and taking the closed increasing limit gives,at common scale $v\geq0$,

$$D_{L,\infty}(v)=v(g-e_Y)+C_Y,\qquad D_{U,\infty}(v)=(v/\tau)(\Delta_J-g).$$

At $v=0$ the labelled tangent cones remain. Dropping that face would remove the ordinary-logit event. The limiting boundary is the same infimum–supremum expression with these sets;the ordinary boundary is

$$\kappa_0=\inf_\eta E\|\Pi_{C_Y}(H-h_\eta)\|^2.$$

The complete positive-power objectives retain the identity $Q_\alpha/\alpha=Q_\lambda/\lambda+(1/\alpha-1/\lambda)L_m(Y)$ for $0<\alpha<\lambda$. The nonnegative human loss makes finite coercive events nested. The closed limiting cone contains every finite positive-power cone and the ordinary feasibility face. Common multiplier/scale bounds and monotone compact convergence give $\kappa_c(\lambda)\uparrow\kappa_\infty$ as positive power decreases. Cone inclusion gives the simultaneous upper bound over all powers,including zero. Fixed-power lower bounds plus nesting give the same strict transition for every almost surely positive,data-dependent $\lambda_m\to0$ in probability. The argument requires no rate condition.

The strict gap has a menu-specific derivative identity. Fix finite $\eta$,write $w=H-h_\eta$,and put $p_Y=\Pi_{C_Y}w$, $z_Y=w-p_Y$. The normal-cone projection has its chosen coordinate at the maximum. Its active unchosen coordinates share that level,so

$$\max_j w_j-w_Y-(g-e_Y)^Tp_Y=\max_j p_{Y,j}-g^Tp_Y.$$

Differentiating the shifted labelled cone and the scaled auxiliary simplex at zero scale therefore yields

$$\frac12\partial_v\Psi_\infty(\eta,0+)=E\left[\max_j p_{Y,j}-g^Tp_Y\right]>0.$$

The expectation is strictly positive:Gaussian $w$ projects to a nonzero point of $C_Y$ with positive probability,and a strictly interior $g$ has weighted average strictly below the maximum of any nonconstant zero-sum $p_Y$. The same coercivity argument gives an attained finite limiting multiplier. Evaluating the derivative there proves $\kappa_\infty>\kappa_0$. This calculation concerns the compensated menu loss,not a new general conic or simplex-projection principle.

## Reductions and finite qualifications

For $J=2$,an orthonormal contrast reduces to the binary calculation. Independent menu signal profiles give signal difference variance two,so the binary signal coefficient is $\sqrt2\gamma$. Twelve matched computations compare this exact parameter convention with the existing binary formula.

At zero signal and $J=3$,the ordinary labelled cone is a planar wedge of angle $\pi/3$. Its classical statistical dimension is $1/2+(\pi/3)/\pi=5/6$. This supplies an independent ordinary-boundary check. It is a geometric reduction,not a new ordinary-logit threshold claim.

Finite menus can have exact support boundaries even with interior predictions and full profile rank. The retained initial four-alternative example has corrected category moment $(7/24,0,17/48,17/48)$. Estimated constants would require a zero mean probability for the second category,which finite softmax utilities cannot supply. Its moment is feasible on a boundary. The corrected validator distinguishes boundary from interior;all initial source snapshots and the exact simulated example remain available.

Projection onto a scaled simplex uses the usual threshold KKT equations. An independent scalar-root calculation agrees within $1.7\times10^{-15}$. The independent generic QP has maximum coordinate discrepancy $2.7\times10^{-7}$ at its stated stopping tolerance and objective discrepancy $1.1\times10^{-13}$;both are retained. The vertex-cone projector agrees with independent NNLS. These numerical facts validate implementation,not priority.

## Relation to the existing real DCE

There is an exact finite-design limitation. If labelled and auxiliary subsets have the same empirical menu frequencies and frozen predictions depend only on those menus,then $L_N(g;\beta)=L_m(g;\beta)$ for every coefficient vector. The correction vanishes at every power. This is an elementary prediction-loss identity.

The current questionnaire repeats eight tasks for every person. Separating versions by their recruitment channels and allocating whole people preserves task balance. Task/version predictions formed from the60 AI pretests have the same frequency in both human subsets. The check uses300 human-recruitment records,keeps the60 pretests outside human labels,and checks120 whole-person allocations. It verifies objective,score,Hessian and returned-root invariance. No phase-law validation or new human preference result is inferred from these repeated fixed menus. The result applies to the checked additive menu logit with those predictions;it does not certify arbitrary person-dependent predictors or latent panel models.

The implication is operational:balanced fixed-task DCE data with menu-only predictions can be screened for an identically zero correction before model fitting. The Gaussian menu boundary addresses varying-profile designs. The finite audit can examine a supplied discrete design while retaining its actual constraints.

## Reading and validation status

The prototype keeps one compensated-objective question,its original prediction correction and its established competitors. General support geometry,conic kinematics,Gaussian rotation and simplex projection retain their prior attribution. The new candidate is the explicit menu-level probability calculation and its all-positive-power derivative argument. The full population grid and a separately planned finite-phase illustration test it. No complete Scopus or Google Scholar review,independent expert proof review,general categorical-design phase law or publication readiness is claimed.
