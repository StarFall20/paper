# A simultaneous upper bound and jointly vanishing positive powers

10 November2026. This derivation is recorded before the planned numerical checks. It strengthens the existing balanced endpoint result within the same corrected logit question. It uses classical cone geometry; it proposes no new general kinematic theorem or inferentially optimal weight.

## Model and statement

Take independent standard Gaussian attribute rows with p slopes and a fixed zero intercept. Human choices are balanced and independent of the attributes. The binary proxy independently flips each choice with fixed probability e in (0,1/2], and g=e+(1-2e)Z. Let N/m tend to tau>0 and p/m tend to kappa. Define kappa_max=1+tau-min_{v>=0}G(v), with G from the existing positive-power derivation. Its minimizer is positive and kappa_max>1/2.

Below kappa_max, some fixed strictly positive power has a unique finite corrected root with probability tending to one. Above kappa_max, the probability that ANY power in (0,1] has a finite root tends to zero. Ordinary zero power also fails above this bound because its balanced Gaussian threshold is1/2.

Consequently any strictly positive random powers lambda_m in (0,1] tending to zero in probability have the same strict two-sided existence boundary kappa_max. They may depend on all the observed data; independence of the selected powers is unnecessary. This is a finite-root existence statement. It supplies no coefficient error, distribution or interval guarantee. The equality case is excluded. A rule assigning exact zero with nonvanishing probability is outside this positive-power conclusion.

## Limiting cone and its zero-scale face

Orient labelled Gaussian rows by 1-2Y and auxiliary rows by 2Z-1. Their joint distribution remains independent standard Gaussian, conditional on the labels and proxies. Labelled interval lower endpoints after division by positive power are r_i=|Y_i-g_i|, taking e with probability1-e and1-e with probability e. Their upper endpoints are r_i+1/lambda-1. The oriented auxiliary interval is the fixed [a,b]=[-(1-e)/tau_m,e/tau_m].

Define the closed polyhedral cone K_max of vectors u for which there exists t>=0 with

u_Li>=t r_i, and t a<=u_Uj<=t b.

At t=0 this cone includes nonnegative labelled vectors with zero auxiliary coordinates. These nonzero rays MUST be retained. Treating the cone as only t>0 would lose closedness and invalidate a statistical-dimension argument. Equivalently, K_max is the closure of the increasing union of the finite positive-power interval cones.

For every positive power, its oriented interval cone is contained in K_max. Thus finite feasibility at ANY power implies that the same Gaussian nullspace has a nonzero intersection with K_max. This inclusion supplies a simultaneous bound, without an uncountable union of probability bounds or a uniform approximation over powers.

## Statistical dimension of the limiting cone

For an independent standard normal vector H, the squared projection norm onto K_max is the supremum over t>=0 of

sum_label[H_i^2-(t r_i-H_i)_+^2]
+sum_aux[H_j^2-dist(H_j,[t a,t b])^2].

At t=0 the labelled contribution is (H_i)_+^2; it is not zero. Normal expectations divided by m give 1+tau-G(t). The labelled r_i are bounded below by the fixed e>0. Their quadratic distance term bounds all maximizing scales on compact sets in probability. Uniform laws of large numbers there and the bound 0<=projection norm squared<=||H||^2 give convergence in L1. Taking the conditional expectation over H yields delta(K_max)/m -> kappa_max in probability.

The classical approximate conic kinematic theorem now bounds the random nullspace intersection. For kappa>kappa_max, the intersection is absent with probability tending to one. The cone inclusion proves that no positive power can restore a finite root in this region. This proves the required upper side directly for all powers, including data selected ones.

## Lower side and the probability sandwich

For kappa<kappa_max, continuity of the previously proved fixed-power boundaries from the right gives a fixed lambda_0 in (0,1) with kappa<kappa_c(lambda_0). Its finite-root event has probability tending to one. The stacked Gaussian design has full column rank with probability one on this strict lower side, since kappa_max<1+tau.

Let E_m(lambda) be its unique finite-root event. The fixed-sample nesting identity implies E_m(lambda_0) is contained in E_m(lambda) for every0<lambda<=lambda_0. Therefore

P(E_m(lambda_m)) >= P(E_m(lambda_0))-P(lambda_m>lambda_0).

The right side tends to one whenever lambda_m>0 almost surely and lambda_m->0 in probability. The upper side already applies simultaneously to every positive selected power. No rate condition such as lambda_m*m->infinity enters an exact existence event. Numerical evaluation and coefficient accuracy can still deteriorate at small power.

## Relationship to prior and the existing manuscript

The upper bound and sandwich strengthen the former iterated-only result for THIS balanced fixed-intercept model. The general signal/intercept theorem remains a fixed-positive-power statement. Extending this joint assertion to its full model requires its own multiplier controls and is not asserted here.

[Amelunxen etal.2014](https://arxiv.org/abs/1303.6672) supplies the random-subspace machinery. [PPI++ Theorem1 and Section6](https://arxiv.org/html/2311.01453v2) supplies fixed-dimensional inference and established power tuning; those claims are not displaced by this proportional-dimensional existence calculation. [Li and Huang2026](https://arxiv.org/html/2407.04194v5) develops positive synthetic-likelihood regularization. The present compensated loss has distinct power-dependent recession geometry. Generic small regularization coefficient divergence remains prior work. The candidate increment is the explicit corrected-loss upper envelope and its finite-root consequence, not a universal theory of optimal tuning.
