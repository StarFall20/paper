# Repeated tasks in the same compensated choice objective

12 November 2026. This is a candidate extension under mathematical and numerical review. The delivered 49-page manuscript remains frozen while this argument is checked. Generic conic geometry, ordinary correlated-pattern capacity and cluster-robust PPI inference retain their original attribution. No new questionnaire or human labels are used.

## Statistical question

An analyst can have the same number of people and tasks, the same single-task distribution and the same covariance of aggregate human and prediction scores under two sampling laws. Can the specified compensated marginal logit criterion still have different finite-estimation boundaries? This question concerns dependence inside a person, a common feature of choice data. Cluster-robust sandwich inference already exists. The proposed calculation studies global existence in an increasing-dimension regime.

The criterion is the same task-averaged loss used in the binary paper. Each of m labelled people and N independent auxiliary people supplies T tasks. The loss averages each person's T Bernoulli log losses and then averages people. It estimates a marginal choice target. It is a working marginal criterion; a joint panel likelihood requires its own objective and analysis.

## Model and proposed boundary

Fix T and a positive definite T by T matrix Sigma with unit diagonal. Write A=Sigma^(1/2), using the symmetric positive definite root. For every person, each fitted Gaussian attribute column is A times an independent standard Gaussian T-vector. Columns and people are independent. Estimate an intercept, an unused Gaussian signal slope and q further Gaussian slopes. The true coefficient is zero. Gaussian attributes are independent of all outcomes and proxy flips.

With probability omega all T human outcomes equal one fair Bernoulli draw; otherwise they are independent fair Bernoulli draws. This gives within-person outcome correlation omega. For each task independently, a proxy Z flips Y with probability epsilon in (0,1/2). The frozen prediction is g=epsilon+(1-2epsilon)Z. Its rowwise calibration and accuracy are fixed across covariance structures. Both full-person samples follow this law. There is no partial labelling or sample overlap.

Let N/m tend to tau>0 and q/m tend to kappa. The dimension ratio counts noise slopes per labelled person. Dividing a boundary by T gives the ratio per labelled choice. For a fixed 0<lambda<=1 the raw residual boxes are

D_L=(1-lambda)[0,1]^T-Y+lambda g,

D_U=(lambda/tau)([0,1]^T-g).

For H standard normal in R^T define

B_A(h,t,D)=||H-h||^2-dist^2(H-h,t A^T D).

In the model above, the proposed boundary simplifies to

kappa_c(Sigma,omega,epsilon,lambda,tau)
=sup_{t>=0}{ E B_A(0,t,D_L)+tau E B_A(0,t,D_U) }.

Below this boundary the probability of a unique finite minimizer tends to one; above it the probability of a finite minimizer tends to zero. The equality case is unspecified. The numerical implementation rescales v=lambda*t. It evaluates projections onto transformed boxes by enumerating their faces. Positive definiteness is essential; no singular covariance limit is silently substituted.

## Derivation and proof obligations

Stack all people. Write X_W=blockdiag(A)G for the remaining Gaussian design. The Gaussian matrix G is independent of the fixed-dimensional tuple of outcomes, predictions and fitted low-dimensional attributes. A gradient equation X_W^T r=0 becomes G^T blockdiag(A^T)r=0. Thus the relevant residual set is the product of the transformed boxes A^T D. Whitening attributes and keeping the original independent-coordinate box would give a different problem.

Let V be the two fitted columns for the intercept and unused signal, and Vtilde=blockdiag(A^(-1))V. The finite event is a nonzero intersection of ker(G^T) with

C=cone(product(A^T D)) intersect ker(Vtilde^T).

For fixed positive lambda, every labelled raw residual coordinate has one sign and absolute value at least lambda*epsilon. Invertibility of A gives a positive lower norm bound for each transformed labelled block. This bounds the common conic scale on bounded residual sequences. The cone is closed and pointed. A nonzero point has positive scale and can be divided by that scale to recover raw residual feasibility.

The low-dimensional qualification follows from support cancellation across the two samples. For u_t=eta_0+eta_1 S_t, expected labelled support plus tau times expected auxiliary support is

sum_t E{(-u_t)_+ + Y_t u_t}=one half sum_t E|u_t|.

It is positive for every nonzero eta. Gaussian integrability and compactness of the unit eta sphere supply a common positive lower bound. Independent-person uniform laws put zero in the interior of the low-dimensional residual image with probability tending to one. Generic Gaussian position then relates a nonzero cone intersection to an interior attainable moment. At unit power, the strict lower side has enough auxiliary rows for curvature rank; at smaller positive powers all rows contribute curvature.

Polyhedral polar calculus gives the usual statistical-dimension identity with an infimum over eta and a supremum over t. For the proposed null model the population infimum is attained at eta=0. Conditional Gaussian symmetry removes the unused signal multiplier by Jensen's inequality. Complementing all Y and g, and changing H to -H, makes the intercept objective even. Its convexity puts its minimum at zero, for each fixed scale.

The scale is tight because dist(A^T D_L,0)>=s_min(A)*lambda*epsilon*sqrt(T). On a bounded multiplier set, squared distance supplies a negative quadratic upper envelope as t grows. The support qualification bounds minimizing multipliers as in the independent-row argument. Uniform laws over these compact sets, then Gaussian integration, give delta(C)/m tending to the displayed population value. The established approximate conic kinematic formula supplies the two strict probability sides. These steps extend the same loss and proof route to fixed dependent blocks; they do not supply a new conic theorem. Independent proof review and a rigorous numerical strict-gap bound remain outstanding.

## Covariance match that motivates the experiment

Put a=1-2epsilon, and let R_Y and R_g be the human and prediction residual vectors at zero. Their second moments are

C_Y=[(1-omega)I+omega 11^T]/4,

E(R_Y R_g^T)=a^2 C_Y,

C_g=[a^2(1-omega*a^2)I+omega*a^4 11^T]/4.

For a Gaussian attribute, the human variance, human/prediction covariance and prediction variance of the average cluster score are the entrywise inner products of Sigma with these three matrices, divided by T^2. For the intercept replace Sigma by 11^T. Distinct Gaussian attribute columns have zero cross moments; intercept/attribute cross moments vanish. The population Hessian is one quarter times the identity. Hence these quantities determine the entire human/prediction cluster score covariance, and the whole fixed-dimensional sandwich at every power and auxiliary ratio.

The experiment fixes T=3 and compares

Sigma_A=[[1,.5,.5],[.5,1,.5],[.5,.5,1]],

Sigma_B=[[1,.99,.255],[.99,1,.255],[.255,.255,1]].

Both matrices have trace 3 and total off-diagonal sum 3. Their score covariance matrices, including the intercept, agree exactly for the stated common outcome/proxy law. Their Gaussian dependence geometry differs. No samplewise identity of realized scores is claimed. The positive-power population calculations and matched finite designs test whether the boundary distinguishes these models.

At omega=1, epsilon=.25 and tau=1 the existing cluster-aware scalar variance rule for a Gaussian slope selects lambda=.8 in both models. At omega=.75 it selects 14/19 in both. The original illustrative simulation powers are fixed at 1,.8,.5,.1; .8 is not called variance-optimal in the omega=.75 finite experiment.

## Classical controls and prior contribution

For Sigma=I the squared distance splits coordinatewise. The boundary is exactly T times the existing independent-row binary result, irrespective of omega. This is a required reduction, not new evidence of a generic dependence effect.

At zero power, ordinary correlated-pattern capacity is classical. For three tasks, a transformed positive orthant has statistical dimension

delta=3/2+[sum_{i<j}arcsin(corr(Sigma^(-1))_ij)-sum_{i<j}arcsin(Sigma_ij)]/(2*pi).

The ordinary boundary in this mixture is omega*delta+(1-omega)*3/2. The formula follows from classical Gaussian orthant probabilities, polar cones and intrinsic-volume identities. Lopez, Schroeder and Opper (1995, equations 10 and 12) already give the pair-pattern capacity and mixtures; correlated ordinary capacity is not a new claim here. The compensated block calculation must bear any proposed increment.

Broska and Howes (2026, Appendix A, Theorems 2 and 3) already develop cluster-robust PPI++ inference and power tuning for general M-estimators, including partial labelling. Their inspected statements concern fixed-dimensional central limits under stated regularity. This extension studies the proportional finite-minimizer event for the specified marginal compensated loss and fully separated person samples. It makes no new standard-error, partial-labelling or efficiency claim. A general claim that local covariance alone fails to characterize a global fit is also established; the proposed increment is the explicit dependence-sensitive compensation boundary and the matched construction within this model.

## Primary sources

- Broska, D., Howes, M. (2026). [Cluster-Robust Prediction-Powered Inference](https://arxiv.org/html/2610.09601v1). Inspected Sections 1,2.2,2.4,2.7 and Appendix A.1-A.4, Theorems 2 and 3.
- Lopez, B., Schroeder, M., Opper, M. (1995). [Storage of correlated patterns in a perceptron](https://arxiv.org/html/cond-mat/9505093). Inspected pair construction and equations 10-16. Published DOI10.1088/0305-4470/28/16/005.
- Amelunxen, D., Lotz, M., McCoy, M.B., Tropp, J.A. (2014). [Living on the edge](https://arxiv.org/html/1303.6672). Established statistical dimension, product/polar properties and approximate conic kinematics.
- Koriyama, T., Bellec, P.C. (2025). [General single-index M-estimator existence](https://arxiv.org/html/2501.03163v2), Assumption 2 and Theorem 2.4. Its finite-row-infimum assumption remains separate from this compensated criterion; an assumption mismatch does not certify novelty.
- [Pinned author implementation](https://github.com/Michael-Howes/ppi_py/blob/2dc0ceede320ddf81b4d1050df04fbd1f48bd31b/ppi_py/cluster_ppi.py), `_ppi_cov`, `_calc_lam_opt`, mean and OLS entry points. Read only; its code was not executed or copied into the analysis.
