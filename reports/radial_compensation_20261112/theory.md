# Known task scale and compensation at equal local information

12 November 2026. Candidate extension under review. This construction addresses a limitation of the preceding repeated-task comparison: ordinary correlated-pattern capacity also changed there. Here the ordinary human separation event is identical on every paired sample. The delivered 49-page manuscript remains preserved while the extension and its priority are checked.

## Model

Let R=(1,S,W), with independent standard Gaussian S and q-vector W. The observed regressor is X=D R, where a specified positive task precision D is independent of (S,W,Y,Z). It multiplies the entire utility, including the fitted intercept. D has a fixed distribution supported on a compact subset of (0,infinity), with E D^2=1. All true coefficients are zero, and Y is fair Bernoulli. An independent proxy flip has probability epsilon in (0,1/2), and g=epsilon+(1-2epsilon)Z is the oracle conditional probability E(Y|X,Z). Both independent samples have the same scale distribution.

The marginal logit model is P_beta(Y=1|D,R)=sigma(D R^T beta). The known scale is an exogenous specification. These null-signal choices do not identify an unknown task-scale distribution. A model with an unscaled common intercept is a separate specification.

The compensated criterion remains Q_lambda=L_m(Y)+lambda{L_N(g)-L_m(g)}, at fixed positive lambda and N/m tending to tau. The parameter ratio is q/m. The loss, label allocation and prediction rule are shared across the scale laws.

## Properties fixed by construction

For each realized labelled sample and every beta, the sign of (2Y-1)X^T beta equals the sign of (2Y-1)R^T beta, because D>0. The complete ordinary human separation event is identical before and after scaling. This is classical positive row-scale invariance, not a new result. The ordinary null-logit threshold remains 1/2, including the fitted intercept.

At zero, the full population Hessian is H=I/4. With a=1-2epsilon, the full human score covariance is I/4, the prediction-score covariance a^2 I/4, and the human/prediction cross covariance a^2 I/4. These identities include the scaled intercept. They follow from independence and E D^2=1. Both scale laws have the same entire regular, fixed-dimensional covariance at every power:

V_lambda=(4/m){1-2lambda*a^2+(1+1/tau)lambda^2*a^2}I.

The established oracle variance rule selects lambda=tau/(1+tau), which is 1/2 at tau=1. Calibration, prediction accuracy, second-moment information and the human separation event are all fixed. The realized empirical score and Hessian need not agree between scaled samples.

## Proposed radial boundary

Let J_lambda(v)=1+tau-F_lambda(v), where F_lambda is the independent-row balanced profile already derived in the paper. With f(x)=E(H-x)_+^2, e=epsilon and b=(1-lambda)/lambda,

F_lambda(v)=(1-e){f(-ev)+f((b+e)v)}
+e{f(-(1-e)v)+f((b+1-e)v)}
+tau{f((1-e)v/tau)+f(ev/tau)}.

The candidate extension is

kappa_c(D)=sup_{v>=0} E J_lambda(Dv).

This follows by conditioning on D and pulling positive row scales from X into each residual interval. The remaining Gaussian columns are again independent standard normal. The fitted low-dimensional columns in this representation are (1,S). Label/proxy symmetry and independence of S put both population multipliers at zero. The scale is tight because D is bounded away from zero for each fixed distribution. The same support qualification, compact uniform laws and established Gaussian kinematic argument then give the two strict probability sides of the boundary. Equality remains unspecified. This is an extension of the specified compensated objective; it is not a new estimator or a new random-subspace theorem.

The ordering of infimum, supremum and expectation remains the same as in the original proof. A single common conic scale v must serve every row. Replacing the boundary by E sup_v J_lambda(Dv) changes that problem.

## Strict loss of the existence region

J_lambda(0)=0 and J'_lambda(0)=sqrt(2/pi)/lambda. Since f''(x)=2Phi(-x), the labelled f(-rv) terms give

J''_lambda(v)<=-c, v>=0, c=E|Y-g|^2=epsilon(1-epsilon)>0.

The other second-derivative terms are nonpositive. Hence J has a unique positive maximizer v_star. For nonconstant D, no v makes Dv equal v_star almost surely. The radial profile has a strict maximum below max J_lambda. Compactness follows from strong concavity and E D^2>0. Thus every nonconstant independent positive scale law strictly shrinks this balanced compensated existence region, even though ordinary human separation and the full regular covariance remain fixed.

Strong concavity also gives an explicit quantitative upper bound:

J_lambda(u)<=J'_lambda(0)u-(c/2)u^2,

kappa_c(D)<=(E D)^2/[pi*lambda^2*epsilon(1-epsilon)*E D^2].

At unit power, labelled intervals are points, so their curvature is exactly -2c. The sharper bound is

kappa_c(D)<=(E D)^2/[2*pi*epsilon(1-epsilon)*E D^2], lambda=1.

The inequalities are mathematical upper bounds, not fitted approximations.

## Collapse at fixed covariance and identical human separability

Let D take a>0 with probability 1-p and b_a=sqrt([1-(1-p)a^2]/p) with probability p, for a small enough that the radicand is positive. E D^2=1 for every member. For any fixed p>0, as a decreases to zero,

kappa_c(D) tends to p*kappa_c(1).

To see this, substitute u=b_a*v. The profile becomes p J_lambda(u)+(1-p)J_lambda((a/b_a)u). Strong concavity bounds maximizers on one compact set for fixed p. The second term tends uniformly to zero there, giving the limit. Choosing p small makes the boundary arbitrarily close to zero. Each distribution used in the proportional limit still has a fixed strictly positive minimum and a fixed finite maximum. No triangular-array exchange or singular scale is needed.

Equivalently, the analytic upper bound tends to zero with (E D)^2/E D^2. For any fixed positive power and any kappa in (0,1/2), some admissible positive scale law makes the compensated objective fail to have a finite minimizer with probability tending to one while ordinary human logit retains one with probability tending to one. This includes the existing oracle variance-optimal power, which remains unchanged across the scale laws.

For tau=1, epsilon=.25, p=.1 and a=.1, initial scalar calculations give a boundary near .0604 at unit power and .1120 at variance power .5. Constant scale gives .3554 and .6029. The simulation fixes q/m=.25 and compares these laws on the same Gaussian rows, labels and proxy draws. A separate small-p family illustrates the analytic arbitrary-collapse bound. These numerical values require independent integral and original-objective replay before being used in a manuscript claim.

## An explicit comparison with a strict gap independent of numerical integration

Take tau=1, epsilon=1/4 and the existing oracle variance power lambda=1/2. Let the slope-to-label ratio be kappa=1/4. Under constant D=1, the profile has F''(v)<=47/16 for v>=0: the negative labelled arguments contribute at most 2E r^2=6/16, the positive labelled arguments at most E(1+r)^2=31/16, and the auxiliary arguments at most 10/16. It follows that

kappa_c(1)>=J(1/2)>=sqrt(2/pi)-47/128>213/640>1/4.

The strict rational comparison uses pi<4 and 1/sqrt(2)>7/10.

For the second law take p=1/100, a=1/1000 and b=sqrt(99.999901). This gives E D^2=1 exactly and E D<101/1000. The general strong-concavity bound and pi>3 give

kappa_c(D)<(64/9)(101/1000)^2=652864/9000000<1/4.

The constant-scale compensated criterion consequently has a finite unique minimizer with probability tending to one, while the heterogeneous-scale criterion has none with probability tending to one. Ordinary human logit has the same finite-estimation event in every paired realization, and its probability tends to one in both. The entire fixed-dimensional oracle covariance is (7/2m)I for both corrected criteria. These inequalities establish the strict boundary ordering without relying on the optimized numerical values. Their applicability still depends on completing the stated radial extension proof. The finite simulation uses the less extreme p=.1,a=.1 law, and is reported as a separate illustration.

## Contribution limits and literature audit

Positive row-scale invariance of ordinary separation, local asymptotic covariance, variance-optimal PPI tuning and conic kinematics are established. The proposed increment is the dependence of the compensated finite-estimation boundary on the full positive-scale law and the collapse construction that fixes the human separation event and complete local information. It does not claim a generic first-order/global distinction, a new scale-heterogeneity model, or superior risk to ridge, pooling, one-step PPI or complete GMM.

The core Gaussian boundary must be extended explicitly to X=D(1,S,W); the original theorem cannot simply be cited as already covering these non-Gaussian columns. The known-precision interpretation must stay explicit. No observed scale is inferred from the private questionnaire, and no field evidence is supplied by these simulations. Closest literature on elliptical logistic separation, scale heterogeneity in choice models and corrected-loss existence still needs theorem-level comparison. The current search is targeted and is not a completed Scopus/Google Scholar systematic review or a priority certificate.


## Primary-source boundaries

Hess and Train (2017), [Journal of Choice Modelling 23, 1-8](https://eml.berkeley.edu/~train/papers/scale.pdf), Section5 on published pages6-7, explains scale/taste identification limits and the need to scale all utility coefficients. The present calculation specifies that scale instead of estimating latent heterogeneity.

Tang and Ye (2020), [Theorem2.7 and Assumptions2.3-2.6](https://arxiv.org/html/1908.06208), develops ordinary binary-response existence for elliptical covariates. The ordinary separation control remains established; the subtracted two-sample objective requires its stated separate calculation.

[The source comparison](closest_prior_and_access.json) also records the inspected general M-estimator theorem and cluster-PPI work. No unavailable full text, inaccessible database query or absence of a search hit is used to certify priority.
