# Label budgets and finite estimates in prediction-assisted choice modelling

Working research draft,9 November2026. The existence results below are candidate mathematical contributions. Priority and sufficient JOCM contribution have not been established. This document does not replace the separately retained300-record human application.

## Abstract

Prediction-assisted estimation uses observed choices to correct a larger collection of predicted choices. The corrected objective can lack a finite parameter minimizer. We characterize this event for binary logit with Gaussian attributes and a frozen auxiliary predictor. An exact finite-label distribution and a proportional-dimension boundary relate finite estimability to the utility signal and prediction residual risk. An informative-proxy example places the boundary at a population variance-optimal weighting limit. Direct simulations cover balanced and nonzero utility signals. A controlled reanalysis of released vaccine choices examines the objective solved by pseudo-label clipping, finite-root failures and established power tuning. The results separate statistical efficiency from finite estimability of the specified objective and identify the label-budget conditions under which an unpenalized corrected fit can be computed.

## Research question

Prediction-assisted choice estimation combines a small collection of observed choices with a larger set of predicted choices. A labelled-data correction can preserve the population target. For a finite label budget, the corrected loss can have no finite parameter minimizer. In that event, additional optimizer iterations cannot recover an unpenalized root. We study how the prediction error and the number of utility contrasts determine the existence of that root.

The immediate evidence comes from a controlled audit of released vaccine choices. Component clipping of corrected one-hot targets exactly converts the labelled targets back to observed choices. It produces a soft-training objective whose finite optimum need not solve the original corrected score. Removing clipping reveals exterior corrected moments in5 of500 allocations at the smallest label budget of the pooled-imputation comparison. A separate separating-ray calculation verifies each case. A second experiment evaluates the published auxiliary-only PPI objective and power tuning. These are distinct objectives with separately reported failure counts.

The theoretical question uses a fixed estimator and target. Existing results characterize ordinary-logit separation, Gaussian attainable moments and prediction-powered efficiency. We use those tools to derive the finite-label existence probability of the rectified objective. The informative-proxy example addresses a setting where the predictions contain useful information and the unit-weight limit is variance-optimal.

## Estimator and proposed result

For logistic loss ℓ(u,y)=log(1+exp(u))−yu, a frozen prediction g and an unlimited independent auxiliary pool, consider

Q_m(α,β)=E_Xℓ(α+X'β,g(X))+m⁻¹Σ_i{ℓ(α+X_i'β,Y_i)−ℓ(α+X_i'β,g(X_i))}.

Under X~N_p(0,I), independent labelled observations and a balanced null Y~Bernoulli(1/2), constant g=q yields an exact probability of a finite minimizer. It is a binomial mixture of chi-square probabilities, displayed with its proof in [the mathematical appendix](../reports/public_conjoint_audit_20261109/existence_candidate_derivation.md). As p/m→κ, the candidate critical ratio is

κ_c(q)=2/[π{1+4(q−1/2)²}].

For nonzero logistic signal γX₁, let a frozen bounded predictor depend on a fixed-dimensional Gaussian subspace and an auxiliary proxy Z. Conditional on that subspace, Y and Z must be independent of the remaining Gaussian coordinates. The candidate boundary becomes

κ_c(γ,g)={φ(0)²−a(γ)²}/E[(Y−g(X,Z))²],

where a(γ)=E[Gσ(γG)]. Independent auxiliary pools with N/p→∞ retain the strict-side limits. Exact finite-label probabilities refer to the population auxiliary limit.

For an informative balanced-null proxy that flips the true choice with fixed probability ε, the conditional-mean oracle has residual risk ε(1−ε). The boundary becomes1/[2π ε(1−ε)]. A triple-binomial mixture gives its finite-label probability. Its population variance-optimal power tends to1 as the auxiliary pool grows, provided0<ε<1/2. This example shows that the finite-root boundary concerns an informative correction. It is a property of the specified unpenalized objective; penalized and one-step estimators have different definitions.

The Gaussian lift zonoid, exponential-family support criterion, ordinary-logit separation boundary and Brier optimality are established tools. The claimed candidate increment is the explicit existence law for this rectified loss. It is not a new generic separation test. The review of [closest prior work](../reports/public_conjoint_audit_20261109/literature_boundary.md) records the remaining priority question.

## Evidence presently available

The matched vaccine audit holds the1200-row covariate pool, target projection, label allocation and nuisance predictions fixed within each clipping comparison. It uses500 allocations and50,100,150,200 labelled choice rows. Source-defined five-row groups are preserved during allocation, without claiming verified respondent identifiers. The reference vector released with the data is also audited against a fully converged fit on4605 human rows. Its discrepancy is reported as an optimization issue, without attributing all published gains to that issue.

The six-method comparison retains the strong point-error performance of clipped soft training and the finite-root failures of the two unpenalized corrected methods. Separate support checks and direct recession rays identify the latter cases. The stabilized parameter-error table, all failure counts, plans and definitions are in [the audit report](../reports/public_conjoint_audit_20261109/公开数据重分析与创新进展.md). A separate power-tuning experiment uses the same allocations and reports human-root and source-style PPI-root initializers. Its finite-root and error tables must accompany the original fixed-weight results. Neither experiment certifies interval coverage.

Gaussian simulations use160 exact-probability conditions and480 signal/predictor conditions, each with2000 direct draws of the original labelled X,Y experiment. Numerical population integration checks nine interior roots and six exterior rays. These results verify calculations and illustrate the boundary. They do not certify originality or turn the Gaussian assumptions into a sample-size rule for categorical DCEs.

The informative-proxy extension adds135 exact-probability conditions and162 nonzero-signal conditions, each with2000 direct X,Y,Z draws. Its uninformative endpoint agrees with the original exact formula within2e-14. The proxy posterior agrees with Bayes' rule within1.34e-15. These calculations test the stated estimator without training or querying a new AI model.

## Evidence needed for a submission claim

A useful choice-methods paper needs a result that changes an analyst's action. The completed point-estimation comparisons retain established power tuning and every unavailable root or initializer. The next substantive evaluation concerns finite auxiliary pools, existing one-step alternatives and interval coverage under a known choice DGP. Strong prediction quality can improve both precision and existence, while clipping changes the objective being solved. A useful treatment must make that decision transparent.

The current300-record Chinese service-choice application remains an exploratory application with bundled attributes and channel/version confounding. It cannot provide randomized validation of the Gaussian theorem. Its results should enter a future paper only if a clearly specified analysis question connects them to the final contribution.
