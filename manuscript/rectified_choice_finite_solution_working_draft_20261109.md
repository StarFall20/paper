# Finite solutions in prediction-assisted binary choice estimation

Working research draft,9 November2026. The existence results below are candidate mathematical contributions. Priority and sufficient JOCM contribution have not been established. This document does not replace the separately retained300-record human application.

## Research question

Prediction-assisted choice estimation combines a small collection of observed choices with a larger set of predicted choices. A labelled-data correction can preserve the population target. For a finite label budget, the corrected loss can have no finite parameter minimizer. In that event, additional optimizer iterations cannot recover an unpenalized root. We study how the prediction error and the number of utility contrasts determine the existence of that root.

The immediate evidence comes from a controlled audit of released vaccine choices. Component clipping of corrected one-hot targets exactly converts the labelled targets back to observed choices. It produces a soft-training objective whose finite optimum need not solve the original corrected score. Removing clipping reveals exterior corrected moments in5 of500 allocations at the smallest label budget. A separate separating-ray calculation verifies each case. This motivates an existence question with a fixed estimator and a fixed target.

## Estimator and proposed result

For logistic loss ℓ(u,y)=log(1+exp(u))−yu, a frozen prediction g and an unlimited independent auxiliary pool, consider

Q_m(α,β)=E_Xℓ(α+X'β,g(X))+m⁻¹Σ_i{ℓ(α+X_i'β,Y_i)−ℓ(α+X_i'β,g(X_i))}.

Under X~N_p(0,I), independent labelled observations and a balanced null Y~Bernoulli(1/2), constant g=q yields an exact probability of a finite minimizer. It is a binomial mixture of chi-square probabilities, displayed with its proof in [the mathematical appendix](../reports/public_conjoint_audit_20261109/existence_candidate_derivation.md). As p/m→κ, the candidate critical ratio is

κ_c(q)=2/[π{1+4(q−1/2)²}].

For nonzero logistic signal γX₁ and a frozen bounded predictor depending on a fixed-dimensional Gaussian subspace containing the signal, the candidate boundary becomes

κ_c(γ,g)={φ(0)²−a(γ)²}/E{Y−g(X)}²,

where a(γ)=E[Gσ(γG)]. Predictors with the same classification boundary can yield different finite-root existence regions through their Brier errors. Independent auxiliary pools with N/p→∞ retain the strict-side limits. The exact finite-label probability applies to the population auxiliary limit.

The Gaussian lift zonoid, exponential-family support criterion, ordinary-logit separation boundary and Brier optimality are established tools. The claimed candidate increment is the explicit existence law for this rectified loss. It is not a new generic separation test. The review of [closest prior work](../reports/public_conjoint_audit_20261109/literature_boundary.md) records the remaining priority question.

## Evidence presently available

The matched vaccine audit holds the1200-row covariate pool, target projection, label allocation and nuisance predictions fixed within each clipping comparison. It uses500 allocations and50,100,150,200 labelled choice rows. Source-defined five-row groups are preserved during allocation, without claiming verified respondent identifiers. The reference vector released with the data is also audited against a fully converged fit on4605 human rows. Its discrepancy is reported as an optimization issue, without attributing all published gains to that issue.

The six-method comparison retains the strong point-error performance of clipped soft training and the finite-root failures of the two unpenalized corrected methods. Separate support checks and direct recession rays identify the latter cases. The stabilized parameter-error table, all failure counts, plans and definitions are in [the audit report](../reports/public_conjoint_audit_20261109/公开数据重分析与创新进展.md). Confidence-interval coverage and power-tuned methods are not evaluated in this experiment.

Gaussian simulations use160 exact-probability conditions and480 signal/predictor conditions, each with2000 direct draws of the original labelled X,Y experiment. Numerical population integration checks nine interior roots and six exterior rays. These results verify calculations and illustrate the boundary. They do not certify originality or turn the Gaussian assumptions into a sample-size rule for categorical DCEs.

## Evidence needed for a submission claim

A useful choice-methods paper needs a result that changes an analyst's action. The next comparison should include the established power-tuned estimator, explicit treatment of nonfinite roots and finite auxiliary pools. It should evaluate the objective actually solved, point error and coverage together. Strong prediction quality can improve both precision and existence, while clipping can trade correction for a different finite objective; a useful treatment must make that decision transparent.

The current300-record Chinese service-choice application remains an exploratory application with bundled attributes and channel/version confounding. It cannot provide randomized validation of the Gaussian theorem. Its results should enter a future paper only if a clearly specified analysis question connects them to the final contribution.
