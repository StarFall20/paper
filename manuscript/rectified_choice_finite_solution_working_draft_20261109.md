# Label budgets and finite estimates in prediction-assisted choice modelling

Working research draft, updated 10 November 2026. This version incorporates finite-auxiliary coverage experiments and the signal/intercept boundary. The mathematical statements remain candidates pending priority and proof review. It does not replace the separately retained 300-record human application.

## Abstract

Prediction-assisted choice estimation corrects predicted choices using a smaller sample of observed choices. Its unpenalized corrected objective can lack a finite minimizer. We study this event for binary logit with Gaussian attributes and a frozen low-index predictor. With finite labelled and auxiliary samples, a computable phase boundary depends jointly on utility signal, proxy quality, the auxiliary-to-label ratio and correction weight. Population-auxiliary limits yield exact finite-label laws. A released vaccine-choice reanalysis retains established power tuning, unavailable initializers and separately verified exterior moments. A known-truth experiment on the released attribute support compares seven estimators, including one-step and penalized alternatives. It reports both interval availability and conditional coverage. These results distinguish the existence of a finite unpenalized fit from the reliability of its uncertainty estimates.

## 1. The analysis decision

Choice analysts can use predicted responses to supplement a limited set of observed choices. A correction based on the observed choices preserves the population score. At a small label budget, the corrected empirical objective can have no finite minimizer. More optimizer iterations cannot recover an unpenalized root in that event. An analyst needs to know which objective is being fitted, whether its root exists, and whether the reported intervals are reliable.

A controlled audit of released vaccine choices motivates the problem. Component clipping of corrected one-hot targets exactly returns the observed labels on labelled rows and the soft predictions on auxiliary rows. The resulting soft-training fit need not solve the original corrected score. Removing clipping reveals exterior corrected moments in 5 of 500 allocations at the smallest label budget of the pooled-imputation experiment. A separate recession-ray calculation verifies every case. The clipped estimator's strong point-error performance is retained in the comparison.

Existing work supplies rectified estimation, power tuning, one-step alternatives and logistic separation theory. The candidate contribution is an explicit finite-estimability boundary for a specified corrected choice objective, including finite auxiliary budgets. Its practical assessment uses those established alternatives. Small-sample interval instability and near-separation have also been discussed in recent PPI comparisons; observing them is not an independent contribution. The [closest-prior review](../reports/public_conjoint_audit_20261109/literature_boundary.md) records this boundary and the remaining priority question.

## 2. The corrected estimator

Write ℓ(u,y)=log(1+exp(u))−yu. Let L_m(Y;β) average this loss over m labelled observations and L_N(g;β) average it over N independent auxiliary observations with frozen predictions g. For fixed positive power 0<λ≤1, the established PPI objective is

Q_λ(β)=L_m(Y;β)+λ{L_N(g;β)−L_m(g;β)}.

Its empirical Hessian combines the labelled logistic Hessian with weight 1−λ and the auxiliary Hessian with weight λ. The corrected linear score moment may lie outside the attainable weighted logistic moments. An interior moment and a full-rank positively weighted design give a unique finite root. Support calculations and separating rays are classical tools; we use them to distinguish a missing root from numerical nonconvergence.

Penalization changes Q_λ. A one-step estimator defines a different operation from solving its root. Both are relevant alternatives, and their interval behavior requires separate evaluation. A finite output alone does not establish nominal coverage.

## 3. Finite auxiliary budgets: a candidate phase boundary

First consider a balanced Gaussian benchmark. Attributes satisfy X~N_p(0,I), choices Y~Bernoulli(1/2), and an informative proxy Z independently flips Y with probability ε in (0,1/2). The frozen conditional-mean oracle is g(Z)=ε+(1−2ε)Z. The model estimates p slopes and fixes its intercept at zero. Let N/m→τ>0 and p/m→κ.

Define f(a)=E[(G−a)_+²]=(1+a²)Φ(−a)−aφ(a) for G~N(0,1), and

F(t)=(1−ε){f(−λεt)+f((1−λ+λε)t)}
+ε{f(−λ(1−ε)t)+f((1−λε)t)}
+τ{f(λ(1−ε)t/τ)+f(λεt/τ)}.

The candidate critical dimension-to-label ratio is

κ_c(τ,ε,λ)=1+τ−min_{t≥0}F(t).

The strict-side statement is that the probability of a unique finite minimizer tends to one when κ<κ_c and to zero when κ>κ_c. No equality-boundary probability is asserted. Zero power is a separate ordinary-logit endpoint. A free intercept and nonzero utility signal are outside this finite-ratio statement.

The proof orients the feature rows by observed choice/proxy signs. Conditional on those signs, the resulting map remains independent Gaussian. Finite-root existence reduces to a random nullspace meeting the cone generated by a product of intervals. The classical ALMT random-subspace theorem places the transition at that cone's statistical dimension. The explicit intervals give the scalar formula above. Cone geometry and its threshold theorem remain attributed prior tools. The full argument, qualifications and [completed numerical checks](../reports/public_conjoint_audit_20261109/finite_ratio_boundary_candidate.md) accompany the statement.

Finite auxiliary counts matter. At τ=1 and ε=0.25, unit power gives κ_c≈0.355. The existing population variance-optimal power λ*=τ/(1+τ)=0.5 gives κ_c≈0.603. The infinite-auxiliary unit-power value is about 0.849. The weight formula is established PPI theory; the candidate result quantifies the finite-root boundary at that weight. These values describe the Gaussian benchmark, without prescribing a sample size for a categorical DCE.

### Utility signal and a free intercept

Let X=(S,W), with S standard normal, P(Y=1|S)=σ(γS), and q independent Gaussian coordinates W that are independent of (S,Y,Z). The estimator now includes an intercept and the signal slope. A fixed low-index prediction g(S,Z) lies strictly between zero and one; it need not be the oracle. Orienting the two samples gives bounded random intervals I_L and I_U, whose endpoints are stated in [the signal appendix](../reports/public_conjoint_audit_20261109/finite_ratio_signal_candidate.md).

For B(h,t,a,b)=1+h²−f(−h−ta)−f(h+tb), the candidate boundary in q/m is

κ_c=inf_(η₀,η₁) sup_(t≥0) [ E B(s_L(η₀+η₁S),t,I_L)+τ E B(s_U(η₀+η₁S),t,I_U) ].

Here s_L=1−2Y and s_U=2Z−1. Under the oracle's joint sign symmetry, η₀=0 is optimal, leaving two scalar operations. The proof controls the multiplier using positive population logistic overlap and controls the cone scale using positive primary residuals. Uniform convergence and an integrable bound then justify the statistical-dimension limit. The generic conic theorem is the same established tool as in the null case.

The completed oracle experiment fixes ε=0.25. At τ=1 and γ=2, unit power gives a candidate boundary of 0.2474 and power0.5 gives 0.4080. The classical ordinary-logit boundary is 0.3449. Utility signal changes the existence comparison. The theorem concerns fixed positive powers; the empirical power-tuning estimators in Section 5 have a different data-dependent rule. Non-oracle phase validation and a useful practical analysis rule remain to be assessed.

## 4. Population-auxiliary limits and utility signal

With an unlimited independent auxiliary pool, constant prediction g=q in the balanced Gaussian model gives an exact binomial-mixture finite-label probability. As p/m→κ, its candidate boundary is

κ_c(q)=2/[π{1+4(q−1/2)²}].

For logistic utility signal γX₁ and a frozen predictor based on a fixed-dimensional Gaussian subspace and an auxiliary proxy, assume that Y and the proxy are conditionally independent of the remaining Gaussian coordinates. The population-auxiliary candidate becomes

κ_c(γ,g)={φ(0)²−a(γ)²}/E[(Y−g(X,Z))²],

where a(γ)=E[Gσ(γG)]. Independent auxiliary pools with N/p→∞ retain the strict-side limits. Gaussian lift-zonoid geometry and Brier-risk optimality are established results used in this derivation.

For the informative balanced-null oracle, residual risk is ε(1−ε), and the unit-power boundary is 1/[2π ε(1−ε)]. A triple-binomial mixture gives its finite-label probability. The finite-ratio formula in Section 3 approaches the same limit as τ grows. Exact laws, assumptions and proofs are in [the population-auxiliary appendix](../reports/public_conjoint_audit_20261109/existence_candidate_derivation.md).

## 5. Released vaccine choices

The public-data audit fixes the 1200-row attribute pool, empirical human-projection target, label allocations and nuisance predictions within each comparison. It uses 500 allocations and label budgets of 50, 100, 150 and 200 choice rows. Five-row source-defined blocks are preserved, without claiming verified respondent identifiers. A separate audit compares the released reference vector with a converged fit to all 4605 human choice rows. The reference matches a 200-step Adam path but can lower its total NLL by 7.013964 under continued converged fitting. This is a convergence result, without attributing all published gains to that discrepancy.

The pooled-imputation comparison includes six matched methods. At 50 labelled choice rows, the AI-corrected objective has 5 exterior moments and its X-only counterpart has 2. Every exterior case has an independently checked positive-gap recession ray. Error comparisons retain the strong clipped-soft baseline and condition explicitly on shared available estimates. They do not count unavailable roots as successful fits.

The auxiliary-only PPI experiment separately evaluates unit power and two initializers for established power tuning. At 50 labelled rows, unit-power AI correction has 6 exterior moments and X-only correction has 2. Human-root initialization returns all 500 tuned estimates for each predictor; PPI-root initialization remains unavailable in the corresponding 6 and 2 cases. These are different objectives and retain separate counts. Full tables and definitions are in [the public-data report](../reports/public_conjoint_audit_20261109/公开数据重分析与创新进展.md) and [the power-tuning report](../reports/public_conjoint_audit_20261109/权重对照与信息代理进展.md).

## 6. Known-truth validation on the released attribute support

A separate simulation samples independent rows with replacement from the released 1200-row support. The converged 11-coordinate fit on that support is fixed as a simulation parameter. New binary choices follow its logistic probabilities; proxies independently flip those choices at ε=0.1 or 0.25. Frozen oracle soft labels use the correct conditional mean given attributes and proxy. This is a known-truth simulation, without new human recruitment or generated LLM responses.

The prespecified grid uses m=20,50,150 and N=150,600, with 500 repetitions per condition. Seven methods share every generated sample: human MLE, unit-power PPI, human-initialized power tuning, ridge-initialized power tuning, one-step PPI from the same ridge start, unit-power ridge PPI, and oracle soft pooling. The ridge penalty is explicit at ρ=1/m on every coordinate. The pooling comparator has extra correct-conditional-mean information and remains a strong best-case baseline.

The table fixes N=600 and ε=0.25. Intervals are nominal 95% marginal Wald intervals, averaged across 11 prespecified coordinates. Conditional coverage uses issued intervals; issued-and-covers uses all 500 repetitions. These are distinct summaries. Ridge intervals do not correct penalization bias.

| Label rows | Estimator | Available estimates | Conditional coverage | Issued and covers | Parameter MSE |
|---:|---|---:|---:|---:|---:|
| 20 | Human MLE | 116/500 | 0.926 | 0.215 | 8.5602 |
| 20 | Unit-power PPI | 412/500 | 0.998 | 0.822 | 4.8362 |
| 20 | Power, human start | 116/500 | 0.907 | 0.210 | 6.0858 |
| 20 | Power, ridge start | 348/500 | 0.801 | 0.558 | 21.5408 |
| 20 | One step, ridge start | 500/500 | 0.827 | 0.827 | 1.5054 |
| 20 | Ridge PPI | 500/500 | 0.961 | 0.961 | 0.2193 |
| 20 | Oracle soft pooling | 500/500 | 0.942 | 0.942 | 0.0057 |
| 150 | Human MLE | 500/500 | 0.941 | 0.941 | 0.1076 |
| 150 | Unit-power PPI | 500/500 | 0.963 | 0.963 | 0.0816 |
| 150 | Power, human start | 500/500 | 0.945 | 0.945 | 0.0806 |
| 150 | Power, ridge start | 500/500 | 0.946 | 0.946 | 0.0801 |
| 150 | One step, ridge start | 500/500 | 0.947 | 0.947 | 0.0786 |
| 150 | Ridge PPI | 500/500 | 0.960 | 0.960 | 0.0624 |
| 150 | Oracle soft pooling | 500/500 | 0.943 | 0.943 | 0.0069 |

The low-budget one-step estimator always returns a finite value in this condition, while its average interval coverage is about 83%. The unit-power fit's conditional coverage of about 100% occurs alongside 88 unavailable estimates. At 150 labels all methods return finite estimates, and the one-step coverage is much closer to nominal. Finite-root existence and interval reliability require separate evaluation.

The oracle pooling advantage is retained. It relies on a correctly specified conditional mean; the current simulation cannot establish its robustness to an unknown biased predictor. Ridge performs well under the frozen coefficient vector, which is relatively close to zero. Its bias and coverage under stronger utility signals remain open. Heavy-tailed small-budget errors are accompanied by medians and 90th percentiles. All conditions, coordinate coverage, paired comparisons and Monte Carlo errors are in [the finite-pool report](../reports/public_conjoint_audit_20261109/有限样本验证与理论扩展.md).

## 7. Numerical validation of the theoretical calculations

The finite-ratio experiment completes 80 prespecified conditions with 500 Gaussian feature/choice/proxy draws each. It varies m=40,80, τ=1,4, ε=0.1,0.25 and unit versus population variance-optimal power, at five dimension ratios around each candidate boundary. There are no numerical-unresolved LP cases. The finite-m probability curves sharpen around the predicted transitions. A separate 100-draw comparison agrees between the original weighted-moment LP and oriented interval formulation in all 43 interior and 57 exterior cases.

Independent quadrature checks the Gaussian tail function and unit-power reduction within 8.9e-16. A large-τ check approaches the population limit. Earlier population-auxiliary experiments cover 160 exact-probability and 480 signal/predictor conditions, each with 2000 draws. The informative-proxy extension separately covers 135 exact-probability and 162 signal conditions. These calculations test the stated models; they do not certify originality or theorems for arbitrary discrete designs.

The finite-ratio signal experiment separately completes 80 conditions with 300 draws each, including a free intercept. All 120 original-moment/interval LP checks agree, and 12 direct cone projections agree with the polar saddle expression within 3.8e-14 after an independent active-face refinement. Population quadrature at 100 and 200 nodes agrees within 7.9e-13. The phase experiment retains every draw and has no unresolved numerical cases. Finite-m curves have smooth transitions and finite-sample shifts; no exact finite-m step is claimed. See [the signal report](../reports/public_conjoint_audit_20261109/非零效用信号与有限样本边界.md).

## 8. Scope and remaining work

The results concern a specified unpenalized corrected binary-choice estimator. They give no impossibility bound for every prediction-assisted estimator. General cone geometry, ordinary-logit separation, power tuning, one-step estimation and penalized coercivity are established prior work. The remaining originality claim concerns the explicit estimator-specific finite-estimability law and a useful analysis decision supported by it.

The 300-record Chinese service-choice analysis remains an exploratory human application. Sixty AI pretests remain separate. Recruitment channel is confounded with questionnaire version, and several attributes vary as a bundle. Its findings cannot validate the Gaussian theorem or identify a randomized version effect. It belongs in a future submission only if its analysis question connects directly to the final contribution.

The next scientific work is to review the conditional proof and nearest equivalent prior results, evaluate imperfect frozen predictors, and assess a useful budget/estimator decision under realistic choice designs. Finite-sample PPI++ costs already have an exact mean-estimation literature, including Mani et al.2026; the candidate claim is the specified logistic-root existence law. The final manuscript, appendices, journal-format package and accurate research declarations remain incomplete. The present draft makes no submission-readiness claim.
