# Counterfactual Policy Path Sensitivity Audit in Discrete Choice Models

A controlled simulation benchmark with an external Swissmetro DCE check

Working manuscript draft

Abstract. Choice models are often used to evaluate attribute changes that were not observed in the estimation tasks. Two models can produce similar observed-task scores while implying different demand responses to such changes. This paper develops a counterfactual policy-path sensitivity audit for discrete choice models. The audit compares an additive MNL and a structured alternative on a finite, feasible path of predeclared interventions, then measures each model against the known oracle in simulation. Across 24 respondent-grouped replications per mechanism, the model gap was large for nonlinear and interaction departures, while random taste heterogeneity produced a small model gap and a modest population oracle gap. An exact finite-support witness shows why model disagreement cannot certify counterfactual accuracy: both models can agree on observed support and share an omitted policy term. A descriptive Swissmetro audit on 6,768 observations from 752 respondents produced a mean absolute policy-path probability gap of 0.0647. The resulting contribution is a calibrated sensitivity audit with an explicit common-mode failure test, not a global recoverability theorem, welfare bound, or causal mechanism test.

## 1 Introduction

Discrete choice models are used to predict market shares, welfare, and policy responses. Those uses require predictions at attribute combinations that may lie outside the exact task profiles used for estimation. A model comparison based only on observed-task loss can miss this problem. Two models can rank the same observed choices similarly while producing different responses to a price, quality, or service change.

This paper develops a counterfactual policy-path sensitivity audit. The audit compares an additive multinomial logit model with a structured model over a finite path of feasible interventions defined before outcomes are evaluated. The path is part of the estimand and encodes the policy question that the choice model will answer. A direct oracle benchmark measures the gap between each fitted model and the data-generating policy surface, while an exact common-mode witness states the support condition under which model agreement is uninformative.

A second diagnostic separates observable utility-form departures from respondent-level variation. After the structured model is refitted, we calculate the dispersion of respondent score contributions relative to the additive calibration distribution. A policy alert without a score alert supports an observable structure departure. A score alert without a policy alert supports persistent heterogeneity. Both signals produce an unresolved status because the available data do not identify a unique mechanism.

The study makes four contributions. First, it defines a pre-outcome counterfactual disagreement estimand for DCE model diagnostics. Second, it reports a direct oracle benchmark that separates between-model sensitivity from counterfactual error. Third, it gives an exact finite-support common-mode witness and a computable support limitation. Fourth, it audits external counterfactual sensitivity in the public Swissmetro DCE. The claim is deliberately narrow: the audit measures sensitivity of a chosen counterfactual surface and exposes a failure mode of disagreement-based certificates; it does not provide a welfare bound, a new random-coefficients estimator, or a causal interpretation of an alert.

## 2 Relation to choice-model specification

Recent choice-modelling work has compared flexible machine-learning predictors with discrete-choice models, tested structural heterogeneity, and quantified the welfare effect of model uncertainty. Those studies motivate the present problem and define its boundary. Flexible prediction addresses fit. Welfare-sensitivity analysis addresses the consequences of selecting a model. The present certificate sits between these tasks: it asks whether two plausible models disagree along the intervention path that will support a policy calculation.

The audit differs from a conventional lack-of-fit test in its target. A standard test asks whether a model family can reproduce observed choices under a specified alternative. The policy-path statistic asks whether the chosen model family is recoverable for a declared counterfactual use. The distinction matters when the observed support is narrow or when the analyst plans to extrapolate a policy response.

The external check is descriptive. Swissmetro provides a public DCE with repeated choices from the same respondents and allows a respondent-grouped split. The public data do not reveal a mechanism label or a randomized policy intervention. The Swissmetro result is used to show that the certificate can expose counterfactual sensitivity in a real DCE, not to attribute that sensitivity to a particular behavioural process.


## 3 Counterfactual policy-path sensitivity audit


## 3.1 Estimand

Let Xᵢ denote the observed choice tasks for respondent i. Let m₀ be the additive MNL and mₛ be a structured model that includes the declared observable terms. Let 𝒫 be a finite set of feasible interventions. For each p∈𝒫, transform the product attributes in Xᵢ to obtain Xᵢᵖ while preserving the choice-set structure and the opt-out alternative. Define

Dₚ(m₀,mₛ) = (1/(|𝒫| N J)) Σₚ∈𝒫 Σᵢ Σⱼ | P̂ₘ₀ⱼ(·|Xᵢᵖ) − P̂ₘₛⱼ(·|Xᵢᵖ) |.

The audit uses Dₚ as a policy-surface disagreement statistic. The path is fixed before test outcomes are used. A high value means that model choice changes predicted choice probabilities over feasible interventions. It does not mean either model is close to the data-generating policy surface. A low value means that the two models imply similar policy responses on the declared path; a shared omitted term can still make both wrong.


## 3.2 Exploratory respondent score gate

The policy signal does not distinguish a missing observable term from unobserved taste variation. We refit mₛ on the development and validation respondents, calculate respondent-level score contributions on the held-out fold, and form a score-overdispersion ratio relative to the additive null calibration. The ratio is denoted Q. The joint rule uses the 99th percentile of Dₚ and the 97.5th percentile of Q. The calibration is repeated at each sample-size setting.



| Policy signal | Score signal | Reported status |

| --- | --- | --- |

| No | No | Base |

| Yes | No | Observable structure |

| No | Yes | Heterogeneity |

| Yes | Yes | Unresolved |



The unresolved branch is part of the method. The audit reports evidence about the counterfactual surface and score dispersion; it does not force a behavioural label when both signals are present.


## 4 Controlled simulation


## 4.1 Data-generating process

The simulation represents a three-alternative smart medical-device DCE with two products and an opt-out alternative. Each product has quality and price attributes used in the policy path, plus additional device attributes used by the base utility. A respondent completes 12 tasks. Each replication contains 400 respondents. All splits are respondent-grouped, so choices from one respondent never appear in more than one split.

The additive condition uses a linear systematic utility. The nonlinear condition adds smooth quality and price curvature. The threshold condition adds a price hinge at 1.25. The interaction condition adds quality-by-support and evidence-by-digital-experience terms. The heterogeneity condition adds a respondent-specific random price coefficient. The combined condition includes observable nonlinear structure and respondent-level variation. The declared structured model includes the observable nonlinear and interaction terms used by the corresponding mechanism; the heterogeneity component remains outside its fixed-effect library.


## 4.2 Estimation and evaluation

The corrected oracle audit uses a respondent-grouped 60% estimation fold and a 20% validation fold. The models are fitted on the estimation fold and evaluated on the fixed path for the validation respondents. The data-generating utility is retained only in the simulation oracle; it is not used to fit either model. The held-out 20% fold is reserved for extensions and is not used to define the reported gap.

The direct oracle audit uses 24 independent replications per condition and the nine quality and price shifts in {−0.6, 0, 0.6}². It reports the model gap, each model’s oracle error, and anchored response error. Alert thresholds and score-gate operating characteristics are reserved for a subsequent corrected calibration because the earlier alert table used an incomplete opt-out library.


**Table 1. Mechanism grid and diagnostic target.**



| Condition | Departure | Expected primary signal |

| --- | --- | --- |

| Additive | None | No signal |

| Nonlinear | Observable curvature | Policy |

| Threshold | Observable boundary | Policy |

| Interaction | Quality × price | Policy |

| Heterogeneity | Random price sensitivity | Score |

| Nonlinear + threshold | Two observable departures | Policy |

| Nonlinear + interaction | Two observable departures | Policy |

| Combined | Observable structure + heterogeneity | Both or unresolved |




## 5 Results


## 5.1 Direct oracle audit

The corrected benchmark adds the opt-out-by-income term to both behavioural libraries and fits each MNL with a converged L-BFGS estimator. The oracle probabilities use the exact data-generating utility, including the respondent-specific price coefficient when present. Table 2 reports the mean absolute model gap (D), additive-model oracle error (E0), structured-model oracle error (ES), and anchored response errors (R0 and RS) over 24 replications.

**Table 2. Model sensitivity and oracle error.**

| Condition | D | E0 | ES | R0 | RS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Additive | 0.0104 | 0.0116 | 0.0159 | 0.0042 | 0.0085 |
| Nonlinear | 0.0982 | 0.0987 | 0.0139 | 0.0632 | 0.0082 |
| Threshold | 0.0186 | 0.0205 | 0.0152 | 0.0177 | 0.0083 |
| Interaction | 0.0902 | 0.0907 | 0.0151 | 0.0387 | 0.0079 |
| Heterogeneity | 0.0098 | 0.0142 | 0.0171 | 0.0061 | 0.0084 |
| Nonlinear + threshold | 0.0954 | 0.0960 | 0.0136 | 0.0650 | 0.0089 |
| Nonlinear + interaction | 0.1278 | 0.1284 | 0.0135 | 0.0724 | 0.0085 |
| Combined | 0.1219 | 0.1222 | 0.0144 | 0.0717 | 0.0094 |

A large model gap tracks observable curvature and interactions in this generator. Heterogeneity produces a small model gap and only a modest population oracle gap after integrating the random coefficient, so D is a sensitivity measure between declared specifications rather than an accuracy certificate. The pooled correlation between D and additive oracle error is 0.996, while its correlation with the smaller of the two oracle errors is 0.069. The corrected results supersede the earlier decision-regret table, which used an incomplete opt-out library.

## 5.2 Common-mode failure test

For observed support q∈{−1,+1}, let the omitted policy term be h(q)=q²−1. It is zero at every observed support point, so two fitted models can agree exactly while the oracle changes at q=−0.5, 0, and 0.5. The saved witness has D=0 and oracle terms −0.75, −1, and −0.75. The result is an exact finite-support counterexample to treating model agreement as a recoverability certificate.

## 5.3 Sample-size calibration

A preliminary small calibration produced an unstable high quantile at n=800. We increased the independent additive calibration to 200 replications for that setting. The corrected n=800 run produced zero additive policy alerts, full policy detection for nonlinear, threshold, and interaction mechanisms, and full score detection for isolated heterogeneity. The result supports sample-size-specific calibration; it does not justify transporting one threshold across designs.


**Table 3. Corrected n=800 calibration and evaluation.**



| Condition | Policy alert | Score alert | Unresolved | Replications |

| --- | --- | --- | --- | --- |

| Additive | 0.000 | 0.000 | 0.000 | 24 |

| Nonlinear | 1.000 | 0.000 | 0.000 | 24 |

| Threshold | 1.000 | 0.000 | 0.000 | 24 |

| Interaction | 1.000 | 0.000 | 0.000 | 24 |

| Heterogeneity | 0.000 | 1.000 | 0.000 | 24 |




## 5.4 Swissmetro external audit

The external audit uses the canonical Swissmetro purpose-1/3 data with 6,768 observations from 752 respondents. We retain respondent grouping and fit an additive MNL and a structured alternative-specific time and cost model. The structured model reduces held-out log loss from 0.7982 to 0.7658. Across a predeclared multiplicative time and cost path with factors {0.8, 1.0, 1.2}², the mean absolute choice-probability gap is 0.0647 and the mean predicted-share L1 gap is 0.0186. The largest path gap is 0.0757.


**Table 4. Swissmetro external policy-path audit.**



| Rows | Respondents | Base test log loss | Structured test log loss | Mean probability gap | Mean share gap | Max probability gap |

| --- | --- | --- | --- | --- | --- | --- |

| 6,768 | 752 | 0.7982 | 0.7658 | 0.0647 | 0.0186 | 0.0757 |



The Swissmetro result establishes external counterfactual sensitivity. The public file does not provide randomized intervention assignment or a ground-truth mechanism. The result is a transfer audit, not a causal test.


## 6 Discussion

The audit changes the diagnostic question from “which model predicts the observed tasks better?” to “does model choice change the declared counterfactual response, and how can shared misspecification be exposed?” This target is aligned with the downstream use of a DCE. It also creates a clear operating boundary. A policy-path alert supports model expansion for the observable utility component. A score alert supports a heterogeneity branch. Joint alerts require additional evidence because both mechanisms can generate individual-level instability.

The statistic is a finite-path object. A path that omits a relevant intervention can produce a false sense of recoverability. The paper addresses this by requiring the path to be declared before evaluation and by reporting the exact path. The audit is relative to the declared candidate models and intervention set. It does not certify global equivalence across the full attribute space.

The threshold condition has a smaller model gap than the curvature and interaction conditions. Its oracle error remains visible in the direct audit, which motivates a corrected power analysis before any threshold-based alert is reported.

The earlier candidate-preserving fibre statistics remain useful negative controls and design audits. On finite discrete support, a support-complete version has the same tested alternative as a saturated profile likelihood ratio. It is not presented as the main innovation. The policy-path audit avoids that collision by making the estimand the downstream counterfactual demand surface.


## 7 Limitations and reporting requirements

- The simulation uses a smart-device setting and a finite mechanism grid. The audit should be recalibrated for the attribute ranges, sample size, task count, and model library of each application.

- The Swissmetro audit is observational and descriptive. It cannot identify whether the observed disagreement reflects nonlinearity, heterogeneity, scale differences, or another process.

- The current score gate detects persistent score dispersion. It does not separate random taste sensitivity from attribute non-attendance without additional design information.

- Policy-path disagreement is not a welfare estimate. A welfare analysis must specify the policy, the population, the integration measure, and the compensating-variation convention separately.

- The final submission should report the calibration replication count, path definition, model library, respondent split, threshold construction, and all unresolved cases.


## 8 Conclusion

This paper proposes a counterfactual policy-path sensitivity audit for discrete choice models. The audit measures disagreement between additive and structured models on a feasible intervention path and reports its direct oracle error in simulation. A respondent-score gate is retained as an exploratory extension. Controlled simulations show separation between observable structure and isolated heterogeneity, with an explicit common-mode failure branch. The Swissmetro audit shows that the same comparison can reveal counterfactual sensitivity in a public DCE. The evidence supports a focused methodological claim: choice-model diagnostics should be evaluated on the counterfactual surface they are used to predict, with calibration tied to the design and with uncertainty reported when the mechanism is not recoverable.


## Appendix A Reproducibility

All source code, calibration outputs, and public-data audit files are versioned at https://github.com/StarFall20/paper. The primary entry points are analysis/policy_path_triage.py, analysis/policy_path_disagreement.py, analysis/policy_path_power_curve.py, and analysis/swissmetro_policy_path_audit.py, and analysis/policy_oracle_audit.py. The main manuscript note is manuscript/policy_path_recoverability_round29.md. Raw public data are not redistributed; the repository records provenance and the retrieval path.

The current locked oracle evaluation uses 24 replications per mechanism, a respondent-grouped 60% estimation fold and 20% validation fold, converged MNL fits, and a known simulation oracle. Output CSV files include row-level model gaps, oracle errors, anchored response errors, and the exact common-mode witness.


## References

Train, K. E. (2009). Discrete Choice Methods with Simulation. 2nd ed. Cambridge University Press.

McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior. In Frontiers in Econometrics, 105–142.

Zhu, W., and Si, W. (2024). Predicting choices of street-view images: A comparison between discrete choice models and machine learning models. Journal of Choice Modelling, 50, 100470. https://doi.org/10.1016/j.jocm.2024.100470

Yang, K. Z., and Vasserman, S. (2025). Robustness Measures for Welfare Analysis. American Economic Review. https://doi.org/10.1257/aer.20220673

Goeken, N., Kurz, P., and Steiner, W. J. (2024). Multimodal preference heterogeneity in choice-based conjoint analysis: A simulation study. Journal of Business Economics, 94, 137–185. https://doi.org/10.1007/s11573-023-01156-6

Wang, S., Mo, B., Zheng, Y., Hess, S., and Zhao, J. (2024). Comparing hundreds of machine learning and discrete choice models for travel demand modeling. Transportation Research Part B, 190, 103061.

Fok, D., and Paap, R. (2025). New misspecification tests for multinomial logit models. Journal of Choice Modelling. https://doi.org/10.1016/j.jocm.2024.100531

ISPOR Conjoint Analysis Good Research Practices Task Force. (2013). Constructing experimental designs for discrete-choice experiments. Value in Health, 16, 3–13.