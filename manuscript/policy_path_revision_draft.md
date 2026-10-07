# Counterfactual Policy Path Recoverability in Discrete Choice Models

A controlled simulation benchmark with an external Swissmetro DCE check

Adkid-Zephyr and Anti-Defensive Writing

Abstract. Choice models are often used to evaluate attribute changes that were not observed in the estimation tasks. Two models can produce similar observed-task scores while implying different demand responses to such changes. This paper defines a counterfactual policy-path recoverability certificate for discrete choice models. The certificate evaluates an additive MNL and a structured alternative on a finite, feasible path of predeclared attribute interventions. The statistic is the average L1 distance between their predicted choice-probability vectors along that path. A respondent-score overdispersion gate is estimated after refitting the structured model and provides a separate signal for persistent individual variation. The resulting decision rule reports base, observable-structure, heterogeneity, or unresolved status. In 24 respondent-grouped replications per mechanism, the additive null produced no policy alerts. Nonlinear and interaction mechanisms were detected in all replications, threshold mechanisms in 75%, and isolated continuous heterogeneity produced a score alert in 95.8% of replications with a 4.2% policy false-alert rate. The combined mechanism produced an unresolved decision in 62.5% of replications. A corrected n=800 calibration run separated all five tested mechanisms at the reported thresholds. An external audit on 6,768 Swissmetro observations from 752 respondents produced a mean policy-path probability gap of 0.0647 between additive and structured models. The certificate is a diagnostic for counterfactual sensitivity; it does not identify a unique behavioural mechanism or replace welfare analysis.

Keywords: discrete choice; model misspecification; counterfactual demand; policy sensitivity; model uncertainty; simulation; Swissmetro


## 1 Introduction

Discrete choice models are used to predict market shares, welfare, and policy responses. Those uses require predictions at attribute combinations that may lie outside the exact task profiles used for estimation. A model comparison based only on observed-task loss can miss this problem. Two models can rank the same observed choices similarly while producing different responses to a price, quality, or service change.

This paper develops a counterfactual policy-path recoverability certificate. The certificate compares an additive multinomial logit model with a structured model over a finite path of feasible interventions defined before outcomes are evaluated. The path is part of the estimand. It encodes the policy question that the choice model will answer. The statistic records whether model choice changes the predicted demand surface over that path.

A second diagnostic separates observable utility-form departures from respondent-level variation. After the structured model is refitted, we calculate the dispersion of respondent score contributions relative to the additive calibration distribution. A policy alert without a score alert supports an observable structure departure. A score alert without a policy alert supports persistent heterogeneity. Both signals produce an unresolved status because the available data do not identify a unique mechanism.

The study makes four contributions. First, it defines a pre-outcome counterfactual disagreement estimand for DCE model diagnostics. Second, it calibrates a finite-sample decision rule under respondent-grouped splits and an additive null. Third, it tests the rule across nonlinear, threshold, interaction, heterogeneity, and combined mechanisms. Fourth, it audits external counterfactual sensitivity in the public Swissmetro DCE. The claim is deliberately narrow: the certificate measures recoverability of a chosen counterfactual surface; it does not provide a welfare bound, a new random-coefficients estimator, or a causal interpretation of an alert.


## 2 Relation to choice-model specification

Recent choice-modelling work has compared flexible machine-learning predictors with discrete-choice models, tested structural heterogeneity, and quantified the welfare effect of model uncertainty. Those studies motivate the present problem and define its boundary. Flexible prediction addresses fit. Welfare-sensitivity analysis addresses the consequences of selecting a model. The present certificate sits between these tasks: it asks whether two plausible models disagree along the intervention path that will support a policy calculation.

The certificate differs from a conventional lack-of-fit test in its target. A standard test asks whether a model family can reproduce observed choices under a specified alternative. The policy-path statistic asks whether the chosen model family is recoverable for a declared counterfactual use. The distinction matters when the observed support is narrow or when the analyst plans to extrapolate a policy response.

The external check is descriptive. Swissmetro provides a public DCE with repeated choices from the same respondents and allows a respondent-grouped split. The public data do not reveal a mechanism label or a randomized policy intervention. The Swissmetro result is used to show that the certificate can expose counterfactual sensitivity in a real DCE, not to attribute that sensitivity to a particular behavioural process.


## 3 Counterfactual policy-path certificate


## 3.1 Estimand

Let Xᵢ denote the observed choice tasks for respondent i. Let m₀ be the additive MNL and mₛ be a structured model that includes the declared observable terms. Let 𝒫 be a finite set of feasible interventions. For each p∈𝒫, transform the product attributes in Xᵢ to obtain Xᵢᵖ while preserving the choice-set structure and the opt-out alternative. Define

Dₚ(m₀,mₛ) = (1/|𝒫|) Σₚ∈𝒫 (1/N) Σᵢ || P̂ₘ₀(·|Xᵢᵖ) − P̂ₘₛ(·|Xᵢᵖ) ||₁.

The certificate uses Dₚ as a policy-surface disagreement statistic. The path is fixed before test outcomes are used. A high value means that model choice changes predicted choice probabilities over feasible interventions. A low value means that the two models imply similar policy responses on the declared path, even if other counterfactuals remain unexamined.


## 3.2 Respondent score gate

The policy signal does not distinguish a missing observable term from unobserved taste variation. We refit mₛ on the development and validation respondents, calculate respondent-level score contributions on the held-out fold, and form a score-overdispersion ratio relative to the additive null calibration. The ratio is denoted Q. The joint rule uses the 99th percentile of Dₚ and the 97.5th percentile of Q. The calibration is repeated at each sample-size setting.



| Policy signal | Score signal | Reported status |

| --- | --- | --- |

| No | No | Base |

| Yes | No | Observable structure |

| No | Yes | Heterogeneity |

| Yes | Yes | Unresolved |



The unresolved branch is part of the method. The certificate reports evidence about the counterfactual surface and score dispersion; it does not force a behavioural label when both signals are present.


## 4 Controlled simulation


## 4.1 Data-generating process

The simulation represents a three-alternative smart medical-device DCE with two products and an opt-out alternative. Each product has quality and price attributes used in the policy path, plus additional device attributes used by the base utility. A respondent completes 12 tasks. Each replication contains 400 respondents. All splits are respondent-grouped, so choices from one respondent never appear in more than one split.

The additive condition uses a linear systematic utility. The nonlinear condition adds smooth quality and price curvature. The threshold condition adds a step response to a quality boundary. The interaction condition adds a quality-by-price term. The heterogeneity condition adds a respondent-specific random price coefficient. The combined condition includes observable nonlinear structure and respondent-level variation. The declared structured model includes the observable nonlinear and interaction terms used by the corresponding mechanism; the heterogeneity component remains outside its fixed-effect library.


## 4.2 Estimation and evaluation

The development fold is used to estimate the additive and structured models and to choose the diagnostics. The validation fold is used to compute the policy-path gap and the score gate. The structured model is then refitted on development plus validation respondents before the held-out test evaluation. This separation prevents test outcomes from defining the alert.

The main simulation uses 100 independent additive calibration replications and 24 evaluation replications per condition. The path contains the nine quality and price shifts in {−0.6, 0, 0.6}². The policy threshold is the 99th percentile of the additive calibration distribution. The score threshold is the 97.5th percentile of the corresponding calibrated score distribution. The evaluation reports policy-signal rate, score-signal rate, action accuracy, unresolved rate, path gap, and test decision regret.


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


## 5.1 Main triage result

Table 2 reports the calibrated decision rule. The additive null generated zero policy alerts and zero score alerts in the 24-run evaluation. Nonlinear and interaction mechanisms produced policy alerts in all replications. Threshold effects were detected in 75% of replications, which marks the current boundary of sensitivity for a weaker local departure. Isolated heterogeneity produced a score alert in 95.8% of replications and a policy false alert in one replication. The combined condition generated both signals in 62.5% of replications and was correctly treated as unresolved in those runs.


**Table 2. Policy-path triage under the controlled simulation.**



| Condition | Policy alert | Score alert | Action accuracy | Unresolved | Dₚ | Test regret |

| --- | --- | --- | --- | --- | --- | --- |

| Additive | 0.000 | 0.000 | 1.000 | 0.000 | 0.0173 | 0.0050 |

| Nonlinear | 1.000 | 0.000 | 1.000 | 0.000 | 0.1039 | 0.0026 |

| Threshold | 0.750 | 0.000 | 0.750 | 0.000 | 0.0267 | 0.0035 |

| Interaction | 1.000 | 0.042 | 0.958 | 0.042 | 0.0908 | 0.0054 |

| Heterogeneity | 0.042 | 0.958 | 0.917 | 0.042 | 0.0167 | 0.0772 |

| Nonlinear + threshold | 1.000 | 0.000 | 1.000 | 0.000 | 0.1021 | 0.0017 |

| Nonlinear + interaction | 1.000 | 0.000 | 1.000 | 0.000 | 0.1330 | 0.0024 |

| Combined | 1.000 | 0.625 | 0.625 | 0.625 | 0.1256 | 0.0864 |



The path gap tracks the direction of model regret across mechanism conditions. The pooled correlation between Dₚ and the base-minus-structured test decision regret is 0.95. This is a mechanism-level association. Within one condition, Dₚ is a diagnostic quantity and should not be interpreted as a calibrated point prediction of regret.


## 5.2 Sample-size calibration

A preliminary small calibration produced an unstable high quantile at n=800. We increased the independent additive calibration to 200 replications for that setting. The corrected n=800 run produced zero additive policy alerts, full policy detection for nonlinear, threshold, and interaction mechanisms, and full score detection for isolated heterogeneity. The result supports sample-size-specific calibration; it does not justify transporting one threshold across designs.


**Table 3. Corrected n=800 calibration and evaluation.**



| Condition | Policy alert | Score alert | Unresolved | Replications |

| --- | --- | --- | --- | --- |

| Additive | 0.000 | 0.000 | 0.000 | 24 |

| Nonlinear | 1.000 | 0.000 | 0.000 | 24 |

| Threshold | 1.000 | 0.000 | 0.000 | 24 |

| Interaction | 1.000 | 0.000 | 0.000 | 24 |

| Heterogeneity | 0.000 | 1.000 | 0.000 | 24 |




## 5.3 Swissmetro external audit

The external audit uses the canonical Swissmetro purpose-1/3 data with 6,768 observations from 752 respondents. We retain respondent grouping and fit an additive MNL and a structured alternative-specific time and cost model. The structured model reduces held-out log loss from 0.7982 to 0.7658. Across a predeclared multiplicative time and cost path with factors {0.8, 1.0, 1.2}², the mean choice-probability L1 gap is 0.0647 and the mean predicted-share L1 gap is 0.0186. The largest path gap is 0.0757.


**Table 4. Swissmetro external policy-path audit.**



| Rows | Respondents | Base test log loss | Structured test log loss | Mean path L1 | Mean share L1 | Max path L1 |

| --- | --- | --- | --- | --- | --- | --- |

| 6,768 | 752 | 0.7982 | 0.7658 | 0.0647 | 0.0186 | 0.0757 |



The Swissmetro result establishes external counterfactual sensitivity. The public file does not provide randomized intervention assignment or a ground-truth mechanism. The result is a transfer audit, not a causal test.


## 6 Discussion

The certificate changes the diagnostic question from “which model predicts the observed tasks better?” to “does model choice change the declared counterfactual response?” This target is aligned with the downstream use of a DCE. It also creates a clear operating boundary. A policy-path alert supports model expansion for the observable utility component. A score alert supports a heterogeneity branch. Joint alerts require additional evidence because both mechanisms can generate individual-level instability.

The statistic is a finite-path object. A path that omits a relevant intervention can produce a false sense of recoverability. The paper addresses this by requiring the path to be declared before evaluation and by reporting the exact path. The certificate is relative to the declared candidate models and intervention set. It does not certify global equivalence across the full attribute space.

The threshold condition is the main weak boundary in the current benchmark. Its detection rate of 75% shows that a small or localized departure can be difficult to recover at 400 respondents and 12 tasks. This result is substantive: a negative or unstable certificate result should trigger a power analysis or an expanded design, not a claim that the additive model is correct.

The earlier candidate-preserving fibre statistics remain useful negative controls and design audits. On finite discrete support, a support-complete version has the same tested alternative as a saturated profile likelihood ratio. It is not presented as the main innovation. The policy-path certificate avoids that collision by making the estimand the downstream counterfactual demand surface.


## 7 Limitations and reporting requirements

- The simulation uses a smart-device setting and a finite mechanism grid. The certificate should be recalibrated for the attribute ranges, sample size, task count, and model library of each application.

- The Swissmetro audit is observational and descriptive. It cannot identify whether the observed disagreement reflects nonlinearity, heterogeneity, scale differences, or another process.

- The current score gate detects persistent score dispersion. It does not separate random taste sensitivity from attribute non-attendance without additional design information.

- Policy-path disagreement is not a welfare estimate. A welfare analysis must specify the policy, the population, the integration measure, and the compensating-variation convention separately.

- The final submission should report the calibration replication count, path definition, model library, respondent split, threshold construction, and all unresolved cases.


## 8 Conclusion

This paper proposes a counterfactual policy-path recoverability certificate for discrete choice models. The certificate measures disagreement between additive and structured models on a feasible intervention path, then combines that signal with a calibrated respondent-score gate. Controlled simulations show separation between observable structure and isolated heterogeneity, with an explicit unresolved branch for combined departures. The Swissmetro audit shows that the same comparison can reveal counterfactual sensitivity in a public DCE. The evidence supports a focused methodological claim: choice-model diagnostics should be evaluated on the counterfactual surface they are used to predict, with calibration tied to the design and with uncertainty reported when the mechanism is not recoverable.


## Appendix A Reproducibility

All source code, calibration outputs, and public-data audit files are versioned at https://github.com/StarFall20/paper. The primary entry points are analysis/policy_path_triage.py, analysis/policy_path_disagreement.py, analysis/policy_path_power_curve.py, and analysis/swissmetro_policy_path_audit.py. The main manuscript note is manuscript/policy_path_recoverability_round29.md. Raw public data are not redistributed; the repository records provenance and the retrieval path.

The current locked evaluation uses 24 replications per mechanism, respondent-grouped development, validation, and test folds, 100 additive calibration replications for the n=400 setting, and 200 additive calibration replications for the corrected n=800 setting. Output CSV files include the row-level path gaps and the summary-level triage decisions.


## References

Train, K. E. (2009). Discrete Choice Methods with Simulation. 2nd ed. Cambridge University Press.

McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior. In Frontiers in Econometrics, 105–142.

Zhu, W., and Si, W. (2024). Predicting choices of street-view images: A comparison between discrete choice models and machine learning models. Journal of Choice Modelling, 50, 100470. https://doi.org/10.1016/j.jocm.2024.100470

Kang, H., and Vasserman, S. (2025). Robust welfare analysis with model uncertainty. American Economic Review. https://doi.org/10.1257/aer.20220673

Goeken, N., Kurz, P., and Steiner, W. J. (2024). Multimodal preference heterogeneity in choice-based conjoint analysis: A simulation study. Journal of Business Economics, 94, 137–185. https://doi.org/10.1007/s11573-023-01156-6

Wang, S., Mo, B., Zheng, Y., Hess, S., and Zhao, J. (2024). Comparing hundreds of machine learning and discrete choice models for travel demand modeling. Transportation Research Part B, 190, 103061.

Fok, D., and Paap, R. (2025). New misspecification tests for multinomial logit models. Journal of Choice Modelling. https://doi.org/10.1016/j.jocm.2024.100531

ISPOR Conjoint Analysis Good Research Practices Task Force. (2013). Constructing experimental designs for discrete-choice experiments. Value in Health, 16, 3–13.