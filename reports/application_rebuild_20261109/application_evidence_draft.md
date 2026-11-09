# Data-sharing packages and service choice in a paid health-management questionnaire

**Exploratory evidence draft, 9 November 2026.** Numerical analyses are complete for the specifications reported here. Recruitment documentation, the applicable research/ethics statement and the final contribution assessment require author review. This is an application draft, with no claim of a new choice-model theory or confirmed submission readiness.

## Abstract

Paid health-management services combine subscription fees with arrangements for sharing user data. We examine which comparisons an existing questionnaire supports and how persistent nonpurchase and recruitment-source patterns affect their interpretation. The analysis uses 360 exported records, each containing eight choices among two service packages and an outside option. Models include version-specific package intercepts, alternative price specifications and record-level participation mixtures. Five-fold internal validation holds each record's eight answers together. The observed price support connects two sharing packages at CNY89; a no-external-sharing package has no price in common with either. A flexible-cell random-intercept model estimates a positive utility difference of 0.445 between the user-decision and deidentified-research-sharing packages at CNY89, with a record-clustered 95% interval of −0.004 to 0.894. Conditional A/B estimates also have intervals spanning zero. All 42 records choosing the outside option in every task belong to one 60-record export-source group. An exact outside-component model improves pooled joint prediction, while complex preference mixtures provide no predictive improvement in the other 300 records. The results support a qualified positive association for the user-decision package and show why source-sensitive participation analysis is needed before interpreting latent groups or monetary trade-offs.

## 1. Application question

Health-service providers choose both a subscription fee and a data-sharing arrangement. A preference comparison is useful when the fee and the service bundle have a clear interpretation. Existing app-purchase research already considers cost, privacy and usage experience, and health-data studies already examine governance arrangements and heterogeneous sharing preferences. [Xie and Or (2023)](https://mhealth.jmir.org/2023/1/e25908/), [Biasiotto et al. (2023)](https://www.jmir.org/2023/1/e47066/) and a [2026 Chinese governance DCE](https://doi.org/10.3389/fpubh.2026.1883099) provide close application comparisons.

This analysis asks a narrower question: within the support of an existing paid-service questionnaire, how does a package allowing users to decide whether and with whom to share compare with deidentified sharing for research? It also examines how the treatment of persistent nonpurchase changes the comparison. The user-decision description combines more than one feature. Its coefficient describes that package and does not isolate a pure control mechanism.

## 2. Questionnaire and analytic records

Two forms each contain 180 records and eight A/B/outside choices. Monthly fees range from CNY59 to CNY119. Sharing descriptions are: no sharing with external organizations (G0), deidentified provision to research institutions (G1), and user decisions about whether and with whom to share (G2). Other service-card attributes remain fixed within each form and alternative. Their separate effects cannot be recovered from these records; form-specific A and B intercepts represent the corresponding bundles.

Different recruitment channels administered different forms, and no assignment log was retained. Form adjustments are interpreted as version/channel associations. The researcher reports that the platform described its service as recruiting human respondents. Sixty exports carry the original source label “AI样本”; this field is retained. The label's meaning is not resolved by choice-model fit. Main analyses retain all 360 records, accompanied by a sensitivity analysis of the other 300 records. Neither repeated outside choices nor short duration is used here as an automatic exclusion rule.

Usage history and previous payment experience can describe the records. Demographic representativeness, adult eligibility, institutional approval, consent and applicable exemption decisions must follow the actual recruitment materials. They are not inferred from answer patterns or invented for this draft. The questionnaire's priority question follows the choices and is not treated as an exogenous mediator.

## 3. Estimands and analysis

### 3.1 Supported fee-rule comparisons

G0 appears only at CNY79 and CNY109. G1 appears at CNY69, CNY89 and CNY99; G2 appears at CNY59, CNY89 and CNY119. The primary estimand is the model-adjusted utility difference between G2 and G1 at CNY89. The two CNY89 descriptions are not directly opposed in a single task; the contrast is identified through cross-task utility restrictions, version-specific package intercepts and the connected choice structure.

A model with unrestricted fee categories and sharing main effects has 12 columns and rank 11. We use an equivalent, identified parameterization with seven observed fee-rule cells relative to G1 at CNY89, plus four version-specific alternative intercept terms. Its rank is 11. This represents all eight observed fee-rule cells. The support graph is a forest, so these cells cannot test general fee-rule interaction against additive unrestricted fee and rule effects. Removing either task 2 or task 5 makes the CNY89 G2–G1 contrast unidentified in this parameterization.

Linear and quadratic fee models provide functional-form comparisons. Monetary trade-offs based on a linear fee coefficient are supplementary, conditional quantities. [Ong et al. (2025)](https://doi.org/10.1186/s13561-025-00658-z) provides a close methodological comparison for discrete costs and WTP. No exact monetary premium is inferred from the observed support alone.

### 3.2 Participation and panel likelihood

Each analytic unit contributes its entire eight-answer vector. Random service-intercept models integrate a common participation propensity across that vector. The two-class participation, scale and preference models are competing representations. Class names are withheld when boundaries or near-zero curvature prevent stable interpretation.

The participation and preference mixtures approached an all-outside boundary. A documented exploratory follow-up fits a classical mixture of an ordinary logit component and a component assigning probability one to choosing outside in all eight tasks. Serial nonparticipation and restricted consideration sets have established treatments; [Campbell and Erdem](https://doi.org/10.1007/s40271-018-0324-6) is the relevant comparator. This model is used to diagnose the observed response distribution, without attributing a psychological cause to a latent class.

Conditional A/B logit uses tasks selecting either service. Under common choice slopes, a common service-propensity term cancels from these conditional odds. This restriction does not accommodate arbitrary random tastes. The 42 all-outside records contribute no A/B likelihood; the pooled conditional model has 318 contributing records, and the other-source sensitivity has 300.

### 3.3 Validation

All models use the same five record-level folds; no person's tasks are split across training and validation. Predictions for held-out records integrate population weights and do not use class posteriors inferred from their held-out answers. Joint log scores for eight-answer vectors and marginal single-task log scores have distinct targets. These folds were reused during the broader exploratory project and provide internal validation.

Analytic gradients are checked by finite differences. Random-intercept integration at 61 nodes is checked against 161 nodes. Multistart results, parameter boundaries and Hessian curvature accompany the estimates. Intervals and p-values are exploratory and conditional on the specified models; they do not provide selection-adjusted or external confirmation.

## 4. Results

### 4.1 The repeated-outside pattern is source concentrated

Forty-two records choose outside in all eight tasks. They comprise 22 of 30 source-labelled records in form 1 and 20 of 30 in form 2. The other 300 records contain no all-outside vector. This is an observed source association; it does not identify the reason for the pattern or determine whether any record is invalid.

In the pooled data, ordinary linear-price logit has a five-fold joint negative log score per choice of 1.040306. A normal random service intercept reduces this to 0.981693. The exact outside-component model yields 0.956676, essentially the same score as the 13-parameter preference mixture, 0.956676. Their marginal single-task scores remain around 1.0402. The apparent gain concerns the joint distribution of repeated responses.

The outside-component model predicts approximately 42 all-outside vectors across the held-out folds; the ordinary model predicts 0.14 and the normal random-intercept model 12.45. The preference-mixture solution has multiple parameters at their bounds and near-zero Hessian curvature. Its fit does not establish stable preference types.

Among the other 300 records, five-fold joint scores are 1.031745 for ordinary logit, 1.031899 for the random-intercept model, 1.032349 for the participation mixture and 1.033868 for the preference mixture. These within-population comparisons provide no predictive improvement from the more complex models. Scores across different populations are not treated as a common ranking.

### 4.2 The rule-package contrast is positive but model dependent

At CNY89, the pooled flexible-cell ordinary model estimates G2 minus G1 at 0.524, interval 0.124 to 0.925. Allowing a random service intercept gives 0.445, interval −0.004 to 0.894. The exact outside-component model gives 0.447, interval −0.005 to 0.899. Conditional A/B logit gives 0.733, interval −0.158 to 1.624.

In the other 300 records, the corresponding flexible-cell ordinary and random-intercept estimates are 0.503 and 0.499, with intervals 0.029 to 0.977 and 0.024 to 0.975. Conditional A/B logit gives 0.787, interval −0.184 to 1.759. Coefficients use model-specific utility scales and are not compared as direct changes in effect magnitude. Positive directions across specifications support an exploratory association; the intervals and assumptions do not support an established premium or an isolated behavioral mechanism.

## 5. Interpretation and manuscript development

The questionnaire retains useful service-choice content. Its price support identifies a specific package comparison under stated utility restrictions. Its source fields explain why a pooled mixture can appear to find a distinct participation group. These observations guide a transparent applied analysis and preserve the value of the completed fieldwork.

The existing evidence does not yet supply a distinctive, robust contribution beyond close app-purchase, data-governance, opt-out and discrete-cost studies. The submission decision requires an application claim with a demonstrated substantive delta, together with accurate recruitment and applicable research statements. Additional model complexity cannot supply the missing substantive identification. No new theoretical acronym, causal version claim or guaranteed publication claim is attached to these results.

## Reproducibility materials

The accompanying package includes the observed fee-rule support, rank/null-vector checks, saved aggregate model parameters and covariances, fold-fit diagnostics, joint and marginal prediction tables, source sensitivity and provenance statements. Original answers, identifiers, IP addresses, timestamps and individual fold assignments remain private. Re-estimation requires authorized access to the private originals; public aggregate files cannot independently verify respondent identity or reproduce private records.
