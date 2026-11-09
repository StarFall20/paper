# Data-sharing packages in a paid health-service questionnaire

*Exploratory application revision, 9 November 2026. Human primary population corrected after the supplied AI-pretest service definition.*

## Abstract

This analysis examines sharing-rule packages in 300 human-recruitment questionnaire records, each containing eight health-service choices. Sixty additional platform AI-pretest records are excluded from human estimation and retained for separate diagnostics. The observed fee support connects a user-decision sharing package and deidentified research sharing at CNY 89. A flexible-cell MNL estimates their utility difference at 0.503 (record-clustered 95% interval 0.029–0.977); a random service-intercept model gives 0.499 (0.024–0.975). Conditional A/B estimation gives 0.787 (−0.184–1.759). A linear-fee ordinary model has the best internal joint prediction among the four fixed primary specifications. The package association depends on cross-task utility restrictions and does not isolate a pure control mechanism or a monetary premium.

## Questionnaire and records

Each version contributes 150 human-recruitment records. Monthly fees range from CNY 59 to CNY 119. Sharing descriptions cover no external sharing (G0), deidentified research sharing (G1), and user decisions about whether and with whom to share (G2). Service-card attributes fixed within version and alternative are represented by version-specific package intercepts. Their separate effects cannot be recovered.

Different recruitment channels administered one version each, without a retained allocation log. Version adjustments describe channel/version associations. Ordinary submission metadata and the researcher recruitment report define the 300-record analytic set; demographic representativeness and eligibility were not independently measured. The supplied platform wording explicitly says that AI samples answer the questionnaire for preliminary research/pretesting. The 60 task-linked records are synthetic pretests, not additional human respondents. All 360 original records remain archived.

## Estimation

The four fixed specifications use linear fees or unrestricted observed fee-rule cells, with or without a normal random service intercept. Each person's entire eight-answer vector enters the panel likelihood and stays in one of the existing five folds. Predictions integrate the population distribution without held-out class posteriors. These reused folds provide exploratory internal validation.

G0 occurs only at fees 79 and 109, G1 at 69, 89, 99, and G2 at 59, 89, 119. G1 and G2 share CNY 89; G0 shares no fee with either. The cell representation has rank 11. The CNY 89 contrast relies on cross-task utility restrictions; removing task 2 or 5 removes its identification. It is not a direct same-fee randomized comparison. Conditional A/B estimates and the stated early-record sensitivity accompany the primary models.

## Results and interpretation

Ordinary linear-fee logit has a held-out joint negative log score per choice of 1.031745. Ordinary observed cells, random-intercept linear fees and random-intercept cells give 1.032050, 1.031899 and 1.032229. More flexible specifications show no predictive improvement for the current human target.

The positive same-fee package association has intervals above zero in ordinary and random-intercept cells, while conditional A/B uncertainty spans zero. Excluding the flagged early record gives 0.508 (0.030–0.987) in ordinary cells. That record remains in the primary analysis; no invalidity is inferred from its timing alone. These model-conditional exploratory results do not establish an isolated decision-right mechanism or population WTP.

App fees, privacy, sharing and usage-history heterogeneity have close prior applications, including [Nittas et al. (2020)](https://www.jmir.org/2020/11/e18889/) and [Xie and Or (2023)](https://mhealth.jmir.org/2023/1/e25908/). The current positive package estimate is retained as an application result. It does not by itself establish an independently innovative JOCM submission. Applicable recruitment, consent and research statements must follow the actual materials, with no invented approval.
