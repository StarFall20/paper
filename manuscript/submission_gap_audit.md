# Submission-gap audit after the innovation search

## Objective verdict

The original claim, “a Random Forest helps discover nonlinear terms for a
choice model,” is below the JOCM submission threshold. Assisted specification,
random-forest diagnostics, recursive partitioning, latent-class neural choice
models, and flexible behavioural networks already occupy that space.

The revised claim, “a calibrated process- and recoverability-aware triage rule
selects, or declines to select, a behavioural model family,” is potentially
publishable. It has a distinct estimand—the expected policy loss of a forced
model-family decision versus an unresolved/defer action—but the repository does
not yet contain the evidence needed to defend it as a contribution.

## Current distance from the gate

| gate | current state | distance |
|---|---|---|
| distinct scientific question | defined in `revision_plan.md` | close |
| formal action set and loss | written conceptually; no fitted policy-loss rule | one implementation |
| functional-form diagnostic | corrected RF prototype exists | one coverage/stability audit |
| respondent heterogeneity gate | exploratory score-overdispersion prototype | bootstrap calibration required |
| process/sequence gate | 30-replication prototype exists | full competing process models required |
| taste versus ANA identification | explicitly shown to be unresolved in the prototype | needs an abstention-cost experiment |
| model-family comparison | additive/structured MNL, RF proxy, small LC/MXL checks | full MXL and ANA/process branch required |
| policy consequences | regret exists in simulation | WTP, elasticities, and scenario loss are missing |
| external validity | planned LPMC/Swissmetro transfer | data preparation and temporal/respondent holdout remain |
| manuscript claim | working draft still follows the earlier ML-assisted framing | full rewrite required |

The paper is therefore at the **candidate-contribution stage**, not at the
submission-ready stage. The innovation idea is no longer the main weakness;
the missing proof that the triage decision improves behavioural and policy
outcomes is.

## What would clear the gate

1. Freeze a decision rule using development data only. It must include
   `enrich`, `segment`, `continuous`, `process`, and `defer` actions.
2. Calibrate score and sequence thresholds by parametric bootstrap. Report
   size, power, and false expansion across respondent counts and task counts.
3. Add out-of-library utility mechanisms and explicit process mechanisms. The
   rule must be tested when the true mechanism is absent from the candidate
   library.
4. Estimate the competing behavioural branches under a common grouped split:
   enriched MNL, Mixed Logit, latent-class MNL, an ANA/process model, and a
   flexible learner.
5. Compare the rule with forced selection using WTP distortion, choice shares,
   elasticities, calibration, decision regret, and computation. The defer
   action must have an explicit measurement cost.
6. Transfer the frozen rule to at least one panel or temporal holdout. If an
   external dataset cannot identify attention or sequence behaviour, the result
   must remain unresolved.
7. Rewrite the introduction and discussion around mechanism recoverability,
   decision risk, and the abstention boundary. The Random Forest becomes one
   diagnostic component.

## Stop conditions

The process-innovation claim should be dropped if the defer action does not
reduce policy loss, if the rule depends on test outcomes, or if full competing
model branches cannot be estimated fairly. In that case the paper should keep
the narrower contribution: a reproducible recoverability benchmark for
ML-assisted utility enrichment, with an explicit statement that taste and
attribute-processing heterogeneity are not identified by the available data.

## Closest literature that sets the bar

- Assisted specification: https://doi.org/10.1016/j.jocm.2021.100285
- Recursive partitioning for heterogeneity: https://doi.org/10.1016/j.jocm.2022.100393
- Saliency and non-attendance: https://doi.org/10.1016/j.jocm.2022.100370
- Non-trading and inertia: https://doi.org/10.1016/j.jocm.2023.100413
- Disjunctive decision rules: https://doi.org/10.1016/j.jocm.2024.100510
- MNL misspecification tests: https://doi.org/10.1016/j.jocm.2024.100531
