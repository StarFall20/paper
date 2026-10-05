# Submission-gap audit after the innovation search

## Objective verdict

The latest independent idea search changes the preferred route. A
model-equivalent choice-pair test has a cleaner new object than the earlier
process-triage proposal: it tests a utility-basis equality created by the
choice design. The prototype is promising, but the empirical claim is blocked
by the paired-task data condition.

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
submission-ready stage. The remaining gap is evidence that the paired-task
equality has correct size, useful power, and a valid empirical instrument.

The observational fallback has now been tested. Cross-fitting and respondent-
clustered multiplier bootstrap give a near-5% null size in the engineered
boundary run, but power is only 0.225 for the nonlinear condition and 0.175
for the interaction condition. On public Swissmetro, the p-value changes from
about 0.43–0.48 under row-order orientation to 0.003–0.033 under time
orientation at the larger tolerances, while cost orientation gives 0.847–0.977.
This orientation sensitivity is a direct failure boundary. The branch is a
documented limitation and does not clear the empirical gate.

The exact randomized benchmark provides a cleaner test of the proposed object:
50 replications give 0.06 null rejection, 0.84 nonlinear power, and 1.00
threshold and interaction power for the base candidate. Damped-Newton repair
reduces rejection to 0.04, 0.02, and 0.02 after the corresponding terms are
added. A four-cell probability-scale factorial audit was then run as a
localisation repair. It gives 0.28 rejection for a pure interaction and 0.86
when an interaction coexists with a nonlinear main effect. The audit shows
that the mixed probability contrast is contaminated by the nonlinear link, so
term-level localization is excluded from the innovation claim.

## New pivot gate

The paired-task test can replace the process-triage claim only if it passes
four checks: bootstrap size near the declared level, power against omitted
nonlinear and interaction terms, loss of rejection after the correct term is
added, and a valid Swissmetro or supplementary paired-task instrument. Term
localization is outside the supported claim after the factorial audit. Without
the final data condition, the test remains a simulation contribution and the
manuscript should present the recoverability benchmark as the main empirical
result.

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
