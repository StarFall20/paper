# Auditing testability in released discrete-choice designs

Technical resource note — 9 November 2026

## Purpose

This resource checks the positive-mass support of a declared candidate menu representation before a new choice experiment is fielded. It implements classical finite-support lack-of-fit dimensions, projected categorical information and a matched-comparator check. No new general test or optimality principle is claimed.

## Demonstrated scope

Two released stated-choice archives are audited: 3,492 Swiss route-choice tasks (3,417 unique menus) and 1,484 three-alternative vehicle tasks. Four value-of-time settings and three ownership horizons, each under two declared representations, produce fourteen audits. The analyst-defined summaries are sensitivity cases, not original-author hypotheses. The original answer outcomes are not used to infer behavioral sufficiency or valid assignment.

At 10 CHF/hour, the route archive has 17 menu contrasts under an absolute summary and 2,157 under the complete utility-difference vector. These are different nonparametric nulls. The vehicle catalogue has singleton fibres in the reported settings. These facts do not invalidate either original study or establish a prevalence rate for efficient-design failures.

## Classical identity

For a saturated fibre response surface and a fixed local departure space, within-fibre categorical information equals the nuisance-parameter Schur complement of the corresponding added-term model. Rank, directional local power and the matched weakest-direction criterion follow from that matrix. Finite-amplitude profiled KL has the same local second-order coefficient.

## Generated diagnostic designs

Sixteen pairs per source archive preserve the complete declared candidate menu under specified marginal ranges and ticks. At a fixed choice budget, balanced two-point allocation maximizes both the fibre and classical information, with candidate parameter information unchanged under the null. Full joint feasibility and respondent comprehension were not established. These generated probes are not deployable questionnaires.

## Comparator evidence

Eighteen controlled settings use 500 replications and 499 fixed-margin reference draws each. The full-profile score and classical CMH produce identical p-values in all 9,000 replications; a saturated profile LR gives similar rejection rates. Simulated units each supply one choice. This benchmark is not a sample-size recommendation for a repeated-question human survey.

## Use

The generic entry point accepts complete raw menus, complete candidate menus and weights as JSON. It uses exact rational comparisons and reports positive-mass contrast dimensions. It never reports a positive rank as evidence of sufficiency, statistical power, original feasibility, actual random assignment or human provenance.

The public source datasets are obtained from their official sites and checked by hashes. Raw DH36 questionnaires remain private and exploratory. Research users should cite the original archives and classical method sources listed in the accompanying decision report.

## Publication status

This is a completed reproducibility and design-diagnostic resource. Independent methodological novelty and readiness for Journal of Choice Modelling submission have not been established. The previous new-method manuscript is retained as an archived working draft. No new questionnaire is recommended on the basis of the retired contribution claim.
