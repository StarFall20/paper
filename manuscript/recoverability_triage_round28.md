# Round 28: recoverability-aware model-family triage

## What was run

The revised experiment does not collect new responses. It uses the existing
smart-device data-generating process and a public Swissmetro DCE. Every
simulation split is by respondent: 60% development, 20% validation, and 20%
test. Thresholds are calibrated once on independent additive simulations and
are frozen before the eight mechanism conditions are evaluated.

The diagnostic has two separate signals. Let (L_0), (L_S), and (L_F)
denote validation log loss for the additive MNL, a predeclared structured MNL,
and a flexible boosted classifier. The observable-structure signal is

\[
G_S=L_0-L_S,
\]

and the broad predictive-departure signal is (G_F=L_0-L_F). For respondent
(i), let (s_i=sum_t s_{it}) be the additive-MNL score accumulated over
their tasks and (I_i=sum_t I_{it}) the corresponding expected information.
The heterogeneity signal is the scale-free score-overdispersion ratio

\[
Q=\frac{\sum_i s_i' s_i}{\operatorname{tr}(\sum_i I_i)}.
\]

The decision rule has four outputs:

1. **base** when neither signal exceeds its additive-calibrated threshold;
2. **observed_structure** when (G_S) is large and (Q) is not;
3. **heterogeneity** when (Q) is large and (G_S) is not;
4. **unresolved** when both signals are large, when the flexible learner
   improves without an improvement from the declared utility library, or when
   a frozen richer coverage basis improves beyond that library.

This is a decision object, not a new likelihood-ratio test. It answers a
different question from an assisted specification search: whether the data
support expansion of the observed utility, expansion of the respondent
distribution, or an explicit unresolved status. The unresolved status is
part of the estimand because observable functional-form error and latent
heterogeneity can generate similar predictive gains.

## Simulation result

The formal run uses 100 additive calibration replications and 24 evaluation
replications per condition, with 400 respondents and 12 tasks. The rule has
the following action accuracy against the known simulation mechanism:

| condition | action accuracy | dominant action | triage regret | validation-only regret |
|---|---:|---|---:|---:|
| additive | 0.792 | base (0.792) | 0.0091 | 0.0039 |
| nonlinear | 1.000 | observed_structure (1.000) | 0.0028 | 0.0028 |
| threshold | 0.542 | observed_structure (0.542) | 0.0063 | 0.0027 |
| interaction | 0.875 | observed_structure (0.875) | 0.0129 | 0.0030 |
| heterogeneity | 0.958 | heterogeneity (0.958) | 0.0749 | 0.0655 |
| nonlinear + threshold | 0.917 | observed_structure (0.917) | 0.0073 | 0.0019 |
| nonlinear + interaction | 0.917 | observed_structure (0.917) | 0.0089 | 0.0025 |
| combined | 0.542 | unresolved (0.542) | 0.0763 | 0.0329 |

The results establish the intended boundary. The rule recovers strong
observable departures and isolated random taste heterogeneity. Thresholds are
harder to expose at the current signal strength. When observable structure
and heterogeneity coexist, the unresolved branch is selected in 54.2% of
replications; the remaining runs are classified as observed structure. The
triage rule is not claimed to dominate validation-only prediction. Its value
is the mechanism and recoverability decision, while the regret columns show
the cost of treating that decision as a forecast rule.

The out-of-library stress test adds a cubic utility term absent from the
primary structured library. A richer coverage basis contains all marginal
squares/cubes, pairwise products, and observed-covariate products. Its
increment over the primary library is calibrated on the same additive
replications. All 24 cubic replications are classified as unresolved. This is
the coverage check: a candidate library can be structurally useful without
being treated as complete.

## Public DCE check

The external audit uses the canonical purpose-1/3 Swissmetro sample: 6,768
choice observations from 752 respondents, with GA-adjusted train and
Swissmetro costs and stated-preference availability. The respondent split is
again 60/20/20. The validation gains are 0.01234 for an alternative-specific
time/cost MNL and 0.02519 for the flexible learner. The additive score
overdispersion ratio is 1.1845. On the independent test respondents, log loss
is 0.79821 for additive MNL, 0.76579 for the structured MNL, and 0.79599 for
the flexible learner. The flexible validation gain therefore does not transfer
to the test block, while the declared structured expansion does. This is an
external predictive check; the public file has no mechanism labels and does
not identify a heterogeneity or functional-form explanation.

## Novelty boundary

The contribution is not a new heterogeneity score, a new flexible learner, or
a new nested model test. JOCM already contains formal structural
heterogeneity diagnostics, assisted specification workflows, and flexible
choice models. The defensible increment is the predeclared mapping from two
different recoverability signals to a model-family action with an explicit
unresolved branch, followed by independent test evaluation of both mechanism
classification and policy regret. The paper must present this as a selective
diagnostic benchmark and decision protocol, not as a universal model selector.

The earlier candidate-preserving fibre statistic is excluded from this claim.
On finite support its saturated version tests the same unrestricted profile
alternative as a profile likelihood-ratio test. It remains a design
certificate and a negative-control audit, not an independent inferential
contribution.

## Required next experiments before submission

- Vary respondent count, task count, and signal strength to obtain power and
  false-expansion curves for (G_S) and (Q).
- Compare the score gate with a converged mixed-logit fit and report WTP and
  policy-regret consequences.
- Keep the Swissmetro table descriptive and label the lack of mechanism
  identification explicitly.

Raw outputs are `results/recoverability_triage.csv`,
`results/recoverability_triage_summary.csv`,
`results/recoverability_triage_calibration.csv`,
`results/recoverability_triage_ood.csv`, and
`results/swissmetro_recoverability_audit.csv`. The executable entry points are
`analysis/recoverability_triage.py`, `analysis/recoverability_triage_ood.py`, and
`analysis/swissmetro_recoverability_audit.py`.
