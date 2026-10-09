# Smart device parity supplement design

## Purpose

The supplied manuscript already defines six smart-device attributes and an
additive MNL candidate. The following paired task uses those exact attributes
to test whether that candidate basis is sufficient. It is a design audit and a
simulation check; it is not a human-data result.

## Candidate-preserving task pair

The four profiles use the manuscript's level order: use-management mode,
intelligent functionality, professional support, clinical evidence, data
management, and price.

| Profile | Mode | Intelligence | Support | Evidence | Data | Price |
|---|---|---|---|---|---|---|
| A+ | Hospital setup + periodic visits | Basic preset control | Periodic follow-up | Multicenter evidence | Authorized cloud sharing | RMB 3,000 |
| A- | Hospital setup + periodic visits | Personalized intelligent adjustment | Periodic follow-up | Single-center evidence | Authorized cloud sharing | RMB 3,000 |
| B+ | Hospital-based | Personalized intelligent adjustment | Periodic follow-up | Single-center evidence | Authorized cloud sharing | RMB 3,000 |
| B- | Home use + remote follow-up | Basic preset control | Periodic follow-up | Basic validation | Authorized cloud sharing | RMB 3,000 |

Under the supplied additive candidate, the four profile utilities are
(V_{A+}=V_{A-}=0.23) and (V_{B+}=V_{B-}=-0.17). The A--B candidate gap is
0.40 in both versions. The opt-out utility remains -0.42, so the full candidate
menu vector is `(0.23, -0.17, -0.42)` in both versions.

The squared distance between the A and B level-index vectors is 6 in both
versions. This gives a direct even-geometry control. The data-generating
interactions already used in the manuscript give

- (h_{A+}=0), (h_{A-}=0.55) from personalized intelligence with cloud sharing;
- (h_{B+}=0.55), (h_{B-}=0) from the same interaction;
- (h_{A+}-h_{B+}=-0.55) and (h_{A-}-h_{B-}=+0.55).

The omitted interaction is exactly odd across the pair while the candidate
menu and the squared A--B geometry stay fixed. The transformation changes which
attribute bundle carries the same candidate utility. It therefore tests the
candidate basis in the original application language.

## Simulation check

`analysis/smart_device_fibre_design.py` enumerates the 3^6 profile space,
selects the task above, verifies every equality, and simulates one-member
random assignment with an A/B/opt-out response. In 200 replications with 400
respondents and 199 respondent-cluster sign flips, the parity rejection rate is
.025 under the additive null and .995 when the manuscript's intelligence-cloud
interaction is active. These values are planning evidence. They do not replace
a paired DCE with real respondents.

## Field implementation constraints

The two versions must be shown to separate respondents or randomized across
respondents so that a participant does not learn the reflection. Wording,
attribute order, visual emphasis, and opt-out placement must be counterbalanced.
The final instrument should include several independent candidate-preserving
pairs, a zero-gap control, response time, certainty, and an even-component
complexity measure. The task should be dropped if pilot interviews show that
respondents interpret the two profiles as different product concepts rather
than as compensated attribute tradeoffs.

## Five-task candidate grid

`analysis/smart_device_fibre_candidate_grid.py` enumerates 677 unique reflections
under the same constraints and selects five pre-outcome tasks with candidate
A--B gaps 0.20, 0.40, 0.60, 0.80, and 1.00. Each selected task has equal plus
and minus squared distance and an odd interaction signal of 0.55. The complete
profiles and candidate menu vectors are in
`results/smart_device_fibre_candidates.csv`. The grid is a design reservoir;
field piloting must remove tasks that fail semantic or cognitive checks.

## Fibre-optimal selection rule

The candidate grid can be ranked before outcomes by the local odd-information
criterion
\[
I_{\mathrm{odd}}(t)=p_t(1-p_t)\,s_t^2,
\]
where (p_t) is the candidate A--B choice probability and (s_t) is the
magnitude of the odd omitted-gap signal. The final ranking also requires the
raw omitted gap to be exactly antisymmetric, so the primary design carries no
planned even interaction component. Among 648 candidates satisfying this
purity constraint, the selected five tasks have candidate gaps approximately
0.20, 0.35, 0.55, 0.75, and 0.95, with odd signal 0.90 in every task. The
criterion is a pre-outcome design rule; semantic screening remains mandatory.

## Calibration-error audit

The exact equalities use the supplied simulation coefficients. In an empirical
study, a separate pilot must estimate the candidate coefficients before the
paired block is finalized. `analysis/smart_device_calibration_robustness.py`
perturbs all candidate coefficients while keeping the selected task fixed and
simulates the additive null. With 100 replications and 400 respondents, the
null rejection rate is .020 at zero perturbation, .030 at coefficient noise
SD .02, and .150 at SD .05. The design should proceed only when a pilot-based
uncertainty envelope keeps the plus/minus candidate-gap mismatch below a
predeclared tolerance; the paired block must use independent respondents from
the calibration sample.

## Multi-task design comparison

A matched five-task simulation compares the ordinary candidate grid with the
fibre-optimal grid at 150 respondents. With a weak omitted interaction
strength (0.3), the ordinary grid rejects at .310 and the fibre-optimal grid at
.890; under the combined omitted-plus-even-scale condition the rates are .270
and .740. Under the additive null, the rates are .040 for both designs, and
under an even-scale nuisance they are .030 and .050. At the stronger
interaction strength (1.0), both designs reach 1.000 power. The design rule
therefore improves sensitivity in the weak-signal regime while preserving
planning size; it is not presented as a universal power dominance claim.
