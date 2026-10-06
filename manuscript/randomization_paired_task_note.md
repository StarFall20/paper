# Behavioral metamorphic specification test

## Method object

The proposed Behavioral Metamorphic Specification Test (BMST) constructs two
held-out choice tasks with identical
candidate utility differences for every alternative. The raw attributes can
change by a common shift, so a linear random coefficient on the shifted
attribute also leaves every individual's utility difference unchanged. The
observed contrast is the one-hot choice vector difference minus the frozen
candidate probability difference.

The candidate-preserving relation is the domain-specific metamorphic relation:
the second task is a follow-up input whose output must equal the first under
the candidate. This imports a test-oracle idea from metamorphic software and
simulation validation into choice-model specification. UFIT names the
utility-fibre relation; BMST names the complete test procedure.

The reference distribution uses respondent-level sign flips. Under the null,
the two task outcomes are exchangeable within the pair. A cluster sign flip
preserves dependence among a respondent's focal pairs. This is a design-based
randomization test; it is separate from the multiplier bootstrap used in the
earlier feasibility benchmark.

## Boundary run

The run uses 50 replications, 300 respondents, three focal pairs, four nuisance
tasks, and 199 cluster randomizations per split. The results are:

| DGP | candidate | rejection rate |
|---|---|---:|
| additive | additive | 0.06 |
| omitted nonlinear term | additive | 0.82 |
| omitted nonlinear term | oracle term | 0.08 |
| omitted threshold | additive | 1.00 |
| omitted threshold | oracle term | 0.02 |
| omitted interaction | additive | 1.00 |
| omitted interaction | oracle term | 0.04 |
| random cost sensitivity | additive | 0.06 |
| nonlinear term + random cost sensitivity | additive | 0.74 |
| nonlinear term + random cost sensitivity | oracle term | 0.02 |
| interaction + random cost sensitivity | additive | 1.00 |
| interaction + random cost sensitivity | oracle term | 0.00 |
| cubic out-of-library departure | additive | 1.00 |

The random-cost boundary is the key check. A common cost shift leaves each
respondent's utility differences unchanged even when the cost coefficient is
random, so the test need not confuse linear taste heterogeneity with a
functional-form violation. The combined conditions show that the test can
retain power against an omitted nonlinear or interaction term in the presence
of random cost sensitivity. The result is a simulation property, not evidence
that arbitrary heterogeneity is identified.

The shift-specific rows provide directional evidence without term-level
identification. Under the additive candidate, nonlinear power is 0.72 for the
time shift, 0.04 for the cost shift, and 0.70 for the joint shift. Threshold
power is 0.06, 1.00, and 1.00; interaction power is 0.24, 1.00, and 1.00.
The combined nonlinear-plus-random-cost condition gives 0.62, 0.04, and 0.66,
while the combined interaction condition gives 0.16, 1.00, and 1.00. These
rows are diagnostic of transformation sensitivity; they cannot be read as a
unique omitted-term decomposition.

## Negative-control benchmark

The separate BMST benchmark tests the mechanism boundary directly. Across 100
replications, rejection is .04 under the additive null, .68 for an omitted
nonlinear term, and .03 under random linear taste. A task-specific error-scale
drift rejects at .23 and a declared order effect rejects at 1.00. These
process and scale departures deliberately break the candidate-preserving
relation. They establish that a rejection identifies a relation violation; the
mechanism requires the declared negative controls and cannot be inferred from
the p-value alone.

The implementation is analysis/bmst_negative_controls.py and the frozen output
is results/bmst_negative_controls.csv.

## Pre-outcome transformation design

The proposed maximin layer is audited separately from the outcome test. On a
5-unit feasible grid, the fixed shifts `(20,0), (0,60), (20,50)` have a
minimum normalized separation score of 1.142 across quadratic, threshold,
interaction, and cubic probe utilities. A constrained maximin search with a
pairwise Manhattan separation of 30 and a total movement budget of 150 selects
`(5,45), (20,0), (35,45)` and raises the score to 1.632. An unconstrained
search repeats `(40,60)` three times; its score of 3.000 demonstrates why a
diversity constraint is part of the design object.

The companion 50-replication benchmark gives null rejection of 0.06 for the
fixed and constrained designs. Constrained maximin raises nonlinear power
from 0.82 to 1.00 and nonlinear-plus-random-cost power from 0.74 to 1.00.
Threshold, interaction, and cubic out-of-library power are 1.00 for both
designs. These results justify treating the constrained search as a pre-outcome
sensitivity layer. They do not turn the score into a new misspecification
estimand, and the repeated-shift solution is excluded.

## Identification conditions and limits

The randomization reference requires a frozen candidate estimated outside the
held-out pairs, exchangeability of the two task outcomes under the candidate,
no carryover or order effect, and a task construction that preserves the
candidate utility differences for every allowed coefficient vector. A shift
that preserves only the population mean utility does not satisfy the null.
The test detects a violation of candidate-task invariance. It does not label a
unique omitted nonlinear, threshold, or interaction term.

The original A/B/opt-out instrument does not automatically satisfy this design.
Its opt-out alternative has no product attributes, so a shift applied only to
the two products changes the product-versus-opt-out utility differences. A
paired-task supplement must apply a common task-level transformation to every
alternative, including opt-out, or define a coefficient-wise compensating
construction. The current public panels do not contain such randomized pairs;
the method is therefore a design requirement for a new supplement, not an
empirical claim from the existing data.

## Literature boundary

Metamorphic testing addresses test-oracle construction in software and
simulation validation. Breitmoser (2021) characterizes conditional logit through observable
translation, presentation, context, and IIA invariances. Fok and Paap (2025)
construct MNL misspecification tests from alternative pairs and composite-
likelihood/GMM moments. The present contribution can be distinguished only at
the method-transfer level: it uses cross-task transformations that preserve
the candidate utility-difference vector, then obtains a cluster randomization
reference from paired task exchangeability. It must not claim to introduce
invariance testing or a general MNL misspecification test.

The implementation is `analysis/randomization_paired_task_test.py`; the frozen
output is `results/randomization_paired_task_test.csv`. It contains all-pair
and predeclared shift-specific rows.

The design audit is implemented in `analysis/metamorphic_design_score.py` and
its finite-sample comparison in `analysis/metamorphic_design_benchmark.py`;
frozen outputs are `results/metamorphic_design_score.csv` and
`results/metamorphic_design_benchmark.csv`.
