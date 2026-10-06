# Candidate-preserving paired-task randomization test

## Method object

The proposed method constructs two held-out choice tasks with identical
candidate utility differences for every alternative. The raw attributes can
change by a common shift, so a linear random coefficient on the shifted
attribute also leaves every individual's utility difference unchanged. The
observed contrast is the one-hot choice vector difference minus the frozen
candidate probability difference.

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

Breitmoser (2021) characterizes conditional logit through observable
translation, presentation, context, and IIA invariances. Fok and Paap (2025)
construct MNL misspecification tests from alternative pairs and composite-
likelihood/GMM moments. The present contribution can be distinguished only at
the implementation level: it uses cross-task transformations that preserve
the candidate utility-difference vector, then obtains a cluster randomization
reference from paired task exchangeability. It must not claim to introduce
invariance testing or a general MNL misspecification test.

The implementation is `analysis/randomization_paired_task_test.py`; the frozen
output is `results/randomization_paired_task_test.csv`. It contains all-pair
and predeclared shift-specific rows.
