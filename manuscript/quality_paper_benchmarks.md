# Method and writing benchmarks from strong choice modelling papers

## Assisted specification

Ortelli, Hillel, Pereira, de Lapparent, and Bierlaire frame specification search as a structured model-development problem. Their paper combines semi-synthetic and real choice data, keeps the output interpretable, and evaluates whether the search produces useful candidate specifications. The revision adopts the same logic at a smaller controlled scale: the learner proposes terms, the behavioural model carries the final interpretation, and the paper reports the conditions under which the workflow succeeds.

Source: https://doi.org/10.1016/j.jocm.2021.100285

## Validation as evidence

Parady, Ory, and Walker show that discrete choice papers often report fit statistics without enough internal or external validation. Their reporting principle is directly relevant here: the paper must state the unit of splitting, keep repeated choices from one respondent together, separate development from evaluation, and add an external holdout when the data structure permits it.

Source: https://doi.org/10.1016/j.jocm.2020.100257

## Changes applied to this revision

1. The research question is stated as a conditional model-development question, not a contest between algorithms.
2. Each simulation condition has one mechanism and the combined condition is reported as a joint stress test.
3. Predictive accuracy, behavioural recovery, calibration, and decision regret are separate outcomes.
4. Candidate-term selection is nested inside respondent-level development data.
5. Latent class and planned Mixed Logit extensions are used to distinguish discrete segmentation from continuous taste heterogeneity.
6. The current benchmark is labelled as a scaffold until the replication target, external datasets, and full model set are complete.

## Writing structure for the final paper

Use one paragraph to establish the research tension, one to state the contribution, one to define the validation design, and one to interpret each result. Put estimator settings, seeds, and extended tables in the supplement. State the operating condition beside each main conclusion so the result can be read without a defensive sequence of qualifications.
