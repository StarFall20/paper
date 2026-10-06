# Journal of Choice Modelling revision plan

## Main research question

Can a fitted candidate basis that coarsens the raw attribute space be tested on
new choice tasks that it declares equivalent? Anchored Behavioral Metamorphic
Specification Testing (BMST) answers this question by constructing task pairs
with the same full menu difference vector and different raw decompositions.
The Utility-Fibre Invariance Test (UFIT) compares their observed choice
probabilities with a zero-difference anchor and a geometry-preserving control.
The primary test is not run when the feasible fibre is a singleton.

This is the only new object in the main paper. Random Forest, XGBoost, Mixed
Logit, latent-class MNL, and process models are comparison or negative-control
tools. They do not form separate contribution claims.

## Contribution statement

The paper develops a candidate-conditioned exchangeability test for discrete
choice specification. A development fold freezes the candidate basis. A
predeclared transformation preserves every alternative-pair difference in the
menu. A one-member-per-pair assignment prevents respondents from seeing both
members. Respondent-cluster sign flips provide the finite-sample reference
under the candidate null. The binary primary arm includes a zero-candidate-
difference task and a common translation that preserves pairwise raw geometry.
The estimand is a choice-probability relation on a declared nontrivial task
fibre.

The paper does not claim a universal MNL misspecification test, a new
invariance axiom, a unique omitted term, or global model correctness.

## Manuscript architecture

1. **Introduction.** Explain why a good holdout score does not test whether a
   coarsened utility library treats its own equivalent tasks consistently.
   State anchored BMST and the local interpretation boundary.
2. **Related literature.** Position BMST against assisted specification,
   axiomatic invariance, MNL misspecification tests, metamorphic testing,
   process models, and DCE design. State the exact delta in one paragraph.
3. **Candidate-fibre theory.** Define the basis, full menu difference vector,
   nontrivial-fibre screen, candidate-preserving transformation, probability
   relation, and estimand. State the softmax equivalence proposition.
4. **Randomized instrument.** Define the development/test split, one-member
   assignment, orientation coin, cluster score, sign-flip reference, order
   counterbalance, opt-out boundary, and failure conditions.
5. **Falsification-oriented design.** Select geometry-preserving translations
   and decomposition probes by a declared movement budget. Report the zero-gap
   anchor, raw-distance balance, and any complexity-only arm before outcomes.
6. **Simulation testbed.** Retain the smart-medical-device generator as a
   controlled benchmark and add the coarsened-burden binary design. Compare
   anchored BMST with an observational LR test and an out-of-fold residual
   learner. Conditions include omitted decomposition, random taste, scale,
   order, and comparison complexity.
7. **Results.** Report size, power, repair, negative controls, design
   sensitivity, sample-size/task-count sensitivity, and the action for mixed
   signatures. Keep mechanism attribution bounded.
8. **Empirical instrument and data boundary.** Report the public-data audit and
   specify the paired-task supplement. If the supplement is not available,
   present the work as a simulation-validated method proposal rather than a
   completed empirical application.
9. **Discussion.** Explain the tested-domain value, limits of non-rejection,
   scale and process confounding, and the path to external validation.

## Role of the original Word manuscript

The original title, “When Does XGBoost Improve Choice Prediction for Smart
Medical Devices?”, and its lambda crossover are retained as an appendix
benchmark. The crossover is specific to the generator and cannot carry the
main novelty claim. The smart-device attributes remain useful because they
provide a transparent, multi-attribute testbed for the BMST relation.

## Evidence already available

- UFIT algebra and the randomization validity proposition are recorded in
  `manuscript/ufit_proposition.md`.
- The one-member assignment benchmark has 100 replications and 600
  respondents. Rejection is .05 under the additive null, .79 for omitted
  nonlinearity, .05 under random linear taste, and 1.00 for an omitted
  interaction.
- The negative-control benchmark has 100 replications. Rejection is .04 for
  the null, .68 for omitted nonlinearity, .03 for random taste, .23 for scale
  drift, and 1.00 for order effects.
- The constrained maximin design raises the weakest normalized probe score
  from 1.142 to 1.632. A repeated-shift unconstrained solution is retained as
  a failure boundary.
- The public-data audit found no existing dataset with the required
  candidate-preserving pair and one-member assignment. See
  `manuscript/public_data_pair_audit.md`.
- The anchored-fibre benchmark uses a coarsened burden (s=a_1+a_2) and an
  omitted decomposition coordinate (z=a_1-a_2). With 100 replications and
  400 respondents, the observational LR and residual-learning proxy reject at
  .04 and .04 under the omitted decomposition condition; anchored rejection is
  .83 for the zero-gap geometry-preserving arm and 1.00 for the nonzero-gap
  geometry-changing arm. Under complexity alone, the zero-gap and
  geometry-preserving rates are .02 and .07, while the geometry-changing
  nonzero-gap rate is .41. See `results/anchored_fibre_benchmark.csv`.
- The 60-replication sample-size planning run reports nonlinear power of .20,
  .42, and .70 at 150, 300, and 600 respondents with three focal pairs;
  interaction power is .90, 1.00, and 1.00. See
  `results/bmst_power_curve.csv`.

The frozen BMST outputs were re-run with the bundled analysis environment on
2026-10-06. Both benchmark CSVs matched the committed files byte for byte.

## Submission gates

1. Complete the target replication and sensitivity runs, including the
   coarsened-fibre power curve at the preregistered replication count.
2. Freeze the nontrivial-fibre screen, transformation grid, zero-gap anchor,
   geometry metric, movement budget, assignment rule, and unresolved-action
   rule before any paired-task outcome is inspected.
3. Obtain a paired-task supplement or a public dataset with an equivalent
   assignment structure.
4. Add the paired-task results to the main text, with usable-pair counts,
   balance checks, response-time/complexity diagnostics, and the LR/residual
   comparison.
5. Run the locked workflow on the external choice datasets only as transfer
   checks; do not relabel observational matches as randomized BMST evidence.
6. Rewrite the full Word manuscript and appendices around the frozen anchored
   BMST results, then run document render and visual QA before submission.
