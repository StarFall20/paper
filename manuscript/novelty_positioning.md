# Novelty and journal-fit audit

## Editorial verdict

The broad ML-assisted model-search framing is below the Journal of Choice
Modelling novelty bar. Recent JOCM work already covers assisted specification,
extensive Mixed-Logit hypothesis search, flexible behavioural models, and
data-driven specification support.

The independent contribution that survives the overlap audit is **anchored
behavioural metamorphic specification testing (BMST) for a coarsened utility
basis**. It tests a candidate-implied equality across randomized choice tasks.
The equality preserves the complete menu vector of candidate utility
differences, not a selected scalar. The design adds a zero-candidate-difference
anchor and a geometry-preserving common translation, then reports a
geometry-changing decomposition arm as a comparison-complexity control.

This is a candidate-conditioned finite-sample audit. It is eligible only when
the candidate basis has nontrivial feasible fibres. A full raw-attribute linear
basis can have singleton fibres, in which case this relation contains no test
content. The method does not claim a universal MNL test, unique omitted-term
identification, or global model correctness.

## What is independent

The contribution has one scientific object: a deliberately designed violation
of a candidate-preserving choice relation.

1. **Object.** For candidate basis \(b\), define
   \(\phi(x)=\{b_j(x)-b_k(x):j<k\}\). Compare task versions with the same
   \(\phi\) and different raw decompositions.
2. **Anchor.** A zero candidate gap fixes the binary probability at \(1/2\)
   under symmetric errors for every positive scale. This gives a scale-invariant
   control.
3. **Geometry control.** A common raw translation leaves pairwise raw geometry
   unchanged. A separate decomposition arm changes geometry and measures the
   comparison-complexity response rather than folding it into utility
   misspecification.
4. **Identification and reference.** One member per pair is randomized at the
   respondent level. The candidate is frozen on a development fold, and a
   respondent-cluster sign-flip reference tests the relation on held-out
   respondents.

These pieces form one design and estimand. The zero-gap and geometry arms are
controls for the same fibre relation; they are not stitched-on theories.

## Closest prior work and boundary

| Study | What it already contributes | Boundary for this paper |
|---|---|---|
| Ortelli et al. (2021) | Search over interpretable choice specifications | BMST does not search a utility library; it tests a predeclared candidate relation on designed tasks |
| Fok and Paap (2025) | Composite-likelihood/GMM tests for MNL and IIA misspecification using alternative pairs | BMST changes task decompositions while preserving the complete candidate menu vector |
| Pedersen et al. (2011) | Randomly includes or excludes a cost attribute and tests parameter equality | Their intervention changes the displayed attribute set; BMST creates candidate-equivalent task versions |
| Shubatt and Yang (2024) | Choice probabilities depend on utility difference and comparison complexity from raw geometry | BMST preserves geometry in its primary arm, includes a zero-gap anchor, and reports geometry-changing responses separately |
| Task-complexity studies | Complexity affects stated-choice responses | Complexity is a measured control and failure boundary, not the target estimand |

The search log records exact-title queries and the limits of direct index access:
manuscript/index_search_log_round7.md. The novelty statement should say
“to our knowledge after the reported search” and identify the tested relation,
rather than claim the first specification test.

## Evidence already available

The 100-replication planning benchmark uses 400 respondents and 199
respondent-cluster sign flips. Under an omitted decomposition, observational
LR and the residual proxy reject at .04 and .04, while the zero-gap
geometry-preserving arm rejects at .83 and the geometry-changing nonzero-gap
arm at 1.00. In a complexity-only condition, the zero-gap arms remain near
size (.02 and .07), while the geometry-changing nonzero-gap arm responds at
.41. A sample-size curve shows nonlinear power of .20/.42/.70 for one focal
pair at 150/300/600 respondents and .90/1.00/1.00 for three focal pairs.

These are planning results. They establish a falsifiable design comparison,
not an empirical behavioural finding.

## Journal and reader alignment

The topic fits JOCM's methodological scope because it studies utility
specification, survey-task design, behavioural interpretation, and model
validation. The paper should lead with the practical problem: a candidate
coarsening can fit the observed support while leaving an omitted decomposition
direction nearly untested. The intended readers are choice modellers who design
stated-choice tasks, specify utility functions, and need diagnostics with a
clear behavioural interpretation.

## Evidence required before submission

- Field a forced-choice supplement or find data with documented one-member
  assignment and candidate-preserving transformations.
- Pre-register the fibre screen, full menu vector, zero-gap and geometry arms,
  assignment, statistic, sign-flip reference, and mixed-signature action.
- Report additive, random-taste, scale, order/carryover, and complexity
  controls, including response time and raw pairwise geometry.
- Compare BMST with observational LR and residual learning on the same sample
  budget.
- Rewrite the full Word manuscript and submission materials around the anchored
  fibre claim. Keep ML/process analyses as comparators and failure boundaries.

Sources: [JOCM aims and scope](https://shop.elsevier.com/journals/journal-of-choice-modelling/1755-5345);
[Ortelli et al.](https://doi.org/10.1016/j.jocm.2021.100285);
[Fok and Paap](https://doi.org/10.1016/j.jocm.2024.100531);
[Pedersen et al.](https://doi.org/10.1016/S1755-5345(13)70044-7);
[Shubatt and Yang](https://arxiv.org/abs/2401.17578).
