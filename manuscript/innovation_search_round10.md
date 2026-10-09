# Round ten: collision audit and bounded novelty claim

## Decision

Retain parity-separated utility-fibre testing as the primary innovation, with a
bounded claim. The contribution is a designed specification audit for a
candidate utility basis. It constructs a reflection inside a nontrivial fibre,
holds the full candidate menu vector fixed, and uses the odd response as the
estimand after even distance/scale mechanisms are cancelled. The contribution
is not a new link function, a generic symmetry contrast, or a claim that every
choice error is an omitted utility term.

## Closest literature and the actual boundary

| Work | What it contributes | What remains distinct here |
|---|---|---|
| He and Natenzon, *Moderate Utility* (2024) | Places many choice models in a class where binary probability depends on utility difference divided by a distance index, with moderate transitivity as the testable implication. | It does not construct a candidate-preserving experimental fibre, impose a reflection, or separate an odd utility response from an even distance response. Our extension uses their class as a robustness boundary. |
| Shubatt and Yang, *Tradeoffs and Comparison Complexity* (2024; updated 2026) | Gives a theory and experiments showing that tradeoff difficulty predicts choice noise and can reverse valuation regularities. | It studies how complexity changes behavior. It does not test whether a declared utility basis is sufficient along a same-candidate-utility fibre. The present design makes complexity an even nuisance and reports its even component. |
| Chicu and Masten, *A specification test for discrete choice models* (2013) | Tests choice-set invariance by comparing probabilities across different menus. | The present test keeps the complete menu vector fixed and changes the decomposition of attributes inside one candidate-equivalence class. |
| Ortelli et al., *Assisted specification of discrete choice models* (JOCM, 2021) | Searches over utility specifications with a multi-objective combinatorial algorithm. | It selects specifications using fit and parsimony. The present test creates a design-based falsification implication before outcomes are observed. |
| Symmetric paired DCE design work (2023) | Uses symmetry and balance to improve experimental information or efficiency. | Symmetry is an efficiency device there. Here reflection is the estimand: the odd component tests candidate sufficiency and the even component diagnoses processing. |
| Procedural-invariance tests in health-state valuation | Compare procedures intended to represent the same underlying utility and interpret differences as procedural effects or preference structure. | Those designs compare elicitation procedures. The present instrument uses an attribute-space orbit, preserves all menu utility gaps under the candidate, and derives an odd/even decomposition with a cluster randomization reference. |
| JOCM work on choice-set complexity and sample size | Models complexity as a source of error variance or scale heterogeneity. | The present design does not estimate a complexity parameter first. It cancels the declared even component algebraically and uses the retained component as a specification contrast. |

The searched web-indexed literature did not reveal an exact discrete-choice
specification test with all four elements together: (i) a candidate-preserving
attribute fibre, (ii) full-menu-vector preservation, (iii) a reflection-coded
odd/even response decomposition, and (iv) a randomized inference rule for the
odd component. This is a preliminary boundary statement. A database-level
Scopus and Google Scholar export still needs to be deduplicated before the
paper makes a stronger novelty sentence.

## Why the difference matters

A standard LR or residual test learns from the support already present in the
data. If the observed support is concentrated near one decomposition, a
nonlinear omitted direction can have little leverage. A menu-change test answers
a different question about context dependence. A conventional symmetric DCE
improves information but does not turn symmetry into a specification
restriction. The parity instrument creates a pre-outcome relation that is
orthogonal to an explicitly declared even distance/scale nuisance and probes an
odd omitted direction on the candidate's own equivalence class.

The incremental planning benchmark makes the distinction observable: under a
narrow observational support, LR and residual rejection are .060 and .030 for
the omitted direction, while parity is 1.000. Under an even scale nuisance,
LR and residual rejection are .030 and .060, parity is .050, and the one-sided
fibre comparator rejects at 1.000. This is evidence for a design contribution,
not evidence that the parity statistic dominates every diagnostic in every
support.

## Novelty sentence for the manuscript

> We introduce a reflection-symmetric utility-fibre specification test for
> discrete choice models. The design holds the candidate's complete menu
> difference vector fixed, pairs each task with its reflected attribute
> decomposition, and uses the odd choice-probability component to test for an
> omitted utility direction while the even component records processing or
> scale changes. The test is local to a declared symmetry library and treats
> rejection as a relation violation rather than unique mechanism identification.

## Failure boundaries that must remain visible

1. A saturated candidate basis makes the fibre a singleton, so the test has no
   content.
2. If the reflection changes perceived meaning, salience, wording, order, or
   attention asymmetrically, the odd contrast is contaminated. These features
   require independent counterbalancing or the nuisance-orthogonal fallback.
3. Preserving one pairwise gap is insufficient in a multinomial task. The full
   menu vector must be held fixed.
4. An omitted term with an even response is invisible to the primary contrast;
   the even component and a separate nuisance audit are required.
5. Non-rejection supports the tested relation under the chosen design. It does
   not prove global sufficiency.

## Required next evidence

- Export and deduplicate the Scopus and Google Scholar search results using
  the exact terms `utility fibre`, `utility equivalence`, `moderate utility`,
  `procedural invariance`, `symmetric paired choice`, and `choice model
  specification test`.
- Lock the DCE wording, order counterbalance, full-menu preservation rule,
  parity statistic, sign-flip reference, and multiplicity rule before viewing
  outcomes.
- Run the moderate-utility link check at the final replication count and add
  an empirical paired DCE or label the paper as simulation-validated method
  development.

## Application feasibility in the supplied smart-device manuscript

The original six attributes contain an exact candidate-preserving reflection.
Using its additive coefficients, the selected profiles satisfy
(V_{A+}=V_{A-}=0.23), (V_{B+}=V_{B-}=-0.17), and the same opt-out utility
(-0.42). The full candidate menu vector is therefore `(0.23, -0.17, -0.42)`
in both versions. The squared A--B level-index distance is 6 in both versions.
The manuscript's existing intelligence-by-cloud interaction changes from
(h_{A+}-h_{B+}=-0.55) to (h_{A-}-h_{B-}=+0.55), giving a clean odd response.

This is a substantive feasibility result: the proposed test can be expressed
using the paper's own device attributes and its own nonlinear data-generating
mechanism. `analysis/smart_device_fibre_design.py` verifies the equalities and
runs a 200-replication A/B/opt-out check. Rejection is .025 under the additive
null and .995 with the interaction active at 400 respondents. The design note
is in `manuscript/smart_device_parity_design.md`. The result remains a planning
simulation until the paired task is fielded.

## Round-eleven boundary update

A closer 2024 Journal of Econometrics neighbor uses transformation-group
invariance to difference out latent unobservables. The current paper should not
claim a general group-invariance theory. Its bounded contribution remains an
observed attribute-space DCE instrument: a candidate-preserving fibre, full-menu
prediction preservation, odd/even response separation, and pre-outcome task
optimization. The collision and revised safe/unsafe novelty sentences are
recorded in `manuscript/innovation_search_round11.md`.
