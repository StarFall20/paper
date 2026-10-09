# Journal of Choice Modelling submission gate — round 33

## Current decision

The manuscript has a coherent finite-support design contribution and a reproducible simulation package. It is not marked submission-ready yet. A randomized human candidate-preserving DCE has not been fielded, and an independent mathematical and editorial review remains necessary.

## Completed gates

- The 3 × 3 counterexample reports candidate-index precision separately from within-fibre exposure. The support-complete rank is 4 with minimum positive eigenvalue 0.1222; the endpoint precision design has rank 0.
- The manuscript now positions the method against Atkinson–Fedorov model-discrimination designs, model-robust/model-sensitive design, compound lack-of-fit criteria, and binary-response lack-of-fit designs.
- The primary finite-support comparator has 500 replications, n = 400, and four methods: unrestricted-fibre support score, all-profile generalized CMH, unrestricted complete-fibre LR, and held-out ML residual score. Null rates are 0.046, 0.046, 0.046, and 0.042; the ML column is exploratory even with 499 draws because it is a frozen predictive learner.
- The power curve varies η and n. The strong-signal ceiling result is separated from the power-planning evidence.
- The continuous extension states an efficient-information proposition with the Fisher-information coefficient κ, the coarea conditional law, and an explicit generalized-cost example. Discrete finite support remains the primary result.
- Swissmetro, Electricity, and Train vehicle are labelled external implementation checks. None is used as human evidence for candidate sufficiency. Swissmetro values are 0.8045 and 0.7686 throughout the current manuscript.
- The questionnaire has respondent-facing Chinese and English files, two randomized versions, and a 16-card candidate-preservation verifier. Researcher-only deployment and analysis instructions are in `manuscript/candidate_preserving_dce_fielding_protocol.md`.

## Blocking gates before submission

1. Complete an independent mathematical review of Proposition 1, including the regularity conditions at support boundaries and the relation between fibre conditional laws and the actual DCE assignment mechanism.
2. Run a small comprehension pilot for both questionnaire versions. Confirm that the attribute descriptions are understood and that version assignment does not change perceived task difficulty or opt-out use.
3. Decide whether to submit as a design-method paper with simulations or wait for the randomized pilot. Do not present Swissmetro or the two public DCE files as a sufficiency validation.
4. Build and test the clean release archive from the manifest. It must contain the dependency closure of the manuscript entry points and exclude exploratory notes, stale binaries, and scripts that cannot be executed from a fresh checkout.
5. Replace placeholders for research team, privacy contact, and ethics reference in the respondent instrument before any fielding.

The earlier round-32 statement that the package was “ready” was a self-assessment and is superseded by this gate record.
