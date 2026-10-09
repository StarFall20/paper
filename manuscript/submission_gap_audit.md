# Submission-gap audit after the anchored-fibre redesign

## Objective verdict

The original ML-assisted model-search claim does not clear the Journal of
Choice Modelling novelty bar. Recent work already covers assisted utility
search, flexible hypothesis testing, and data-driven specification support.
The broader BMST claim also needed a sharper domain: a full raw-attribute
linear basis can have singleton fibres, and a joint rejection can mix omitted
utility, scale, task complexity, framing, and order effects.

The defensible contribution is now narrower and more testable:
**anchored behavioural metamorphic specification testing for a coarsened
utility basis**. A candidate basis \(b(x)\) is eligible only when its full menu
difference vector

\[
\phi(x)=\{b_j(x)-b_k(x):j<k\}
\]

has a nontrivial feasible fibre. The design randomizes one member of each
candidate-equivalent pair, includes a zero-gap anchor, and uses a common raw
translation to hold pairwise geometry fixed. A geometry-changing arm reports
comparison-complexity sensitivity as a separate result. This is one coherent
specification audit with declared controls, not a collection of term-search
modules.

## Current distance from the submission gate

| gate | current state | status |
|---|---|---|
| distinct scientific question | candidate-conditioned equivalence on a nontrivial fibre | passed conceptually |
| scope and boundary | coarsened basis only; full menu vector; singleton fibres excluded | passed |
| scale/complexity separation | zero-gap anchor, geometry-preserving translation, geometry-changing arm | simulation passed |
| observational comparator | LR and out-of-fold residual proxy on near-flat omitted support | simulation passed |
| sample-size planning | power curve for one and three focal pairs | passed as planning evidence |
| empirical DCE | no current public file has the required randomized one-member assignment | open |
| manuscript | core draft and audit rewritten; Word manuscript still needs full alignment | open |
| reproducibility | scripts, locked CSVs, manifest, and search log synchronized to GitHub | passed |

The project is therefore at a strong **method-and-design candidate** stage, not
submission-ready. The open item is an empirical instrument that actually
creates the declared fibres. A public panel with approximate repeated profiles
cannot be relabelled as this experiment.

## Locked planning evidence

The anchored benchmark uses 100 replications, 400 respondents, and 199
respondent-cluster sign flips. Under an omitted decomposition direction,
observational LR and the residual proxy reject at .04 and .04, whereas the
zero-gap geometry-preserving arm rejects at .83 and the geometry-changing
nonzero-gap arm at 1.00. Under a complexity-only condition, the zero-gap arms
remain near size (.02 and .07), while the deliberately geometry-changing
nonzero-gap arm responds at .41. These are planning results, not evidence from
human respondents.

The sample-size curve uses 60 replications and 49 sign flips per replication.
For the nonlinear condition, power is .20, .42, and .70 at 150, 300, and 600
respondents for one focal pair; the three-pair design gives .90, 1.00, and
1.00. Additive rejection remains below .08. The empirical protocol should use
the three-pair design unless piloting shows unacceptable burden.

## Remaining empirical gates

1. Pre-register a coarsened candidate basis, feasible fibre grid, full menu
   \(\phi\), pairwise geometry index, zero-gap anchor, translation arm,
   decomposition arm, assignment coin, split rule, statistic, and sign-flip
   reference.
2. Collect a forced-choice supplement or locate a dataset with documented
   one-member-per-pair assignment. Define the opt-out profile if the test is to
   include product-versus-opt-out differences.
3. Report additive, random-taste, order/carryover, and complexity controls.
   A zero-gap rejection is a relation violation; it is not by itself proof of
   an omitted utility term.
4. Compare the designed test with the observational LR and residual methods on
   the same respondent budget. Show the direction in which the ordinary
   diagnostics are weak and the designed probe is informative.
5. Refit the complete manuscript around this claim. Keep the earlier ML and
   process analyses as comparison and failure-boundary material.

## Stop conditions

Drop the empirical novelty claim if the fibre is singleton, the assignment is
not randomized, the zero-gap placebo rejects under an additive process, or the
geometry control is not measured. If the supplement cannot be fielded, release
the work as a simulation and design paper and state that the human-data gate is
open.

## Closest literature setting the bar

- Assisted specification: https://doi.org/10.1016/j.jocm.2021.100285
- MNL misspecification tests: https://doi.org/10.1016/j.jocm.2024.100531
- Random cost-attribute split in forced and unforced choices:
  https://doi.org/10.1016/S1755-5345(13)70044-7
- Comparison complexity and tradeoffs: https://arxiv.org/abs/2401.17578
- Task complexity in stated choice: https://doi.org/10.1016/j.tre.2022.102744

The search boundary and index-access limitations are recorded in
\`manuscript/index_search_log_round7.md\`.
