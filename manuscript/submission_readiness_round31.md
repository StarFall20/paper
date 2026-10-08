# Journal of Choice Modelling submission readiness round 31

## Current decision

The recommended manuscript is a finite-support, design-based audit of utility-basis
sufficiency. The central object is the relation Y ⟂ X | φ(X), where φ is the
complete candidate menu summary. The manuscript reports the support rank,
minimum exposure eigenvalue, and support-repair rule before outcomes are
generated. The finite score is explicitly compared with a saturated fibre LR;
the paper claims the design certificate and its use for DCE construction.

The counterfactual policy-path oracle and Swissmetro specification comparison are
bounded controls. They show why model agreement is not a counterfactual
certificate, but they are not separate headline contributions.

## Gates already passed

- Finite theorem and explicit distinction between structural visibility and
  statistical power.
- Full-menu preservation in the multinomial construction.
- Exact 3 × 3 candidate-precision counterexample: variance 4.00, exposure rank 0.
- Support-complete design recheck: rank 4 and minimum eigenvalue 0.1222.
- Smart-device reflected-task pool: rank 3 and a finite support-repair result.
- Thirty-replication hidden-direction benchmark and saturated-LR boundary check.
- Public Swissmetro implementation check with respondent grouping.
- Word manuscript generated and rendered; all five pages visually inspected.
- Code, round-31 CSVs, innovation memo, and progress memo prepared for GitHub.

## Remaining gates

1. Update the repository manifest and commit the final tracked manuscript,
   scripts, results, and scope memo.
2. Verify the GitHub main tree contains the same manuscript, code, and CSV
   hashes as the local release.
3. Add a concise cover letter and data-availability statement that describe the
   work as a method-and-design study with a public-data implementation check.
4. Run the final smoke checks with the bundled runtime and record any missing
   optional dependencies.
5. Keep the human-data claim open. A randomized candidate-preserving DCE is
   required before claiming a rejection from human responses; no questionnaire
   is needed for the current computational submission package.

## Claims to avoid

Do not claim a new universal lack-of-fit test, global nonparametric
identification, a first conditional-randomization method, a causal mechanism
from Swissmetro, or a distinct inferential family beyond the saturated fibre
alternative. Do not treat a positive eigenvalue as proof of power without an
effect size and sample-size calculation.

## Submission package

The minimum coherent package contains:

- JOCM_fibre_sufficiency_revision.docx;
- the finite-support design, benchmark, and saturated-LR scripts;
- the smart-device structural outputs;
- the Swissmetro provenance and implementation output;
- the innovation decision memo and this checklist;
- a manifest and one reproducible smoke-check command.
