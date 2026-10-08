# Progress status round 31

## Current position

The project has passed the major computational correction stage. The direct
policy-path oracle audit is complete at n=400 and n=800, the population
heterogeneity benchmark uses quadrature integration, and the exact common-mode
witness is saved. A separate finite-support audit has now been rechecked and
selected as the stronger primary contribution: a support-complete fibre
certificate for a coarsened utility representation.

## Verified this round

- Support-complete design recheck: rank 4 and minimum exposure eigenvalue
  0.1222 on the declared \(3\times3\) support.
- Candidate-precision endpoint design: candidate-summary variance 4.00 and
  support-complete rank 0.
- Thirty-replication benchmark: hidden direction rejection 1.000 for the
  support-complete score versus 0.067 for the hand-written dictionary; null
  rejection 0.033 versus 0.133.
- Full-menu algebra check: the smart-device reflected-task pool has full
  declared rank 3.
- Saturated-LR boundary check: the support-complete score and the saturated
  fibre LR use the same finite conditional alternative and show the same
  rejection pattern. The manuscript now treats the certificate and design
  repair rule as the increment, not the LR statistic itself.
- Independent innovation decision memo added as
  `manuscript/innovation_decision_round31.md`.
- New support-complete Word manuscript generated and rendered; all five pages
  passed visual inspection. The draft is
  `manuscript/JOCM_fibre_sufficiency_revision.docx`.
- A markdown companion is included for repository review. The new manuscript,
  round-31 outputs, innovation memo, readiness checklist, and manifest were
  written to GitHub main through the connected GitHub integration and fetched
  back byte-for-byte for verification. The binary Word file remains the local
  submission artifact; the markdown companion is the repository-readable copy.

## Release status

The no-questionnaire computational package is complete. The Word manuscript,
markdown companion, cover letter, data-availability statement, release
manifest, innovation decision memo, readiness checklist, scripts, and round-31
outputs are present locally. The connected GitHub main branch contains the
repository-readable files, and each was fetched back byte-for-byte against the
local copy.

The legacy all-in-one smoke script stops before the current fibre modules
because this runtime does not include scikit-learn, which the older ML-assisted
entry point imports. The support-complete design recheck, 30-replication
benchmark, saturated-LR boundary check, smart-device rank check, and policy
oracle checks run independently with the bundled NumPy/SciPy runtime. The
missing optional dependency is recorded as an environment limitation rather
than a failure of the primary release.

No required work remains within the current scope. A randomized
candidate-preserving DCE or a second public DCE would be a future empirical
extension and would add about one working day.
