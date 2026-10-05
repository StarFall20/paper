# Revision progress status

## Completed

- Reframed the paper around ML-assisted utility specification for the Journal of Choice Modelling.
- Corrected the data-generating process with an explicit opt-out indicator and unobserved respondent-specific price sensitivity.
- Locked the corrected 30-replication benchmark as the current primary simulation scaffold.
- Added two-class Latent Class MNL, grouped Random Forest, and boosted-tree proxy extensions.
- Added a targeted random-price Mixed Logit extension and documented its boundary interpretation.
- Added paired-draw Monte Carlo stability checks and an explicit framework/model audit.
- Added mechanism-focused results prose, methods and writing benchmarks, data provenance instructions, a smoke test, and a SHA-256 results manifest.
- Synchronized the reproducibility repository with GitHub and rendered the Word working draft for visual review.

## In progress

- Expand the benchmark to the target replication count.
- Upgrade the targeted Mixed Logit check to a full random-coefficient model with WTP recovery and convergence diagnostics.
- Run exact XGBoost in an environment with a working OpenMP runtime.
- Add coefficient recovery, calibration, term stability, parsimony, and sample-size/task-count sensitivity analyses.

## Pending before submission

- Run the locked workflow on LPMC with a final-year temporal holdout.
- Run the locked workflow on Swissmetro with respondent-grouped validation.
- Freeze the final result tables and rewrite the complete manuscript around them.
- Complete declarations, data availability, author information, and the final public release commit.

## Submission gate

The paper is in an advanced revision scaffold. It is not submission-ready until the in-progress and pending evidence blocks are complete and the manuscript, tables, declarations, and repository refer to the same frozen version.
