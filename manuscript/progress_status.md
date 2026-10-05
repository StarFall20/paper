# Revision progress status

## Completed

- Reframed the paper around ML-assisted utility specification for the Journal of Choice Modelling.
- Added a novelty and journal-fit audit; the paper is now positioned as a mechanism-controlled validation benchmark rather than a new generic specification algorithm.
- Corrected the data-generating process with an explicit opt-out indicator and unobserved respondent-specific price sensitivity.
- Locked the corrected 30-replication benchmark as the current primary simulation scaffold.
- Added two-class Latent Class MNL, grouped Random Forest, and boosted-tree proxy extensions.
- Added a targeted random-price Mixed Logit extension and documented its boundary interpretation.
- Replaced the original MNL-only candidate selector with a respondent-level Random Forest diagnostic followed by nested behavioural refitting.
- Added paired-draw Monte Carlo stability checks and an explicit framework/model audit.
- Fixed opt-out price leakage in the data generator and regenerated the primary, latent-class, tree, and Mixed Logit result files.
- Added mechanism-focused results prose, methods and writing benchmarks, data provenance instructions, a smoke test, and a SHA-256 results manifest.
- Added a deeper innovation gate that ranks recoverability-aware model-family triage, choice-set-aware diagnostics, and policy-loss-constrained selection; the current RF selector also received a preliminary product-alternative permutation audit.
- Synchronized the reproducibility repository with GitHub and rendered the Word working draft for visual review.
- Applied the Idea-DNA workflow from the public Idea Generator skill and completed an independent innovation audit. The selected high-risk pivot is a model-equivalent choice-pair specification test; adaptive design, permutation invariance, reliability, and standalone process models were stopped as overlapping routes.
- Added a 500-replication proof-of-concept for the paired-task test. The additive null rejection rate is 0.06 and the omitted-quadratic rejection rate is 0.986 at the declared 5% level. This is feasibility evidence; it does not yet establish empirical validity.

## In progress

- Expand the benchmark to the target replication count.
- Regenerate the primary and latent-class benchmarks with the Random Forest-assisted selector and report term recovery metrics.
- Upgrade the targeted Mixed Logit check to a full random-coefficient model with WTP recovery and convergence diagnostics.
- Run exact XGBoost in an environment with a working OpenMP runtime.
- Add coefficient recovery, calibration, term stability, parsimony, and sample-size/task-count sensitivity analyses.
- Add candidate-library coverage, recoverability-ceiling, formal heterogeneity-gate, and permutation-robustness experiments before claiming a general decision rule.
- The public Swissmetro file has been audited: it has 10,728 rows for 1,192 IDs and only eight exact repeated full profiles within an ID, concentrated in two IDs. That structure is insufficient for a clean paired-task empirical test. A paired-task supplement or a different dataset is now a hard requirement for promoting the pivot to the main empirical contribution.
- A first score-overdispersion prototype is now implemented. It flags all continuous-heterogeneity replications and 83% of combined replications at an exploratory threshold while producing a 3% false-positive rate in the additive condition. Bootstrap calibration is still required.

## Pending before submission

- Run the locked workflow on LPMC with a final-year temporal holdout.
- Run the locked workflow on Swissmetro with respondent-grouped validation.
- Freeze the final result tables and rewrite the complete manuscript around them.
- Complete declarations, data availability, author information, and the final public release commit.

## Submission gate

The paper is in an advanced revision scaffold. It is not submission-ready until the in-progress and pending evidence blocks are complete and the manuscript, tables, declarations, and repository refer to the same frozen version.
