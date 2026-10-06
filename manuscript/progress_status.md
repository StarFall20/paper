# Revision progress status

## Current decision

The independent innovation gate is passed at the conceptual and simulation
levels. The main paper will be rebuilt around **Behavioral Metamorphic
Specification Testing (BMST)** and its **Utility-Fibre Invariance Test
(UFIT)** relation. BMST freezes a candidate utility basis, constructs
candidate-equivalent task versions, and tests their observed choice-probability
equivalence with one-member assignment and respondent-cluster randomization.
The earlier XGBoost-versus-MNL crossover and the model-family triage remain
controlled benchmarks. They no longer carry the main contribution claim.

The full submission gate remains open until a paired-task supplement or a
dataset with the same candidate-preserving assignment structure is obtained.
This is an evidence requirement for the selected innovation, not a request for
another conceptual route.

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
- Completed a fourth-round literature and identification audit. The transportable-utility prototype is retained as exploratory work only because its penalty was target-tuned and its raw coefficient stability criterion confounds logit scale. UFIT remains the leading candidate under a narrower candidate-basis sufficiency claim; a formal proposition and paired-task instrument are now explicit submission gates.
- Added the BMST negative-control benchmark. In 100 replications, rejection is .04 for the additive null, .68 for omitted nonlinearity, .03 for random linear taste, .23 for task-specific scale drift, and 1.00 for order effects. The result keeps mechanism attribution separate from relation violation.
- Reworked the primary supplement protocol to assign one member of each focal pair per respondent. The 600-respondent, 100-replication assignment benchmark gives .05 null rejection, .79 nonlinear power, .05 random-taste rejection, and 1.00 interaction power.
- Completed a fifth-round cross-literature novelty audit. Daly's indistinguishability and scale boundary is now explicit; BMST is limited to a declared choice-probability relation on a tested task fibre.
- Added a 500-replication proof-of-concept for the paired-task test. The additive null rejection rate is 0.06 and the omitted-quadratic rejection rate is 0.986 at the declared 5% level. This is feasibility evidence; it does not yet establish empirical validity.
- Audited a public repeated-task dataset from TUDelft as a possible empirical BMST source. Its documented schema and random task construction do not provide the one-member-per-pair assignment needed for the primary estimand. The audit is in `manuscript/public_data_pair_audit.md`; the paired-task supplement remains the empirical gate.
- Completed a sixth-round independent innovation audit against recent JOCM directions. BMST/UFIT is retained as one coherent candidate-fibre exchangeability test; conformal prediction, welfare model averaging, adversarial stress testing, and environment-invariant WTP are recorded as comparison or secondary routes. The original XGBoost-versus-MNL crossover is demoted to a controlled benchmark.
- Completed a seventh-round stress audit prompted by the comparison-complexity objection. The main claim is now restricted to nontrivial fibres of coarsened candidate bases. A zero-candidate-difference anchor and a geometry-preserving common translation control the symmetric-scale and tradeoff-complexity channels. The new 100-replication anchored-fibre benchmark shows .04/.04 rejection for observational LR and residual-learning comparators under an omitted decomposition, versus .83/.56/1.00 for the anchored arms; the complexity-only control is concentrated in the geometry-changing nonzero-gap arm. A 60-replication sample-size curve is committed for planning.

## In progress

- Expand the benchmark to the target replication count.
- Regenerate the primary and latent-class benchmarks with the Random Forest-assisted selector and report term recovery metrics.
- Upgrade the targeted Mixed Logit check to a full random-coefficient model with WTP recovery and convergence diagnostics.
- Run exact XGBoost in an environment with a working OpenMP runtime.
- Add coefficient recovery, calibration, term stability, parsimony, and sample-size/task-count sensitivity analyses.
- Add candidate-library coverage, recoverability-ceiling, formal heterogeneity-gate, and permutation-robustness experiments before claiming a general decision rule.
- The public Swissmetro file has been audited: it has 10,728 rows for 1,192 IDs and only eight exact repeated full profiles within an ID, concentrated in two IDs. That structure is insufficient for a clean paired-task empirical test. A paired-task supplement or a different dataset is now a hard requirement for promoting the pivot to the main empirical contribution.
- An observational fallback is specified in `analysis/audit_swissmetro_pairs.py`: freeze the candidate utility coordinates on development data, match within-respondent held-out tasks at a declared tolerance, and use a clustered bootstrap. The public file yields 26, 85, 535, and 1,924 candidate pairs at tolerances 0.01, 0.02, 0.05, and 0.10. This route remains weaker than randomized pairs and needs its own size and power study.
- The observational fallback has now been implemented in `analysis/observational_equivalence_test.py` with respondent-level cross-fitting, availability-aware MNL probabilities, deterministic covariate-only pair orientation, and a respondent-cluster multiplier bootstrap. In a 40-replication boundary run, the null rejection rate is 0.050; rejection rates are 0.225 for the engineered nonlinear condition and 0.175 for the engineered interaction condition. The cross-fitted public Swissmetro run yields 126, 236, 1,371, and 3,430 pairs at tolerances 0.01, 0.02, 0.05, and 0.10. The row-order orientation gives p-values 0.427, 0.440, 0.430, and 0.477; time orientation gives 0.430, 0.387, 0.003, and 0.033; cost orientation gives 0.130, 0.460, 0.847, and 0.977. This orientation sensitivity is a failure boundary, so the fallback remains an explicit feasibility limit, not the main empirical contribution.
- The exact randomized three-alternative benchmark is implemented in `analysis/exact_paired_task_benchmark.py`. Its 50-replication run gives 0.06 rejection under the additive null, 0.84 against omitted nonlinearity, and 1.00 against threshold and interaction conditions. The damped-Newton oracle repair reduces rejection to 0.04, 0.02, and 0.02 for nonlinear, threshold, and interaction conditions. The repair gate passes for detection. A four-cell probability-scale factorial audit found 0.28 rejection for a pure interaction and 0.86 when an interaction coexists with a nonlinear main effect, so term-level localization is excluded from the claim. Details are in `manuscript/factorial_contrast_note.md`.
- The cluster randomization audit is implemented in `analysis/randomization_paired_task_test.py`. Its 50-replication run gives 0.06 null rejection, 0.82/1.00/1.00 rejection against nonlinear/threshold/interaction conditions, 0.08/0.02/0.04 after the corresponding repair terms, 0.06 for random cost sensitivity, and 0.74/1.00 power when nonlinear or interaction terms are combined with random cost sensitivity. This is the sharper candidate-preserving test. Its finite-sample validity requires held-out pairs, exchangeability, no carryover, and coefficient-wise utility-difference preservation. The result does not establish a unique omitted term.
- The randomization output now includes all-pair and predeclared time-, cost-, and joint-shift rows. The shift-specific patterns are directional and remain explicitly non-identifying for a unique omitted term.
- Added a deterministic candidate-preserving design-score audit and a finite-sample fixed-versus-maximin benchmark. The constrained maximin design raises the normalized weakest probe score from 1.142 to 1.632, keeps null rejection at 0.06, and raises nonlinear power from 0.82 to 1.00. The unconstrained repeated-shift solution is retained as a documented failure boundary. A cubic out-of-library condition is now included in the frozen randomization output.
- Added and audited the Mechanism-Fingerprint Choice Audit as a bounded extension to the Utility-Fibre Invariance Test. UFIT is the independent core: a frozen candidate is tested on held-out task pairs with equal candidate utility differences and different attribute decompositions. MFCA adds common-shift, presentation, and sequence relations with shared-sign maxT adjustment. In 50 replications, null rejection is 0.02/0.00/0.00; nonlinear, position-bias, and inertia conditions produce the intended dominant signatures in 39/50, 49/50, and 49/50 replications. Combined mechanisms remain mixed or unresolved. The overlap audit records why MFCA is not a wholly new order-effect theory, and a three-relation empirical supplement remains required.
- A first score-overdispersion prototype is now implemented. It flags all continuous-heterogeneity replications and 83% of combined replications at an exploratory threshold while producing a 3% false-positive rate in the additive condition. Bootstrap calibration is still required.

## Pending before submission

- Run the locked workflow on LPMC with a final-year temporal holdout.
- Run the locked workflow on Swissmetro with respondent-grouped validation.
- Freeze the final result tables and rewrite the complete manuscript around them.
- Complete declarations, data availability, author information, and the final public release commit.

## Submission gate

The paper is in an advanced revision scaffold. It is not submission-ready until the in-progress and pending evidence blocks are complete and the manuscript, tables, declarations, and repository refer to the same frozen version.
