# Innovation gate for the JOCM submission

## Independent pivot after the Idea-DNA search

The preferred new direction is a **model-equivalent choice-pair test**. The
analyst creates two randomized choice tasks with the same candidate-model
utility-difference vector and different raw attribute decompositions. The
candidate model implies equal choice probabilities for the pair. A bootstrap-
calibrated share difference becomes a direct specification test. Pair labels
can stratify the violation by a predeclared transformation, but they do not
identify a unique omitted term. The first 500-replication prototype gives a
0.06 rejection rate under
the additive null and 0.986 under an omitted quadratic term.

This pivot is a separate design-based object. It does not combine the Random
Forest selector with the process gate. Its submission status depends on a
paired-task instrument: Swissmetro must contain usable matched tasks, or the
study needs a small paired-task supplement. Until that condition is verified,
the pivot remains a tested candidate and the recoverability benchmark remains
the empirical baseline.

The overlap audit lowers the confidence of any priority claim. Fok and Paap's
JOCM misspecification tests already use alternative pairs and overidentifying
moments. The present candidate differs through cross-task equality of utility
difference vectors and attribute decompositions, but that distinction must be
stated precisely and checked against the full specification-test literature.
The observational fallback now controls null size near 5% but has modest power
and no descriptive rejection on public Swissmetro. It remains a feasibility
audit until an exact randomized supplement is available.

The exact three-alternative benchmark now supplies the randomized supplement
in simulation. It has strong base power, and the damped-Newton repair returns
rejection to the declared size after the corresponding term is added. The
joint shift still changes multiple raw loci, so interaction localization is
incomplete. A four-cell probability-scale factorial contrast was audited as a
repair; it fails when an interaction coexists with a nonlinear main effect.
The extension is excluded from the primary claim. The supported object is a
test for candidate-invariance violation, with no unique omitted-term label.

## Objective assessment

The current paper has a credible question and a corrected simulation scaffold, but its original innovation claim is too small for a strong JOCM submission. A generic statement that machine learning can screen nonlinearities and interactions is already close to assisted specification, random-forest-assisted portfolio choice, extensive mixed-Logit hypothesis search, model-based recursive partitioning, latent-class neural networks, reinforcement-learning specification, and LLM-supported specification. The current results show that the workflow can approach an oracle structured MNL in a combined synthetic condition. That is useful evidence, but it is not a new method by itself.

The publishable contribution should be a falsifiable decision framework that explains when a diagnostic can recover observable utility structure, when the candidate library is insufficient, and when the analyst must change the model family to represent heterogeneity. The paper should make the operating boundary the object of study.

## What strong recent papers do

Recent high-quality choice-modelling papers generally combine four elements:

1. A clearly defined model, estimator, or diagnostic object. Examples include the Smooth Bounded Choice Model, model-based recursive partitioning with Mixed Logit leaves, and ANN-based latent-class choice models.
2. A formal behavioural or computational property that the method is designed to preserve or improve, such as smooth probabilities, parameter-stability tests, interpretable segmentation, or WTP recovery.
3. A controlled simulation in which the proposed method can succeed and fail for identifiable reasons.
4. An empirical application or external dataset showing that the mechanism observed in simulation matters in practice.

The repository and manuscript should follow the same evidence architecture. The replication repository for MOB-MIXL is a useful implementation benchmark because it separates source code, simulation replication, empirical replication, installation instructions, and a worked example.

## Recommended contribution: recoverability-aware model-family triage

### Central claim

The paper develops and validates a recoverability-aware triage rule for choice-model specification. The rule uses grouped data diagnostics to decide among an enriched MNL, a discrete-segmentation model, a continuous-heterogeneity model, and candidate-library expansion. It reports the expected predictive gain, behavioural recovery, WTP distortion, calibration, and policy regret for each branch.

### Why this clears the novelty gate

The contribution is a tested mapping from data-generating mechanism to model family. It is distinct from a new search heuristic because the scientific output is the boundary of recoverability and the consequence of choosing the wrong model family. It is also distinct from a pure ML-versus-MNL comparison because every branch is evaluated using behavioural and policy quantities.

### Minimum implementation

- Expand the mechanism grid to include observable nonlinear terms, thresholds, omitted interactions, discrete classes, continuous random coefficients, and out-of-library mechanisms such as cubic, spline, or unlisted interactions.
- Add a candidate-coverage factor. For each condition, record whether the true mechanism is represented in the candidate library, partially represented, or absent.
- Fit additive MNL, enriched MNL, RF-assisted MNL, Latent Class MNL, Mixed Logit, and a flexible learner under the same respondent-grouped validation budget.
- Define a pre-registered triage rule using only development data. The rule chooses a model family before test outcomes are observed.
- Evaluate triage accuracy, term recovery, coefficient and WTP recovery, calibration, choice-share error, decision regret, model complexity, and stability across replications.
- Repeat the rule on LPMC temporal holdout and Swissmetro respondent-grouped validation.

### Main figures and tables

1. A mechanism-to-model decision map showing the selected model family and its regret relative to the oracle model.
2. A recoverability curve showing how term recovery changes with sample size, task count, signal strength, and candidate coverage.
3. A policy table showing how wrong specification changes WTP, elasticities, market shares, and decision regret.
4. An external-transfer table showing whether the triage rule survives temporal and cross-dataset validation.

## Stronger optional contribution: policy-loss-constrained specification

The selection stage can be upgraded from fit-only screening to a constrained objective that penalizes predictive loss, behavioural instability, WTP distortion, policy regret, and unnecessary terms. The simulation can measure the true policy loss; empirical data can use calibration, bootstrap stability, and externally validated policy predictions. This would create a method contribution, but it requires a formal objective, an estimation algorithm, and a fair comparison with the Pareto-search method of Ortelli et al. It should be attempted only after the triage benchmark is complete.

## Additional high-value innovations found in the audit

### 0. Process-aware triage with an explicit unresolved branch

Process heterogeneity is already a mature JOCM topic. Existing papers model
attribute non-attendance, saliency, inertia, non-trading, and disjunctive
decision rules. Adding one of those mechanisms alone would duplicate that
literature. The stronger route is to make process heterogeneity a branch in a
recoverability-aware triage rule and to add an explicit abstention outcome.

The rule should distinguish three signals: (i) observable utility-form error,
(ii) respondent-level score dispersion, and (iii) sequence residuals. A high
score-dispersion statistic can arise from random taste sensitivity or
attribute non-attendance. If no attention or process indicator separates them,
the rule should report an unresolved mechanism and request additional evidence
instead of forcing a Mixed Logit or ANA label. This is the key method-transfer
opportunity from selective prediction: the model-selection system can decline
an unsupported behavioural interpretation.

The exploratory process prototype supports this design. In 30 respondent-
grouped replications, random price sensitivity and price non-attendance both
produced high price-score dispersion and near-zero sequence residuals, while
inertia produced a positive sequence residual. The combined condition produced
both signals. The overlap between random taste and non-attendance is a useful
failure boundary, not a result to hide. Details and raw outputs are in
`manuscript/process_gate_note.md` and `results/process_gate_30rep.csv`.

### 1. A recoverability ceiling for observed-term diagnostics

The current heterogeneity condition can support a formal distinction between approximation error and latent-heterogeneity error. When all candidate terms are functions of observed attributes, a selector can improve the observed utility component but cannot identify respondent-specific random coefficients that are independent of those attributes. Define the recoverability ceiling as the performance of the best candidate-term model under the true observed component, and report the residual policy regret relative to the true random-coefficient model. The paper can then test whether the ceiling predicts the point at which MNL enrichment should stop and Mixed Logit should begin.

This is stronger than reporting that RF performs poorly under heterogeneity. It turns the failure into a model-selection quantity that can be estimated, stress-tested, and transferred to empirical data through residual and stability diagnostics.

### 2. A choice-set-aware diagnostic audit

The current RF stage flattens each choice task into one row with alternative blocks. That preserves the choice set in the input, but it leaves the diagnostic vulnerable to alternative-position effects and does not enforce permutation equivariance. A stronger version should compare the current RF with a choice-set-aware diagnostic built from utility differences, alternative permutation augmentation, or a DeepSets-style invariant architecture. The key test is whether candidate ranking and term recovery remain unchanged when product alternatives are permuted while the opt-out alternative is kept distinct.

This creates a concrete methodological question: does a learner that ignores the structural symmetry of choice sets produce unstable or misleading utility diagnostics? The permutation-invariance literature and open replication repositories show that this is a recognized structural issue in modern choice learning. The audit should be a robustness layer unless the invariant diagnostic is developed, theoretically justified, and shown to improve recovery.

### 3. A formal heterogeneity gate before model expansion

Add a score or residual-based gate before fitting Latent Class or Mixed Logit. The gate tests whether respondent-level score contributions or out-of-fold residuals show systematic instability across tasks. Its decision can be compared with the true DGP in simulation and with the full model-family search in external data. The 2025 JOCM work on Lagrange-multiplier tests shows that formal heterogeneity diagnostics are a current methodological direction. Combining a formal heterogeneity gate with RF term discovery would give the workflow a behavioural test stage and a flexible functional-form stage.

The gate should be judged by false expansion, missed heterogeneity, WTP distortion, and computational savings. A model that expands to Mixed Logit for every dataset is not a decision rule.

### Prototype evidence

The first score-overdispersion prototype already separates the current mechanisms. After fitting the structured MNL, the mean score ratio is 1.03 in the additive condition, 1.07 in the nonlinear condition, 1.05 in the interaction condition, 1.89 in the continuous-heterogeneity condition, and 1.50 in the combined condition. Using an exploratory threshold of 1.30, the gate flags 100% of heterogeneity replications and 83% of combined replications, while flagging 3% of additive replications. These values are development evidence only; the threshold must be calibrated by a parametric bootstrap before it enters the main paper. The raw diagnostics are in `results/heterogeneity_gate_30rep.csv` and the summary is in `results/heterogeneity_gate_summary_30rep.csv`.

## Innovation ranking after the deeper audit

1. **Recommended and feasible:** recoverability-aware model-family triage with candidate coverage, formal heterogeneity gate, policy regret, and external transfer.
2. **Strong methodological addition:** choice-set-aware and permutation-robust candidate ranking.
3. **Potentially publishable extension:** policy-loss-constrained specification with a declared multi-objective criterion.
4. **High-risk pivot:** a new constrained neural or lattice choice model. This space already contains domain-constrained flexible DCMs and would require a new estimator, theory, and extensive benchmarking.

The first route can be built from the current project. The second route directly addresses a structural weakness in the current RF implementation. The third route can follow after the evidence pipeline is stable. The fourth route would replace the paper's research question and is not the efficient path for this revision.

## Preliminary self-audit result

A paired product-alternative permutation check was run on the current selector while keeping the opt-out position fixed and remapping the chosen label. The core selected terms were stable in the interaction, combined, and heterogeneity conditions. The nonlinear condition produced one additional false-positive term under one permutation. This does not establish a fatal implementation error, but it confirms that term-stability and permutation robustness must be reported. The audit should become a formal appendix experiment before the selector is described as structurally reliable.

## Routes that should not be the main innovation

- Adding another off-the-shelf learner such as XGBoost without a new estimand or decision rule.
- Reporting only higher accuracy than MNL. Recent work shows that flexible learners can match or exceed prediction while producing weak behavioural interpretation.
- Presenting WTP recovery alone as the novelty. Recent constrained DCM work already treats WTP interpretability as a central benchmark.
- Calling a larger simulation an innovation. Larger replication counts improve credibility but do not create a contribution by themselves.
- Treating LPMC and Swissmetro as two routine case studies. They are necessary transfer tests; the novelty lies in whether the same triage rule transfers.

## Submission gate

The manuscript has a defensible innovation claim only when all of the following are present:

- the triage rule is defined before test evaluation;
- the candidate-coverage and out-of-library stress tests are included;
- the model-family branches include full Mixed Logit and a documented flexible-learner baseline;
- coefficient, WTP, calibration, regret, and stability outcomes are reported;
- the rule is tested on at least one external dataset or temporal holdout;
- the introduction distinguishes the paper from assisted search, recursive partitioning, latent-class neural networks, reinforcement learning, and LLM specification;
- the repository contains one-command replication, data provenance, frozen tables, and a release commit.

## Literature and implementation references

- JOCM aims and scope: https://shop.elsevier.com/journals/journal-of-choice-modelling/1755-5345
- Assisted specification: https://doi.org/10.1016/j.jocm.2021.100285
- Association rules and random forests: https://doi.org/10.1016/j.jocm.2022.100397
- Extensive Mixed-Logit hypothesis testing: https://doi.org/10.1016/j.jocm.2023.100409
- Model-based recursive partitioning: https://doi.org/10.1016/j.jocm.2022.100393
- ANN latent-class choice models: https://doi.org/10.1016/j.jocm.2023.100452
- Smooth Bounded Choice Model: https://doi.org/10.1016/j.jocm.2025.100574
- Model choice and WTP-related behavioural consequences: https://doi.org/10.1016/j.jocm.2024.100520 and https://doi.org/10.1016/j.jocm.2024.100524
- Monotonic flexible DCM and WTP interpretability: https://doi.org/10.1016/j.trb.2024.102947
- Replication repository example: https://github.com/alvarogutyerrez/mobmixl
