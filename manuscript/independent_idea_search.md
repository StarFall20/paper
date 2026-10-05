# Independent innovation search

## Scope and locks

This search treats the current Random Forest-assisted specification and the
process-aware triage prototype as existing baselines. They are not ingredients
to be combined into a new claim. A candidate survives only if it has one
defining scientific object, a falsifiable estimand, a choice-modelling use
case, and a validation experiment that can fail.

The venue lock is the *Journal of Choice Modelling*: the idea must improve a
choice-modelling decision about utility, heterogeneity, identification, or
policy interpretation. The implementation lock is reproducibility with the
repository's grouped validation and simulation framework. The writing lock is
direct exposition with one argumentative duty per paragraph.

## Idea-DNA representation

Each candidate is recorded by seven loci:

| locus | question |
|---|---|
| object | What new object is estimated or certified? |
| intervention | What change in alternatives, tasks, or policy is evaluated? |
| estimand | What number, set, or decision is reported? |
| identification | What makes the object distinguishable from a fit score? |
| validation | Which controlled failure should be visible? |
| policy output | Which WTP, share, elasticity, or regret quantity changes? |
| stop rule | When must the method decline to make a claim? |

The search was run with two source domains: utility specification in discrete
choice and selective decision-making under model uncertainty. The final
candidate below uses one mutation of the research object. It does not combine
an RF, a process model, an attention measure, and an abstention rule into a
single algorithm.

## Candidate audit

Scores use a five-point scale. Novelty means distance from the closest choice
modelling papers; usefulness means the decision risk addressed; feasibility
means that the current repository can test the idea without inventing an
unverifiable estimator; fit means alignment with JOCM readers. A high overlap
score is a reason to stop.

| candidate | defining object | novelty | usefulness | feasibility | fit | overlap risk | decision |
|---|---|---:|---:|---:|---:|---:|---|
| Policy-equivalence set | the set of models that are observationally adequate within a declared predictive tolerance, paired with the width of their policy outputs | 3 | 5 | 4 | 5 | 3 | retain as backup |
| Intervention-stable contrast audit | a utility basis is accepted only when policy-relevant contrasts remain stable under predeclared choice-set and attribute interventions | 3 | 4 | 3 | 5 | 3 | retain as backup |
| Library-insufficiency certificate | the minimum slack or adversarial perturbation required for the candidate library to rationalize observed choices | 4 | 5 | 2 | 4 | 3 | retain as theory extension |
| Model-equivalent choice-pair test | a designed pair of choice tasks has the same candidate-model sufficient statistics but a different attribute decomposition; systematic choice differences reject the candidate basis | 4 | 5 | 3 | 5 | 3 | selected high-risk pivot |
| Adaptive discriminating task design | choose the next task to separate competing utility mechanisms | 2 | 4 | 2 | 4 | 5 | stop |
| Permutation-invariant learner | enforce alternative-exchangeability in the diagnostic learner | 2 | 4 | 3 | 4 | 5 | stop as primary idea |
| Reliability-based mechanism separation | use repeated choices or test-retest stability to distinguish taste from processing | 2 | 4 | 3 | 4 | 4 | stop as primary idea |
| Process-aware abstention | route high score dispersion to an unresolved branch when sequence evidence is absent | 3 | 5 | 4 | 5 | 4 | keep only as prior baseline |

The adaptive-design candidate is stopped because optimal model
discrimination designs already form a choice-experiment literature. The
permutation candidate is stopped because exchangeability and permutation
invariance are now explicit objects in choice-model theory and open software.
Reliability and attribute non-attendance are also established measurement and
process-model topics. The process-aware abstention prototype is useful for a
failure-boundary experiment, but it does not become the new contribution.

The model-equivalent choice-pair test is selected as a high-risk pivot. Its
object is a design-based specification test,
not a policy-uncertainty interval. Under a candidate utility basis, two tasks
with the same vector of alternative utility differences must have the same
choice probabilities. A randomized pair that preserves those differences while
changing the attribute decomposition creates a direct test of the basis. The
test can reject an apparently well-fitting model. Its pair labels can show
which predeclared transformations produce violations, but they do not identify
a unique omitted term.
It requires purpose-built paired tasks or a strong matched-task structure in
the empirical data, which the current LPMC and Swissmetro files have not yet
been shown to provide. It cannot enter the main paper without that data
condition.

A direct audit of the public Biogeme Swissmetro file found 10,728 rows for
1,192 IDs and only eight exact repeated full alternative profiles within an
ID, concentrated in two IDs. This is too small and too irregular for a clean
paired-task validation. The pivot therefore needs a paired-task supplement or
a different dataset; the existing Swissmetro application can validate the
baseline workflow only.

An observational fallback can match within-respondent tasks on two frozen
candidate utility-difference coordinates. The public file yields 26, 85, 535,
and 1,924 pairs at coordinate tolerances 0.01, 0.02, 0.05, and 0.10. This
creates a power-versus-approximation trade-off and requires a separate
clustered bootstrap. The reproduced counts are in
`results/swissmetro_pair_audit.csv`. It is a possible extension of the pivot,
not evidence that the existing panel is a randomized paired-task experiment.

## Selected pivot: model-equivalent choice-pair test

### Scientific question

Can a utility basis be rejected by a choice experiment that holds its predicted
utility differences fixed while changing the underlying attribute composition?

### New object

Let \(b(x)\) be a candidate alternative-specific basis and let
\(\widehat\beta\) be estimated on development data. Construct two choice tasks
\(A_r\) and \(B_r\) so that

\[
\{b(x_{Aj})'\widehat\beta-b(x_{Ak})'\widehat\beta\}_{j,k}
=
\{b(x_{Bj})'\widehat\beta-b(x_{Bk})'\widehat\beta\}_{j,k},
\]

while the raw attribute decompositions differ. Under the candidate model, the
two tasks have the same choice-probability vector. Define the equivalence
violation for alternative \(j\) as

\[
\Delta_j=\Pr(Y=j\mid A_r)-\Pr(Y=j\mid B_r),
\]

and estimate it with a grouped difference in choice shares. A bootstrap under
the equality restriction supplies the critical value. The test rejects the
candidate basis when the observed violation exceeds that value. Pairs that
change one omitted locus at a time can localize the missing nonlinear,
threshold, or interaction term.

This is one design-based specification test. The existing Random Forest is a
comparison method and the process gate is a separate failure-boundary audit;
neither is part of the definition of \(\Delta_j\).

### Falsifiable predictions

1. Under a correctly specified additive basis, the equivalence violation is
   zero in expectation and the bootstrap test has the declared size.
2. An omitted nonlinear term produces different choice shares for pairs with
   the same candidate utility differences.
3. An omitted interaction can produce a violation for pairs that alter the
   interacting attributes, while the probability-scale test can also transmit
   nonlinear main effects through the choice-probability map. The design must
   report this boundary instead of claiming unique interaction localization.
4. A model with good aggregate log loss can still fail the paired-task test;
   this is the key distinction from an ordinary holdout comparison.
5. The violation must disappear after the missing basis term is added. If it
   does not, the test is detecting process or heterogeneity rather than a
   functional-form omission and the paper must report that boundary.

### Validation design

The simulation creates paired tasks with equal candidate utility differences
and varies one omitted mechanism at a time. The development split estimates
the candidate basis and freezes the pair construction; the test split contains
fresh respondents who see one member of each pair. Size is calibrated by a
parametric bootstrap under the candidate model. Power is measured against
nonlinear, threshold, interaction, and latent-heterogeneity alternatives.

The empirical test is feasible only if the Swissmetro instrument contains
matched tasks or if a small paired-task supplement can be collected. LPMC is
used as a transfer benchmark for ordinary specification recovery, not as proof
of the paired-task invariance when its task design lacks the required pairs.

### Why the pivot is distinct, and where it overlaps

Fok and Paap (JOCM 2025) already use pairs of alternatives to build composite-
likelihood and GMM overidentification tests for MNL misspecification and IIA.
That is a direct adjacent precedent. Utility-neutral experimental designs and
invariance tests also show that equivalent utility representations are an
established design and theory concern. The present object is narrower: it
creates pairs of *tasks* with different attribute decompositions and equal
candidate utility-difference vectors, then tests whether the candidate utility
basis is sufficient across those tasks. The null is cross-task basis
sufficiency, not the within-task binary-pair implication of IIA.

This distinction is a hypothesis for a literature audit, not a priority claim.
The method must be described as a new task-level construction only after a
systematic search confirms that no existing paper has already proposed the
same restriction. It does not rank learners, average models, or add a process
mechanism. It can be rejected if the test has poor size, fails to localize the
omitted term, or cannot be implemented with a valid paired-task instrument.

## Policy-equivalence set as the backup route

The policy-equivalence set remains a feasible fallback if the paired-task data
condition cannot be met. Its initial prototype exposed a weakness: pointwise
model intervals had poor finite-sample coverage in several conditions. A valid
version would need parameter-bootstrap or confidence-set construction before it
could support a submission claim. Counterfactual sensitivity and partial
identification literatures make this route less distinct than the paired-task
pivot.

## Decision gate before manuscript adoption

The pivot enters the main paper only after a simulation gives all of the
following: null size near the declared level; useful power against at least
two omitted mechanisms; localization when only one attribute locus changes;
loss of rejection after the correct term is added; and a credible paired-task
implementation for an empirical study. The observational fallback currently
has a near-5% null size but only 0.225 and 0.175 rejection rates for the
engineered nonlinear and interaction conditions in a 40-replication boundary
run. The public Swissmetro file gives 126, 236, 1,371, and 3,430 approximate
pairs at tolerances 0.01, 0.02, 0.05, and 0.10, with descriptive p-values
between 0.404 and 0.468. These results keep the fallback out of the main
claim. If an exact supplement cannot be fielded, the pivot is reported as a
rejected innovation and the paper keeps the narrower recoverability benchmark.

The exact randomized benchmark now passes the null, power, and repair checks
in a 50-replication run: rejection is 0.06 under the additive null, 0.84 for
the nonlinear condition, and 1.00 for the threshold and interaction
conditions. After adding the corresponding term, rejection falls to 0.04,
0.02, and 0.02 for the nonlinear, threshold, and interaction conditions.
The remaining gap is localization: the joint shift changes multiple raw
loci, so the interaction signal is not uniquely separated. A four-cell
probability-scale factorial contrast was audited as a repair. It reduced
rejection after an interaction term was supplied, but rejection rose to 0.86
when an interaction coexisted with a nonlinear main effect. The audit is in
`manuscript/factorial_contrast_note.md`; it excludes term-level localization
from the claim. The paired-task pivot remains high-risk until a purpose-built
supplement validates the narrower invariance-violation test.

## Literature boundary used for the audit

- JOCM scope: https://shop.elsevier.com/journals/journal-of-choice-modelling/1755-5345
- Model choice and framing effects: https://doi.org/10.1016/j.jocm.2024.100524
- External validity review in stated choice: https://doi.org/10.1016/j.jocm.2021.100322
- Partial identification and counterfactual bounds: https://doi.org/10.1016/j.jeconom.2024.105854
- Latent utility and permutation invariance: https://doi.org/10.1016/j.jeconom.2024.105844
- Optimal experimental designs for model discrimination: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4095610
- Choice-set confounding: https://www.cs.cornell.edu/~arb/papers/choice-set-confounding-KDD-2021.pdf
- Fok and Paap, new MNL misspecification tests: https://doi.org/10.1016/j.jocm.2024.100531
- Utility-neutral design and underidentification boundary: https://doi.org/10.1111/bmsp.12256
