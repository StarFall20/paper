# Round-4 innovation audit: what survives a stricter novelty test

## Decision

The transportable-utility prototype is not ready to carry the paper. Its
penalty weight was chosen after inspecting target-environment performance, so
the reported improvement is exploratory. The raw coefficient-dispersion
penalty also confounds taste instability with the logit scale and depends on
the number of included terms. A source-only tuning rule, scale normalization,
and post-freeze target environments are required before this direction can be
considered again.

The Utility-Fibre Invariance Test (UFIT) remains the strongest independent
candidate, with a narrower claim than the current draft used. The useful
object is a **candidate-basis sufficiency audit**: an analyst creates held-out
choice tasks that lie in the same candidate utility-difference fibre while
changing the raw attribute decomposition, then tests cross-task equality of
choice probabilities with respondent-cluster randomization. The method tests
whether the declared basis is sufficient on the tested intervention domain.

UFIT is not a new invariance axiom, a general MNL misspecification test, or a
causal guarantee. Breitmoser (2021) already establishes observable invariance
foundations for conditional logit. Fok and Paap (2025) already construct
pair-based misspecification tests with composite likelihood and GMM. The
defensible contribution is the finite-sample experimental construction that
turns a fitted candidate basis into a cross-task overidentifying restriction,
with a predeclared design and a clustered randomization reference.

## The mathematical object

For alternative (j), let (b_j(x)) be the candidate basis and let

\[
  d(x)=\{b_j(x)-b_k(x):j<k\}.
\]

A paired-task transformation (T) is candidate-preserving when

\[
  d(Tx)=d(x).
\]

Equivalently, the rows of (b(Tx)) differ from the rows of (b(x)) by one
common vector. The candidate therefore predicts the same probability vector
for the two tasks for every coefficient vector. The test estimand is

\[
  \Delta_j(T)=P(Y=j\mid x)-P(Y=j\mid Tx),
\]

with the null (\Delta(T)=0) for each declared transformation. A respondent
sees one randomly assigned member of each pair; the assignment is independent
of the respondent and choice. The sign-flip statistic uses the respondent as
the randomization cluster.

The key interpretation is basis sufficiency on the tested transformation
domain. If a richer utility (g(x)) changes its alternative differences
under (T), the pair can reject the candidate even when ordinary holdout loss
does not. A rejection does not name a unique omitted term. If a candidate
survives, the result supports the tested fibre and design region; it does not
prove global correctness.

The exact softmax equivalence and the randomization conditions are stated in
ufit_proposition.md; analysis/ufit_algebra_check.py checks the algebraic
construction used by the simulation.

## Method-transfer framing

The independent method transfer is **Behavioral Metamorphic Specification
Testing (BMST)**. Metamorphic testing supplies a test oracle when the correct
output for one input is unavailable: a valid transformation creates a second
input whose output must satisfy a known relation. BMST obtains that relation
from the candidate choice model's maximal utility-difference invariant. UFIT is
the formal utility-fibre relation used by BMST.

This is one coherent transfer from software and simulation validation into
choice-model specification. It does not combine a learner, a process model,
and a policy rule. A targeted search found metamorphic-testing methods for
classifiers and simulation validation, and invariance-based choice theory and
MNL misspecification tests, but no exact candidate-basis metamorphic
specification test for discrete choice. This is a search result, not a priority
claim; the introduction must state the boundary and cite the adjacent
literatures. The practical value is a test oracle for a utility library when
single-task fit cannot reveal whether an omitted attribute relation matters.

## Why this is distinct enough to investigate

Breitmoser’s axiomatic result characterizes conditional-logit probabilities
from broad invariance assumptions on observables. UFIT starts with a fitted
candidate basis and makes its implied equivalence classes operational in a
finite experiment. Fok and Paap’s tests compare binary-pair estimators or use
extra pair moments inside a choice set. UFIT compares two *tasks* with the same
candidate difference vector and a different attribute decomposition. The null
is candidate-basis sufficiency across the designed fibre, not IIA across
alternative subsets.

This delta is a method-and-design extension. Its value rests on a practical
question: can a choice modeller obtain a specification diagnostic that targets
the utility library while keeping the inference valid after the library was
estimated? The answer requires the design and the empirical instrument, not a
new label for translation invariance.

## Required negative controls

The main paper must include the following boundaries.

1. **Correct additive basis.** The null rejection rate must be close to the
   declared level under the candidate model.
2. **Out-of-library utility.** Nonlinear and interaction utilities must change
   the predicted choice shares along a fibre and raise rejection.
3. **Repair.** Adding the relevant basis term must reduce rejection to the
   calibrated level.
4. **Scale-only change.** A respondent-specific or environment-specific error
   scale can change fitted coefficients while leaving WTP ratios intact. UFIT
   must distinguish this nuisance from a utility-basis violation; the
   transportability prototype cannot currently do so.
5. **Process change.** Position or sequence effects should reject the relevant
   task relation, with no claim that the utility basis itself is wrong.
6. **Fibre richness.** A very flexible basis can make the fibre nearly empty.
   The paper must report the intervention support and the number of usable
   pairs instead of treating a non-rejection as confirmation.

The existing 50-replication simulations pass the first three controls for the
declared nonlinear, threshold, and interaction probes. They do not identify a
unique omitted term, and the empirical data audit has not found enough exact
paired tasks. Those two facts remain submission conditions.

A separate 100-replication negative-control benchmark gives rejection rates of
.04 for the additive null, .68 for an omitted nonlinear term, .03 for random
linear taste, .23 for a task-specific error-scale drift, and 1.00 for a
declared order effect. The scale and order results show why a BMST rejection is
a relation violation; it is not an automatic functional-form diagnosis. The
implementation is analysis/bmst_negative_controls.py and the frozen output is
results/bmst_negative_controls.csv.

The recommended one-member-per-pair assignment was then evaluated separately
with 600 respondents and 100 replications. Rejection is .05 for the additive
null, .79 for omitted nonlinearity, .05 for random linear taste, and 1.00 for
an omitted interaction. This design removes carryover from the primary
estimand and is the preferred supplement protocol. The implementation is
analysis/bmst_assignment_benchmark.py and the frozen output is
results/bmst_assignment_benchmark.csv.

## Submission gate

UFIT can become the main contribution after four additions:

- a theorem or proposition stating the candidate-basis sufficiency restriction
  and its tested-domain interpretation;
- a preregistered paired-task supplement, or an existing dataset with the same
  randomizable structure;
- a locked simulation with scale, random taste, process, and combined-mechanism
  controls;
- an empirical illustration that reports the fibre construction, usable-pair
  count, null calibration, and the action for a mixed or unresolved violation.

Without these additions, the work should be presented as a tested method
proposal and recoverability benchmark. It should not claim a completed JOCM
innovation.

## References defining the boundary

- Breitmoser (2021), *An axiomatic foundation of conditional logit*,
  https://doi.org/10.1007/s00199-020-01281-1
- Fok and Paap (2025), *New misspecification tests for multinomial logit
  models*, https://doi.org/10.1016/j.jocm.2024.100531
- Heinze-Deml, Peters, and Meinshausen (2018), *Invariant Causal Prediction
  for Nonlinear Models*, https://arxiv.org/abs/1706.08576
- Arjovsky et al. (2020), *Invariant Risk Minimization*,
  https://arxiv.org/abs/1907.02893
- Swait and Louviere (1993), *The Role of the Scale Parameter in the
  Estimation and Comparison of Multinomial Logit Models*,
  https://doi.org/10.1177/002224379303000303
