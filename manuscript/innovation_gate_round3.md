# Round-3 independent-innovation gate

## Decision

The strongest remaining candidate is the **Utility-Fibre Invariance Test
(UFIT)**, implemented by the candidate-preserving paired-task randomization
design. Round 4 records the stricter boundary: UFIT is a method-and-design
extension whose submission-level originality still depends on a formal
candidate-basis proposition and a valid paired-task instrument. The
Mechanism-Fingerprint Choice Audit (MFCA) remains a useful extension, but it
should not carry the whole novelty claim.

UFIT starts with a fitted candidate basis (B), then constructs a pair of
held-out tasks whose candidate utility differences are equal for every
coefficient vector while their raw attribute decompositions differ. For a
candidate (V_j(x)=B_j(x)\beta), the transformation (T) belongs to the
candidate's utility fibre when

\[
 B_j(Tx)\beta-B_k(Tx)\beta
 =B_j(x)\beta-B_k(x)\beta
\quad\text{for every }j,k\text{ and every }\beta.
\]

The candidate then predicts exchangeability of the two observed choices. A
frozen candidate fitted on development respondents, a respondent-cluster
sign-flip reference, and a predeclared task transformation turn that
prediction into a finite-sample specification diagnostic. The estimand is the
violation of candidate-implied equality across a utility fibre. It is not a
general misspecification test and it does not identify a unique omitted term.

The transformation is defined by the candidate's utility representation, the
outcome comparison is across tasks rather than alternative pairs within one
task, and inference is design-based after the candidate is frozen. Existing
invariance theory and pairwise MNL tests are close foundations, so the paper
must claim this exact construction and its operating boundary, not priority
over all invariance or misspecification work. The full round-4 decision is in
`innovation_gate_round4.md`.

## What is already known

Breitmoser (2021) gives an axiomatic foundation in which translation,
presentation, context, and IIA invariances characterize conditional logit.
Fok and Paap (2025) construct pair-based MNL misspecification tests using
composite likelihood and overidentifying moments. Day et al. (2009) formulate
signature patterns for position- and precedent-dependent ordering effects.
Boxebeld (2024) reviews ordering effects in DCEs. These studies rule out three
overclaims:

1. UFIT is not the first test based on an invariance assumption.
2. UFIT is not the first paired-alternative or overidentification test.
3. MFCA is not the first signature-based audit of presentation or sequence
   effects.

The defensible delta is the candidate-conditioned **utility-fibre** relation,
with cross-task equality, a frozen development candidate, cluster
randomization, and an explicit unresolved outcome when several axes reject.
That combination is a method extension requiring an empirical paired-task
instrument. It is not a stitched estimator that merges nonlinear utility,
order effects, and inertia into one model.

## MFCA boundary

MFCA should be described as a localization extension to UFIT. Its three axes
are a utility-fibre shift, alternative relabelling, and a repeated task after a
neutral filler. The shared maxT reference controls the familywise test. The
simulation gate shows the intended operating pattern: nonlinear utility
activates the fibre axis, position bias activates the presentation axis, and
inertia activates the sequence axis. Combined mechanisms are mixed or
unresolved. This is useful because the action rule refuses unsupported causal
labels.

The order-effect literature means MFCA must not be presented as a wholly new
theory of process effects. The contribution is narrower: use the same frozen
candidate and candidate-implied relations to report an invariance signature,
with a prespecified ambiguity action. If the empirical instrument cannot
support all three relations, MFCA becomes an appendix simulation and UFIT
remains the main contribution.

## Gate evidence

The corrected 50-replication MFCA benchmark uses 300 respondents, eight
nuisance tasks, a `(35,45)` fibre shift, a counterbalanced identity swap, and a
repeated focal task separated by a neutral filler. A shared respondent sign is
used for all three axes. The null rejection rates are .02/.00/.00 for the
additive condition and .02/.02/.02 for linear random price sensitivity. The
single-mechanism rates are .82 fibre for nonlinear utility, 1.00 presentation
for position bias, and 1.00 sequence for inertia. These values are
development evidence, not empirical validation.

## Submission-level requirement

The paper may make a strong innovation claim only after it adds a forced-choice
paired-task supplement, or an equivalent dataset, that supports the fibre
transformation and respondent-level randomization. The supplement must be
preregistered, include a negative-control transformation, and compare fixed,
constrained-maximin, and repeated-shift designs. The main analysis must report
null size, power against declared out-of-library departures, sensitivity to
linear random taste heterogeneity, and the unresolved action for mixed
violations. Without this empirical gate, the contribution is a well-supported
method proposal with simulation evidence, not a completed JOCM submission.

## References used for the boundary

- Breitmoser (2021), *An axiomatic foundation of conditional logit*,
  https://doi.org/10.1007/s00199-020-01281-1
- Fok and Paap (2025), *New misspecification tests for multinomial logit
  models*, https://doi.org/10.1016/j.jocm.2024.100531
- Day et al. (2009), *Task independence in stated preference studies: A test
  of order effect explanations*, https://www.econstor.eu/obitstream/10419/48813/1/615014658.pdf
- Boxebeld (2024), *Ordering effects in discrete choice experiments: A
  systematic review*, https://doi.org/10.1016/j.jocm.2024.100489
