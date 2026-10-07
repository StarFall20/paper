# Fibre quotient completeness for candidate-basis audits in discrete choice models

## Abstract

Choice-model diagnostics are limited by the variation present in the observed
choice tasks. A candidate utility basis may fit those tasks while remaining
silent about departures that the design never exposes. This paper introduces
fibre quotient completeness, a design-based audit for a coarsened candidate
basis. Raw profiles are grouped by their candidate sufficient statistic. A
reflected task preserves that statistic alternative by alternative, so the
candidate utility difference is unchanged for every candidate coefficient
vector. The odd contrast between the two reflections becomes an observable
operator on a declared local departure space. Its rank gives the dimension of
departures that the task block can test; its null space identifies blind
directions before outcomes are collected. A structural-fibre search for a
smart-device testbed yields 1,024 admissible reflections. The full pool has
rank three. A five-task block covers five probability-gap bins and has
information eigenvalues 1.648, 6.063, and 7.352. In matched planning
comparisons, a reflected fibre design detects a weak decomposition departure
that ordinary observational LR and out-of-fold residual diagnostics rarely
recover on narrow support. The result is a conditional specification audit,
not a universal misspecification test. The method requires a nontrivial fibre,
a declared departure space, and an independent paired-task implementation.

## 1. Motivation and contribution

Choice-modelling studies commonly select a utility specification by likelihood,
information criteria, holdout prediction, or comparison with a richer model.
Those diagnostics operate on the support already present in the data. A good
fit can coexist with a candidate basis that treats two raw attribute
configurations as equivalent even when respondents respond differently to
those configurations.

The paper asks a design question: which departures from a candidate basis are
observable after candidate utility is held fixed? The answer has one central
object. A candidate sufficient-statistic map partitions raw profiles into
fibres. Reflected tasks remain inside those fibres. The odd response across a
reflection is a quotient-space observable because the candidate contribution
cancels. The rank of the resulting operator measures local testability; its
null space provides a structural blind-direction certificate.

The contribution has three parts. First, the paper formalizes fibre quotient
completeness and gives a first-order rank characterization. Second, it gives a
reproducible task construction that preserves candidate utility for every
coefficient vector and balances predeclared nuisance signatures. Third, it
shows how a complete block differs from an efficient observational design and
from ordinary LR or residual checks. Maximin selection is used after the rank
condition to improve conditioning among complete blocks. It is not presented
as a new generic maximin design criterion.

## 2. Candidate fibres and the odd operator

Let (x) denote a raw alternative profile and let (b(x)) be the candidate
sufficient-statistic vector. The candidate utility is

\[
V_0(x;\beta)=b(x)'\beta .
\]

For a reflected binary task (t=(A_+,A_-,B_+,B_-)), impose

\[
b(A_+)=b(A_-),\qquad b(B_+)=b(B_-).
\]

The candidate A--B utility difference is the same in the plus and minus
versions for every (eta). A respondent-level orientation coin assigns one
version of each pair, and the reference distribution flips the respondent
cluster signs.

For a local departure (h), define

\[
D_t(h)=\frac12\{[h(A_+)-h(B_+)]-[h(A_-)-h(B_-)]\}.
\]

For (H=\operatorname{span}\{h_1,\ldots,h_r\}), a task block (S) induces
(D_S=[D_t(h_ell)]). Define

\[
\operatorname{FQC}(S;H)=\operatorname{rank}(D_S),
\qquad
r_{\max}=\operatorname{rank}(D_{\mathcal X}),
\]

where (mathcal X) is the complete admissible structural pool. The block is
complete for (H) when (operatorname{FQC}(S;H)=r_{max}). Directions in
(ker(D_S)) are locally indistinguishable on the selected block. Directions
in (ker(D_{mathcal X})) are structurally invisible on the available
attribute support.

### Proposition: fibre quotient completeness

Suppose the candidate model is linear in (b(x)), reflected pairs preserve
(b(x)) alternative by alternative, and the random reflection sign is assigned
at the respondent level. Under a common random-utility error law, the
candidate expected odd contrast is zero. For a local perturbation
(V(x)=b(x)'\beta+\eta h(x)), the first-order expected odd contrast is
proportional to (D_S(h)). The dimensions of the locally distinguishable
departure space and its blind space equal the rank and nullity of
(D_S).

Equality of (b(x)) removes the candidate component for every (eta). The
respondent-level sign reverses the odd perturbation while retaining the even
component. A smooth logit probability changes by (p(1-p)) times the reflected
utility perturbation, which gives (D_t(h)) after the plus-minus contrast.
The same argument extends to a smooth symmetric binary choice link with an
even nuisance scale. Presentation, order, attention, and taste heterogeneity
remain possible sources of a relation violation and require separate controls.

## 3. Structural-fibre design

The smart-device testbed uses raw attributes for use-management mode (m),
intelligence (i), support (s), evidence (e), data management (d), and
price (q). The coarsened candidate basis is

\[
b(x)=(m+s,\ i+e,\ d,\ q).
\]

The generator enumerates profiles with identical basis vectors, forms reflected
pairs, and retains only pairs with equal raw comparison signatures. The screen
matches squared distance, Hamming distance, total absolute level change, and
the counts of positive and negative component changes. It also imposes five
predeclared candidate-gap bins.

The pool contains 1,024 reflections. The declared departure space contains a
total quadratic term, a midpoint mode-support interaction, and an
intelligence-cloud interaction. The full pool has rank three. A three-task
subset reaches rank three with eigenvalues approximately 0.483, 2.217, and
5.055. The selected five-task structural maximin block uses pool indices 201,
59, 557, 997, and 764. It covers gaps near 0.22, 0.36, 0.56, 0.66, and 0.82;
its information eigenvalues are 1.648, 6.063, and 7.352.

The endpoint mode-support interaction from the earlier simulation is constant
on the (m+s) fibre. The rank audit excludes it before inference. This is a
support result: increasing the sample size cannot recover a direction that is
constant on every admissible fibre.

## 4. Simulation evidence

The structural calibration benchmark uses 600 respondents, 499 respondent-
level sign flips, and 200 replications. Under the structural null, maximin
null rejection is .045 at zero coefficient perturbation and remains between
.020 and .045 as the perturbation standard deviation increases to .10. Under a
weak departure with strength (eta=.30), the minimum feature power is .67 at
zero coefficient perturbation and .65 at perturbation standard deviation .05.
A random structural block reaches .40 and .45.

The incremental benchmark compares ordinary diagnostics with a reflected
candidate-preserving supplement under the same respondent budget. When
observational tasks are concentrated near the centre of a decomposition
coordinate, an added-term LR test and an out-of-fold residual proxy reject an
omitted decomposition in .06 and .03 of 200 replications. The parity-fibre
block rejects in 1.00. Under an even-scale nuisance, the corresponding rates
are .03, .06, and .05. The reflected design creates support for the declared
odd departure; the scale control prevents the paper from treating every
rejection as utility evidence.

These are planning simulations. They establish the design logic and its
finite-sample boundaries. They do not establish a human preference result.

## 5. Relation to adjacent literature

Model-robust DCE design already includes maximin efficiency criteria across
main-effects and interaction models. The present paper does not claim that
criterion as new. Its increment is the restriction to candidate-preserving
fibres and the resulting rank/null-space testability certificate.

JOCM misspecification tests use outcome-side moments or composite likelihood;
axiomatic work characterizes invariance restrictions; recent choice-design
papers optimize precision for a chosen model. These strands motivate the
problem and define comparison baselines. The fibre quotient asks a different
question: which departures remain testable after the candidate relation is
imposed by construction?

## 6. Submission scope and empirical gate

The method applies when the candidate basis is a coarsening of the raw
attribute space. A saturated raw basis has singleton fibres. A departure that
is a function of the candidate sufficient statistic is invisible on the fibre.
A rejection identifies failure of the candidate relation on the tested support,
with mechanism attribution left open.

The submission version requires a preregistered paired DCE or an external
choice dataset with the same assignment structure. It must freeze the
candidate basis, departure library, nuisance signature, gap bins, orientation
assignment, and action for mixed violations before outcomes are inspected. It
must report usable-pair counts, semantic screening, response-time and
complexity diagnostics, sample-size/power analysis, and direct comparisons
with an efficient block, an added-term LR test, and an out-of-fold residual
learner. Until that gate is met, the paper should be presented as a
simulation-validated design method and describe it as a simulation-validated design method until that gate is met.
