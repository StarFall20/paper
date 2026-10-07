# Fibre quotient completeness for candidate-basis audits

## Research problem

Choice-model diagnostics usually begin after data collection. A richer utility
specification, a residual learner, or a likelihood-ratio test can detect a
departure only when the observed tasks contain variation in that direction. A
candidate basis can therefore pass a conventional diagnostic because the task
space is blind to the relevant departure. The design question is prior to
estimation: which departures remain observable after the candidate basis is
held fixed?

## Core construction

Let (b(x)) be the candidate sufficient-statistic vector for a raw alternative
profile (x). Construct reflected profiles so that

\[
b(A_+)=b(A_-),\qquad b(B_+)=b(B_-).
\]

Candidate utility differences are then identical under the two reflections for
every coefficient vector. This removes dependence on a single pilot estimate.
For a local departure (h), define the odd contrast

\[
D_t(h)=\frac12\{[h(A_+)-h(B_+)]-[h(A_-)-h(B_-)]\}.
\]

For a declared departure space (H=\operatorname{span}
\{h_1,\ldots,h_r\}), stack these contrasts into (D_S). The task block is
**fibre-complete** for (H) when its rank reaches the rank available in the
full structural pool. The null space of (D_S) lists the departures that the
selected block cannot distinguish. This is the paper's main theoretical
object; maximin selection is used later to improve conditioning among complete
blocks.

## Smart-device implementation

The candidate basis is (b=(m+s,i+e,d,q)), a deliberately coarsened
representation of care intensity, clinical quality, data management, and
price. The enumerated pool contains 1,024 reflections with equal raw
comparison signatures. The declared local departure space contains a total
quadratic term, a midpoint mode-support interaction, and an
intelligence-cloud interaction. The full pool has rank three. A three-task
subset reaches rank three; the selected five-task block adds coverage of five
probability-gap bins and has information eigenvalues 1.648, 6.063, and 7.352.

The endpoint mode-support interaction used in the earlier simulation is
constant on the (m+s) fibre. The rank audit excludes it before inference. This
is a structural blind-direction result, not a failure to obtain a significant
coefficient.

In 200-replication simulations with 600 respondents and 499 respondent-level
sign flips, the maximin block's minimum feature power is .67 at zero candidate
coefficient perturbation and .65 when perturbation standard deviation is .05.
A random structural block reaches .40 and .45. Null rejection for the
maximin block is .045, .035, .045, and .020 as the perturbation standard
deviation increases from 0 to .10. These simulations establish design
coverage; they do not substitute for a human DCE.

## Positioning

The paper does not claim the first maximin robust DCE design. Errore,
Nachtsheim, and Li's 2013 conference abstract already develops maximin
model-robust designs for main effects and interactions. The contribution here
is the candidate-preserving quotient and its algebraic completeness and
blind-direction certificate. Ordinary efficient-design, MNL misspecification,
invariance, and comparison-complexity papers remain important comparators, but
they answer different questions.

## Scope

The guarantee is conditional on the candidate basis, departure library,
attribute support, nuisance screen, and assignment protocol. A full raw basis
can have singleton fibres. A departure that is a function of the candidate
basis is invisible on that fibre. A rejection identifies failure of the
candidate relation on the tested support and does not identify a unique
behavioural mechanism. The submission version must add a preregistered paired
DCE or an external dataset with independent calibration and direct comparisons
against an efficient block and ordinary added-term/residual diagnostics.
