# Paired-task supplement design

## Research question

The supplement asks whether a fitted utility specification makes the choice
probabilities invariant to a transformation that should preserve every
candidate utility difference. The question is about specification adequacy,
not predictive accuracy alone.

## Task construction

Let a choice task contain alternatives (j=1,ldots,J), profiles (x_j), and
candidate utility

\[
V_{nj}=\alpha_j+\beta^{\top}x_{nj}.
\]

For a paired task, choose a fixed shift (Delta) and present

\[
x'_{nj}=x_{nj}+\Delta
\]

for every alternative in the choice set. The design must preserve the same
alternative labels, availability, and task framing. Under the additive
candidate,

\[
V'_{nj}-V'_{nk}=V_{nj}-V_{nk}
\]

for every respondent and every coefficient vector (beta). This remains true
with random coefficients on the shifted attributes. A nonlinear term, a
threshold, or an interaction generally changes the difference after the shift.

Use at least two shifts that target different attribute directions and one
joint shift. Randomize which member of each pair appears first and counter-
balance the assignment across respondents. Keep the pair members separated by
nuisance tasks when needed to reduce memory effects, while retaining a pair
identifier for analysis.

## Opt-out handling

The original instrument assigns no product attributes to the opt-out. A
product-only shift therefore changes product-versus-opt-out differences and is
not a valid null transformation. The supplement has two defensible options:

1. Use a forced-choice block with the same product alternatives and state that
   the diagnostic targets the in-product utility specification.
2. Give the opt-out a fully defined profile and apply the same shift to it as
   to every product. The profile must be visible in the task design and
   included in the preregistered utility equation.

The first option is simpler and preserves the interpretation of the existing
opt-out model. The second option permits a joint test that includes the
opt-out-specific constant, but it requires new instrument development.

## Estimation and randomization reference

Split respondents before estimation. Fit the candidate on one respondent
fold, freeze its parameters, and evaluate the paired contrasts on the other
fold. For pair (r), define

\[
C_r=(Y_{r0}-Y_{r1})-(\hat p_{r0}-\hat p_{r1}),
\]

where (Y) is the one-hot choice vector. Aggregate a squared norm of the mean
contrast across pairs. The reference distribution flips the sign of all pair
contrasts contributed by a respondent. This retains arbitrary dependence
within a respondent and uses only pair exchangeability under the candidate.

The preregistration must state the transformation, pair order rule, candidate
terms, fold split, statistic, number of randomization draws, and rejection
threshold before outcomes are inspected. Report the all-pair test and the
pre-specified shift-specific contrasts with multiplicity control or a single
global decision rule.

## Falsification and failure checks

The supplement must include an additive null, a nonlinear alternative, an
interaction or threshold alternative, and a random-coefficient condition. Add
an order-effect placebo by reversing presentation order in a separate block.
If the placebo rejects under the additive null, the task protocol violates the
exchangeability condition and the diagnostic result is uninterpretable.

The test identifies a violation of candidate-preserving task invariance. It
does not identify a unique omitted term, and it does not establish that the
candidate is correct when the test does not reject. Existing data without
randomized paired tasks cannot be relabelled as this experiment.

