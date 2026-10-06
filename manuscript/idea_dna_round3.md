# Idea-DNA round three: mechanism-fingerprint choice audit

## Search question

The candidate-preserving paired-task test currently rejects a utility basis but
does not tell the analyst whether the failure comes from functional form,
presentation, or task sequence. The next idea asks whether a small, balanced
set of model-implied transformations can localize the *mechanism family* that
is inconsistent with the candidate while preserving an explicit unresolved
branch when signatures overlap.

## Proposed object

The **Mechanism-Fingerprint Choice Audit (MFCA)** is a design-based diagnostic
with three predeclared relations:

1. **Utility-fibre relation.** Add a common shift to every alternative,
   including a fully defined opt-out profile. A candidate utility basis keeps
   every alternative utility difference unchanged. A violation loads on the
   functional-form axis.
2. **Presentation-relabel relation.** Randomly permute the displayed order of
   alternatives and map the observed choice back to the identity of the
   alternative. A static choice model predicts the same identity probabilities.
   A violation loads on the presentation or position axis.
3. **Sequence-repeat relation.** Repeat a calibrated task at different points
   in the sequence, with counterbalanced order and an independent filler task.
   A static candidate predicts the same probabilities conditional on the
   displayed attributes. A violation loads on the sequence or state axis.

The held-out respondent-level contrast vector is

\[
 C_n=(C_{n,\mathrm{fibre}},C_{n,\mathrm{presentation}},
 C_{n,\mathrm{sequence}}).
\]

Each component subtracts the candidate-implied probability relation. Cluster
randomization gives a reference for each component. A shared-sign maxT
reference controls the familywise rejection rate across the three axes. The
output is a **mechanism signature**, not an identified omitted term. For
example, a large fibre component with small presentation and sequence
components supports a functional-form diagnosis; a large sequence component
with the other two near zero supports a process warning. A combined signature
is sent to the unresolved branch.

## Why this is a bounded extension

Order effects and process heterogeneity are established topics. Day et al.
(2009) already formulate signature patterns for position- and precedent-
dependent ordering effects, and Boxebeld's 2024 JOCM review covers the wider
ordering literature. Kazagli and de Lapparent (2023) and Pilli et al. (2022)
model non-trading, inertia, and the cost of assigning process heterogeneity to
taste heterogeneity. Fok and Paap (2025) provide pair-based MNL
misspecification tests. These precedents mean MFCA is not a wholly new
process-effect theory.

Its bounded delta is a reproducible design extension: one frozen candidate is
tested through a utility-fibre relation plus presentation and sequence
placebos, and the result is reported as a calibrated signature with an
explicit ambiguity outcome. The independent contribution remains the
candidate-preserving utility-fibre test; MFCA is retained only if the
single-mechanism gate and empirical instrument support it.

## Falsification and overlap controls

- The additive condition must maintain joint size near the declared level for
  all three axes.
- A quadratic utility departure should activate the fibre axis only.
- A position-bias departure should activate the presentation axis only.
- An inertia departure should activate the sequence axis only.
- Combined mechanisms must produce an unresolved or mixed signature rather
  than a forced label.
- Random taste heterogeneity with a common shift must remain compatible with
  the fibre relation; otherwise the test is detecting a preserved linear
  heterogeneity, not functional form.
- A placebo permutation and a no-repeat sequence block must not activate the
  corresponding axis.

If the single-mechanism signatures do not separate at the planned sample size,
MFCA is a failed extension and the paper keeps the narrower candidate-
preserving test. The benchmark must report the confusion matrix of signatures,
not only average power.

The first 50-replication gate passes this screen. After maxT adjustment, the
additive null rejects on the fibre, presentation, and sequence axes at rates
0.02, 0.00, and 0.00. Nonlinear utility gives 0.82 fibre power with 0.00
presentation power; position bias gives 1.00 presentation power with 0.02
fibre power; inertia gives 1.00 sequence power with 0.02 fibre power. The
dominant single-axis signatures occur in 39/50 nonlinear, 49/50 position-bias,
and 49/50 inertia replications. Combined conditions yield mixed signatures and
are not forced into a single label. These are development results; the method
still requires the empirical supplement and an external transfer check.

## Required empirical design

The supplement needs a forced-choice block or an explicit opt-out profile whose
attributes participate in the fibre transformation. Alternative order is
counterbalanced within respondent. Sequence repeats are separated by a filler
task, and the first and later positions are balanced across respondents. The
candidate is estimated on development respondents and frozen before all three
held-out contrasts are formed.

## Literature boundary

- Boxebeld (2024), ordering effects review:
  https://doi.org/10.1016/j.jocm.2024.100489
- Kazagli and de Lapparent (2023), heterogeneous decision rules:
  https://doi.org/10.1016/j.jocm.2023.100413
- Pilli, Swait, and Mazzon (2022), process heterogeneity and profitability:
  https://doi.org/10.1016/j.jocm.2022.100359
- Fok and Paap (2025), MNL misspecification tests:
  https://doi.org/10.1016/j.jocm.2024.100531
- Day et al. (2009), task independence and order-effect signatures:
  https://www.econstor.eu/obitstream/10419/48813/1/615014658.pdf
- Segura et al. (2016), metamorphic testing:
  https://doi.org/10.1109/TSE.2016.2532875
