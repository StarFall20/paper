# Smart device parity supplement preregistration specification

## Estimand

The primary estimand is the global odd choice response across a prespecified
set of candidate-preserving task pairs. A task pair contains a plus and a minus
profile version. One orientation coin is drawn for each respondent block and
the same orientation is used across that respondent's five tasks. The complete
candidate menu vector, including the opt-out, is held fixed across the two
versions.

The null states that the candidate utility basis and the declared even
processing library imply exchangeability of the oriented response. Rejection
is a relation violation. It does not identify a unique omitted interaction.

## Calibration stage

1. Estimate the candidate additive MNL coefficients in an independent pilot.
2. Freeze the coefficient estimate, its covariance, and the candidate-gap
   tolerance before constructing the paired block.
3. Keep a pair only if both candidate menu vectors match within the frozen
   tolerance, the plus/minus squared geometry matches within its frozen
   tolerance, the value-weighted \(L_1\) comparison distance matches, the
   number of changed attributes and raw level-change load match, and the
   intended odd interaction is antisymmetric within its frozen tolerance.
4. If a task fails any tolerance or semantic comprehension screen, replace it
   from the precomputed candidate pool without inspecting paired outcomes.
5. Do not use calibration respondents in the primary parity test.

## Primary task block

Use five tasks from the nuisance-balanced grid with candidate A--B gaps near
0.32, 0.38, 0.57, 0.80, and 0.98. Randomize one reflected member per respondent and
counterbalance alternative position, attribute order, wording emphasis, and
opt-out placement. Record response time, stated certainty, and an even
complexity index.

## Primary statistic and inference

For respondent (i) and task (t), code the A--B response as (+1), (-1),
or (0) for opt-out. Multiply by the randomized orientation and the
precomputed sign of the odd candidate interaction. Average across the five
tasks within respondent. The global statistic is the absolute mean of this
cluster score. Generate its reference distribution by independently flipping
respondent cluster signs 1999 times. Use a two-sided 0.05 threshold.

The five task-specific contrasts are secondary. Control their family-wise error
with a max-{T} sign-flip reference, or report a single global test and the
unadjusted task contrasts as descriptive diagnostics. The global test is the
only confirmatory decision.

The secondary profile reports the signed odd response, standard error, and
candidate gap for every retained task. The max-{T} reference keeps the largest
studentized contrast in each respondent-cluster sign-flip draw. A profile peak
is a localization diagnostic; it is not treated as identification of a unique
omitted term.

## Secondary diagnostics

1. Report the even component as a complexity or scale diagnostic using the same
   cluster structure.
2. Fit the conventional candidate MNL, an expanded LR specification, and an
   out-of-fold residual learner on the calibration-independent data.
3. Report order, position, response-time, certainty, and attribute-attendance
   checks. An odd process effect triggers the nuisance-orthogonal fallback and
   prevents a utility-specific interpretation.
4. Report null, even-scale, order, and semantic-comprehension negative controls.
5. Include a complexity-only negative control based on the moderate-utility
   form \(F(\Delta V/d_{L1})\). The primary utility interpretation is withheld
   if a pair fails the predeclared nuisance-signature check.
6. Include a random-taste boundary control in which individual coefficients are
   drawn around the calibrated population vector. If this condition rejects,
   report a candidate-relation violation and do not attribute it to a specific
   omitted interaction unless the heterogeneous candidate is modeled directly.

## Design and reporting rules

The task grid, coefficient tolerance, semantic screen, randomization seed,
statistic, sign-flip count, and multiplicity rule are frozen before primary
outcomes are inspected. Non-rejection supports the tested relation within this
design. It does not establish global utility sufficiency. The final paper must
report the number of generated candidates, screened tasks, retained pairs,
respondent exclusions, and every replacement made before outcome analysis.
