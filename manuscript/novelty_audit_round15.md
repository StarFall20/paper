# Round fifteen: fibre-violation profile audit

## Why add a profile

A single global rejection only says that a candidate relation fails somewhere
in the task block. The same nuisance-balanced fibre construction yields a
vector of task-specific odd responses and a familywise-controlled max-T
diagnostic. This makes the method useful for model repair: it shows which
candidate gaps or attribute decompositions need a richer utility basis.

The profile is a summary of the existing estimand. It does not add a second
choice model, latent process theory, or machine-learning stage.

## Collision check

Existing nonparametric specification tests compare a fitted parametric model
with a flexible alternative on observed support. Fosgerau's JOCM test and
likelihood-ratio tests such as [Testing for discrete choice
models](https://doi.org/10.1016/j.econlet.2007.04.027) are broad tests of
misspecification. Surrogate residuals diagnose mean, interaction, and
individual-coefficient errors after estimation. Assisted specification methods
search over utility libraries.

The fibre profile differs in the source of variation. It constructs new tasks
that the frozen candidate maps to the same complete menu vector, matches a
declared complexity signature, and randomizes one member per respondent. The
max-T step only reports the location of violations across these designed
directions. It should be presented as a diagnostic layer of the design-based
audit, with nonparametric and residual tests as comparators.

## Planning evidence

The balanced five-task block was simulated with 600 respondents, 200
replications, and 999 respondent-cluster sign flips:

| Condition | Global test | Max-T profile | Peak localization |
|---|---:|---:|---:|
| Null | .035 | .050 | — |
| Omitted direction localized to task 3 | .330 | .885 | .990 |
| Omitted direction present in all tasks | 1.000 | 1.000 | — |

The profile's gain appears when the misspecification is local. It does not imply
universal power dominance. The global statistic remains the primary omnibus
decision, and the max-T profile is secondary unless the final preregistration
specifies otherwise.

The rerun uses one orientation coin per respondent block, matching the
respondent-cluster sign-flip reference. A random-taste boundary benchmark gives
rejection rates .085, .095, and .085 at coefficient standard deviations 0, .1,
and .2. This result limits interpretation: the design audits a calibrated
candidate relation, so individual heterogeneity outside that candidate produces
a relation violation without identifying a unique structural omission.

## Claim boundary

The defensible claim is that a nuisance-balanced candidate-fibre design can
produce a controlled map of directional relation violations. The paper should
not call this a new general nonparametric specification test or claim that the
peak identifies the omitted mechanism.
