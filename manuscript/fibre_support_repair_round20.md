# Fibre support repair as a direct FQC consequence

## Purpose

Fibre quotient completeness (FQC) can return a blind direction because the
declared departure is constant on every fibre supported by the current
attribute domain. That result is actionable: it identifies a support repair
problem before data collection. The repair layer asks for the smallest
admissible support expansion that makes the direction vary on at least one
fibre. It is a design-audit output of FQC, not a second model-selection
method.

## Repair map

Let `h` be a departure in the declared local space and let `X` be the current
raw-profile support. Compute the fibre spread

`spread_g(h) = max_{x in g} h(x) - min_{x in g} h(x)`

for each candidate fibre `g` induced by `b(x)`. If every spread is zero, the
corresponding column of the full structural operator is zero. Search over
predeclared support expansions `X'` and select the smallest expansion for which
`max_g spread_g(h) > 0`; the repaired pool can then be screened for a
nonzero odd contrast and included in the FQC rank calculation.

## Smart-device certificate

The current testbed uses levels 0, 1, and 2 and candidate basis
`b(x)=(m+s, i+e, d, q)`. The endpoint interaction
`h(x)=1{m=2 and s=2}` has zero fibre spread and zero odd contrast across all
1,024 reflections. Increasing the mode/support domain to include level 3 is the
smallest tested repair. The repaired search contains fibres with spread 1; one
example has basis `(4, 3, 0, 0)` and includes profiles with `(m,s)=(1,3)` and
`(2,2)`. The certificate is saved in
`results/fibre_support_repair_certificate.csv`.

The result says that additional respondents cannot recover this interaction on
the original support. A support change is required before a power calculation
for that direction has scientific meaning. The same rule applies to any
departure library and keeps the estimand local to the tested domain.

## Boundary with adaptive model discrimination

Adaptive model-discrimination designs choose stimuli that separate competing
models. The FQC repair map starts after a candidate fibre and a departure
direction have been declared. It diagnoses whether the available raw support
can expose that direction and returns the smallest support change that restores
variation. The object being repaired is the candidate fibre quotient, not a
model list.

## Role in the submission

Keep this layer subordinate to FQC. Report it as a practical audit-and-repair
step after the rank/null-space certificate. Do not claim a separate adaptive
design theory. The submission still requires the preregistered paired DCE or
an external dataset with the same candidate-preserving assignment, plus a
full direct Scopus and Google Scholar search record.
