# Fibre violation profile

## Purpose

The global odd contrast answers whether the candidate relation fails somewhere
in the prespecified task block. A specification audit also needs to show where
the failure occurs. The same nuisance-balanced fibre design provides that
information without introducing a second model.

For (K) retained task pairs, let (delta_k) be the oriented odd response
for pair (k). The profile is the vector

\[
\boldsymbol{\delta}=(\delta_1,\ldots,\delta_K).
\]

Each component uses the same respondent-level assignment, candidate freeze,
and nuisance-signature screen as the global test. The confirmatory profile
statistic is

\[
M=\max_{k\le K}\frac{|\widehat\delta_k|}{\widehat{\operatorname{se}}(\widehat\delta_k)}.
\]

Its reference distribution uses respondent-cluster sign flips and keeps the
maximum over tasks in every draw. The global average remains the primary
omnibus test; the profile is a multiplicity-controlled secondary diagnostic.

## Interpretation

The profile identifies the candidate-menu region and reflected direction where
the relation is most strongly violated. It does not identify a unique omitted
term. A task-specific peak can reflect nonlinear utility, heterogeneity,
presentation, or another process effect. The semantic and odd-process checks
remain necessary before attaching a utility interpretation.

This output differs from a residual plot built on the original observational
support. Residual diagnostics evaluate where observed data are misfit. The
fibre profile evaluates predesigned directions that the candidate itself treats
as equivalent. Existing nonparametric specification tests and surrogate
residuals provide useful comparators, but they do not impose this
candidate-preserving intervention.

## Planning evidence

The balanced smart-device block contains five tasks. A binary moderate-utility
simulation uses 600 respondents, 200 replications, and 999 respondent-cluster
sign flips.

| Condition | Global rejection | Profile rejection | Correct peak when localized |
|---|---:|---:|---:|
| Null | 0.035 | 0.050 | — |
| Omission localized to task 3 | 0.330 | 0.885 | 0.990 |
| Omission present in every task | 1.000 | 1.000 | — |

The corrected benchmark draws one orientation coin per respondent block and
shares it across the five tasks. A random-taste boundary control gives
rejection .085, .095, and .085 at coefficient standard deviations 0, .1, and
.2. Those values define a candidate-relation boundary; they do not identify a
unique omitted utility term.

The profile is most useful when the omitted direction is local to a subset of
candidate gaps or attribute decompositions. When the signal is diffuse, the
global and profile tests converge. The task-specific result is planning
evidence; the method still requires a human or external-data validation.

## Relation to existing diagnostics

Fosgerau's JOCM nonparametric specification test and later nonparametric
likelihood-ratio tests are designed to detect broad misspecification in
observed choice data. Surrogate residuals diagnose mean, interaction, and
individual-coefficient misspecification after a model is fitted. The fibre
profile adds a different experimental lever: it creates candidate-equivalent
tasks before outcomes are observed and reports the direction-specific response
with familywise control.

The paper should present the profile as a diagnostic layer of the
nuisance-balanced fibre audit. It should not be marketed as a new general
nonparametric test.
