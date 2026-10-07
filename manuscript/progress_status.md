# Revision progress status

## Round 25 status: public Swissmetro model run completed

I ran the candidate MNL and the cross-fitted approximate fibre audit on the
official public Swissmetro DCE. The raw file contains 10,728 rows; after
dropping 9 `CHOICE=0` missing records, 10,719 rows from 1,191 respondents
remain. The 12 orientation/tolerance cells produce 126--3,540 approximate
pairs and 60--905 respondents with a pair. At tolerance 0.02, the omnibus
p-values are .310 (`index`), .346 (`time`), and .434 (`cost`); at tolerance
0.05 they are .010, .008, and .070. The orientation sensitivity is a direct
empirical boundary of the observational fallback. The run verifies the
candidate fitting, full-menu matching, cross-fitting, and respondent-cluster
reference implementation on real DCE data. Swissmetro has no documented
one-member-per-pair assignment, so it cannot close the primary randomized
paired-DCE gate. Details are in
`manuscript/swissmetro_external_validation_round25.md`, with provenance in
`data/provenance_swissmetro_2026-10-07.md` and results in
`results/swissmetro_external_validation.csv`; the baseline fit and grouped
two-fold score are in `results/swissmetro_model_summary.csv`.

## Round 24 status: sequential coverage transfer rejected

I tested a method transfer from sequential and adaptive DCE design: use a
first wave to estimate the under-exposed residual direction, choose a
candidate-preserving fibre for a second wave, and conduct inference only on
the independent second-wave assignment. The 500-replication gate selected the
same central fibre in every replication. Null rejection was .040 for the
adaptive rule versus .074 for the static comparator; power was 1.000 versus
1.000 for $h_1$, .048 versus .026 for $h_2$, and 1.000 versus 1.000 for a
mixed direction. The update created no structural-rank or meaningful power
gain, so it is rejected as a primary innovation. The negative result confirms
that an adaptive algorithm cannot create a missing feasible fibre. Details are
in `manuscript/sequential_coverage_gate_round24.md`, with code and output in
`analysis/sequential_fibre_coverage_gate.py` and
`results/sequential_fibre_coverage_gate.csv`.

## Round 23 status: independent innovation search and policy-weighting gate

The new literature pass did not identify a stronger, lower-overlap replacement
for the fibre-conditional sufficiency audit. Method transfer, variable
innovation, and reverse-design candidates were screened against recent JOCM
work on assisted specification, decision-rule heterogeneity, attribute
attendance, choice-set size, model-choice sensitivity, and MNL misspecification.
The primary object remains the candidate-preserving fibre, efficient residual
exposure, and finite rank certificate.

I also tested a policy-weighted exposure criterion. A naive trace score that
weights residual directions by a declared deployment profile selected supports
with structural exposure rank 1, even when the policy-direction matrix had
rank 2. The structural log-determinant reference retained rank 2 and minimum
eigenvalue 0.1805. This failure rules out policy weighting as a replacement
theory; it can remain a constrained sensitivity analysis only after the policy
functional and a full-rank requirement are frozen. The audit is in
`manuscript/independent_innovation_search_round23.md`, with code and results
in `analysis/fibre_policy_weighted_design.py` and
`results/fibre_policy_weighted_design.csv`.

## Round 22 status: conditional-sufficiency reformulation and design gate

The innovation has been narrowed to one testable object: a candidate summary
\(\phi(X)\), a complete menu-preserving fibre, and the null
\(Y\perp X\mid\phi(X)\). The conditional response function remains
unrestricted. A local probability tangent gives the exposure identity
\(\mathcal I(h)=\mathbb E[\kappa_Z\operatorname{Var}(h(X)\mid Z)]\), and the
finite-support rank/eigenvalue certificate separates structural visibility
from statistical power. Directions of the form \(q(\phi)\) stay inside the
null and are excluded from the raw-coordinate violation claim.

The new support optimizer is implemented in
`analysis/fibre_exposure_design_optimizer.py`. On a finite (3\times3\)
support with \(\phi(a,b)=a+b\) and residual dictionary \((a-b,ab)\), the
unconstrained candidate-precision design reaches candidate variance 4.00
while exposing no residual direction (rank 0). A fibre-aware coordinate
exchange design, constrained to put at least 10% mass in each non-singleton
fibre, reaches exposure rank 2 and minimum exposure eigenvalue 0.1805. The
result is a design-level separation from ordinary candidate precision, with a
transparent Pareto table rather than a universal optimality claim. Outputs:
`results/fibre_exposure_design_summary.csv` and
`results/fibre_exposure_design_frontier.csv`.

Current gate status:

```
fibre-sufficiency contribution      [#########.] 92%
finite-support proof/certificate    [#########.] 90%
support optimizer and trade-off     [########..] 85%
multinomial full-menu gate          [########..] 88%
overlap audit                       [#######...] 75%
Scopus/Google Scholar exports       [###.......] 30%
paired human DCE validation         [##........] 20%
full manuscript integration         [######....] 60%
submission package                  [###.......] 35%
overall submission readiness        [######....] 63%
```

The design and mathematical gates have advanced. The submission claim still
requires a paired DCE or a public dataset with documented candidate-preserving
assignment, direct institutional database exports, and final manuscript
integration.

## Current decision

The selected innovation is now a **nuisance-balanced candidate-preserving
utility-fibre audit** for a coarsened utility basis. The candidate is frozen,
the complete menu utility vector is preserved, the weighted L1 comparison
signature is matched, and one member of each candidate-equivalent pair is
assigned at the respondent level. The odd response is confirmatory; the even
response and complexity-only arm are process diagnostics.

The original generic ML-assisted search, process-triage, and broad
localization claims have been demoted to comparisons and failure boundaries.
They no longer carry the paper's innovation claim.

## Completed this round

- Restricted the estimand to nontrivial fibres of a coarsened candidate basis;
  singleton fibres are explicitly excluded.
- Formalized the full multinomial menu vector, zero-gap scale anchor,
  geometry-preserving translation, and fibre-screen proposition.
- Added the anchored benchmark against observational LR and an out-of-fold
  residual proxy.
- Added the sample-size and focal-pair power curve.
- Audited the comparison-complexity objection using Shubatt and Yang's model,
  and added geometry, response-time, confidence, order, and scale controls to
  the instrument specification.
- Completed targeted JOCM, ScienceDirect, Google Scholar-compatible, and
  Scopus-linked searches. Direct Scopus and Google Scholar pages were
  inaccessible or authentication-gated; the search log records this boundary
  and avoids an unsupported “first ever” claim.
- Rewrote the core draft, novelty audit, paired-task supplement design,
  submission-gap audit, and readiness checklist.
- Synchronized code, planning results, manifests, and manuscript documents to
  https://github.com/StarFall20/paper.

## Locked planning evidence

The 100-replication anchored benchmark uses 400 respondents and 199
respondent-cluster sign flips. Under an omitted decomposition, observational
LR and the residual proxy reject at .04 and .04; the zero-gap
geometry-preserving arm rejects at .83; the geometry-changing nonzero-gap arm
rejects at 1.00. Under a complexity-only condition, the zero-gap arms remain
near size (.02 and .07) and the geometry-changing nonzero-gap arm responds at
.41.

The 60-replication sample-size curve gives one-pair nonlinear power of .20,
.42, and .70 at 150, 300, and 600 respondents; three focal pairs give .90,
1.00, and 1.00. Additive rejection stays below .08. These are planning
results, not human-data findings.

## Remaining gates

- Field a forced-choice supplement or find data with documented one-member
  assignment, a nontrivial candidate fibre, and the required full-menu
  preservation.
- Preregister the basis, feasible transformations, zero-gap and geometry
  arms, assignment coin, fold split, statistic, sign-flip reference,
  multiplicity rule, and mixed-signature action.
- Run additive, random-taste, scale, order/carryover, and complexity
  controls. Treat rejection as a relation violation, not unique term
  identification.
- Compare BMST with LR and residual diagnostics on the same respondent budget.
- Rewrite the supplied Word manuscript, supplement, cover letter, and
  declarations so they use the anchored-fibre claim and the same frozen
  repository commit.

## Round-eight innovation audit

The current strongest upgrade is nuisance-orthogonal utility-fibre testing. It
projects the omitted-utility signal away from declared scale, geometry,
framing, and order tangents before selecting the task-pair contrast. An
algebraic audit removed the L1 geometry index because it is constant on this
fixed-sum fibre; the corrected benchmark uses squared Euclidean distance. In a
200-replication planning run with 600 respondents, the orthogonal contrast has
rejection rates .035 under the null, .750 for a moderate omitted decomposition,
.020 under declared scale, .035 under shared framing, .035 under unlisted
nonlinear scale, .980 under an out-of-library geometry-framing interaction,
and .385 under the combined condition. The raw-signal comparator gives .025,
1.000, .025, .020, .040, 1.000, and .850 for the corresponding main rows. The
result is a calibrated operating boundary, not final superiority. The local
proposition, sample-size expression, multinomial condition, and out-of-library
interpretation are recorded in `manuscript/nuisance_orthogonal_proposition.md`.
The empirical paired-task supplement and LR/residual comparison remain open.

## Round-nine parity upgrade

The round-nine search found a sharper primary instrument for the main
comparison-complexity objection. A reflected candidate-preserving trajectory
uses z_A(t)=c+t and z_B(t)=c-t. Squared geometry and symmetric scale are even
in t, while the omitted quadratic decomposition term is odd. In the 200-run
binary planning benchmark, the parity test has .050 null rejection, .050 under
even scale, 1.000 under the omitted direction, .055 under shared framing, and
.325 under an intentionally odd scale nuisance. A three-alternative check keeps
the full menu vector phi=(10,20,10) fixed and gives .040 null rejection, .035
under even scale, and 1.000 under the omitted direction. This is a stronger
primary design when the substantive transformation supports the reflection; the
round-eight nuisance projection remains the fallback for arbitrary fibres.

## Submission decision

The design-level innovation gate is provisionally passed after the nuisance
balance correction. The empirical and full-manuscript gates remain open. Until
those gates close, the defensible status is a method-and-design study with a
predeclared human-data validation plan.

## Round-fifteen fibre-violation profile

The balanced task block now reports a vector of task-specific odd responses in
addition to the global omnibus contrast. A max-T respondent-cluster sign-flip
statistic controls the family-wise error rate and identifies where the
candidate relation fails. In a 200-replication, 600-respondent planning run
with 999 sign flips, a localized omitted interaction gives global power .330,
profile power .885, and correct task localization .990. Null rejection is .035
for the global test and .050 for the profile; a diffuse omitted interaction
gives 1.000 for both. This diagnostic is derived from the same
nuisance-balanced fibre object and is not a separate theory.

## Incremental diagnostic comparison

A four-condition planning benchmark now compares observational likelihood and
out-of-fold residual diagnostics with the reflected parity supplement under the
same 600-respondent budget. Across 100 replications, the omitted-direction
condition gives rejection rates .060 (observational LR), .030 (residual
proxy), and 1.000 (parity). An even scale nuisance gives .030, .060, and .050,
while the one-sided fibre comparator rejects at 1.000. The combined omitted
plus even-scale condition gives .050, .060, and 1.000. These values support the
incremental design claim and remain planning evidence until the locked DCE
generator is rerun with the final replication count.

## Link-robustness extension

The parity proposition has been generalized from multinomial logit scale to the
moderate-utility form (F(\Delta V/D)). A 100-replication check under logistic
and probit links gives omitted-direction rejection of 1.000 for both links and
even-scale rejection of .020 for both; null rejection is .020 and .010. The
combined condition gives .750 and .990. These short runs support link
robustness, with final size and power still requiring the locked replication
plan.

## Application-specific fibre audit

The original smart-device attribute space supports an exact paired instrument.
The selected A+/A- and B+/B- profiles preserve each alternative's additive
candidate utility, the A--B gap (.40), the opt-out utility (-.42), and the
squared level-index distance (6). The existing intelligence-by-cloud
interaction changes from a -0.55 A--B gap in the plus task to +0.55 in the
minus task. A 200-replication A/B/opt-out simulation with 400 respondents gives
.025 null rejection and .995 rejection when that interaction is active. This
establishes semantic feasibility inside the supplied manuscript; it is still a
planning result without a human paired DCE.

## Five-task application grid

An exhaustive profile audit finds 677 unique reflections satisfying exact
candidate-menu preservation, equal plus/minus squared geometry, and an odd
existing nonlinear interaction. A pre-outcome grid selects five tasks with
candidate A--B gaps .20, .40, .60, .80, and 1.00. Each has an odd interaction
signal of .55. This provides a feasible multi-task supplement template; pilot
semantic checks and final power analysis remain open.

## Fibre-optimal design criterion

The application design now has a pre-outcome information rule
(I_{odd}=p(1-p)s^2), with an additional exact antisymmetry constraint on the
omitted interaction. Of 648 pure-odd candidate reflections, the selected grid
has candidate gaps approximately .20, .35, .55, .75, and .95 and odd signal .90
in every task. This gives the supplement a reproducible design criterion beyond
hand-picked examples.

## Calibration uncertainty gate

A coefficient-perturbation audit shows null rejection .030 at candidate
coefficient noise SD .02 and .150 at SD .05 for the selected smart-device
pair. The method therefore requires an independent calibration pilot and a
predeclared candidate-gap tolerance. This is an identified empirical gate,
not a hidden assumption of exact coefficients.

## Transformation-group collision audit

A 2024 Journal of Econometrics paper on latent utility and permutation
invariance is a closer theoretical neighbor than the prior search found. The
novelty claim has been narrowed accordingly: the paper will present an observed
attribute-space DCE specification audit with candidate menu preservation and
odd/even response separation, not a general latent transformation-invariance
theorem.

## JOCM positioning audit

Recent JOCM papers reinforce the fit of a design-based specification audit:
Bayesian design algorithms, ordering-effect reviews, randomized level-overlap
studies, interpretable ML, and model-building workflow analyses all treat
experimental validity and transparent design as central. The submission should
lead with the parity-fibre audit and information criterion; the smart-device
simulation remains a testbed and the original XGBoost crossover remains a
benchmark.

## Matched multi-task design comparison

At 150 respondents and five tasks, a weak omitted interaction gives .490 power
for the ordinary candidate grid and .870 for the fibre-optimal grid; the
combined omitted-plus-even-scale rates are .335 and .785. Null rejection is
.015 and .055, and even-scale rejection is .040 versus .045. Strong omitted
signals saturate at 1.000 for both. The information criterion is retained as a
weak-signal design improvement, not a universal power-dominance claim.

## Pre-registration specification

The smart-device supplement now has a frozen protocol covering independent
calibration, candidate-gap and geometry tolerances, pure-odd screening, one-
member randomization, global cluster sign-flip inference, max-T secondary task
contrasts, even-component diagnostics, and order/semantic negative controls.
`analysis/verify_smart_device_preregistration.py` checks the five-task grid and
reports `preregistration_invariants=PASS`.

## Manuscript-ready innovation section

A direct prose-and-equation draft now states the research problem, reflected
utility-fibre estimand, moderate-utility extension, fibre information criterion,
smart-device implementation, contribution claims, and failure boundaries. It
is saved as `manuscript/innovation_section_draft.md` and can anchor the Word
manuscript rewrite.

## Revision package round thirteen

A complete manuscript-level package now specifies the recommended title,
abstract, section order, contribution claims, result hierarchy, tables, figures,
and material to demote. The main paper will lead with the design-based
specification audit; the XGBoost crossover will appear as a bounded predictive
benchmark.

## Round-fourteen anti-stitch and complexity-balance audit

The nearest-neighbour review found that the method cannot be presented as a
general symmetry theory. Healy and Leo's 2026 Journal of Economic Theory paper
characterizes experiments that test deterministic preference models. Shubatt
and Yang's comparison-complexity model shows that equal utility gaps can still
produce unequal choice rates when value-weighted (L_1) distance differs.
McGranaghan et al. (2024) show that differential noise can bias paired-choice
tests. These papers change the design requirements and narrow the claim.

The core contribution is now one design object: a candidate-preserving fibre
with a matched nuisance signature. The signature requires equality of the
candidate menu vector, weighted (L_1) comparison distance, changed-attribute
count, raw level-change load, and component-sign counts. Reflection, odd/even
contrasts, and task optimization follow from this object; they are not separate
borrowed theories.

The original smart-device grid failed the full signature audit. None of its
five selected pairs matched the weighted (L_1) profile. A corrected search
finds 58 pure-odd nuisance-balanced reflections and selects gaps near .32,
.38, .57, .80, and .98. In a binary moderate-utility stress test with 600
respondents, 200 replications, and 999 cluster sign flips, the loose grid
rejects a complexity-only null in .185 of samples, while the balanced grid
rejects in .050. With the omitted interaction added, both designs reject in
1.000 of samples. The result is a direct negative-control justification for the
new balance condition.

The innovation section, preregistration specification, revision package, and
reproducibility scripts now use the nuisance-balanced design. The remaining
empirical gate is a human or external-data DCE with independent calibration,
semantic screening, and the frozen assignment protocol.

## Round-sixteen assignment and heterogeneity audit

The first versions of the profile and multi-task benchmarks drew a new
orientation inside each task while using respondent-cluster sign flips. Those
versions are discarded. The corrected scripts draw one orientation coin per
respondent block and share it across the five tasks. The sign-flip reference
now matches the assignment unit.

With 600 respondents, 200 replications, and 999 flips, the corrected profile
has global null rejection .035 and max-T null rejection .050. A localized
omitted direction gives global power .330, profile power .885, and localization
.990. A random-taste boundary control gives rejection .085, .095, and .085 at
coefficient standard deviations 0, .1, and .2. These values keep the central
claim bounded: the design audits a candidate relation and does not identify a
unique omitted mechanism under unmodelled heterogeneity. The full audit is in
`manuscript/novelty_audit_round16.md`.

## Round-eighteen structural-fibre redesign

The topic audit in `manuscript/jocm_topic_gap_audit_round18.md` queried the
2021--2026 JOCM DOI corpus and separated four active streams: assisted
specification and ML, efficient DCE design, behavioural extensions, and
specification/invariance tests. The unresolved problem is pre-outcome
coverage of departures from a coarsened candidate basis.

The structural search in `analysis/structural_fibre_design.py` constructs
candidate-preserving pairs by exact equality of the candidate sufficient
statistic vector. This makes the equality valid for every candidate
coefficient vector. It enumerates 1,024 admissible reflections, enforces equal
raw comparison signatures, and selects a five-task block with a maximin local
coverage criterion. The selected gaps are .22, .36, .56, .66, and .82; its
coverage eigenvalues are 1.648, 6.063, and 7.352.

In 200-replication simulations with 600 respondents and 499 sign flips, the
structural maximin block has minimum feature power .67 at zero coefficient
perturbation and .65 at perturbation standard deviation .05. A random
structural block has minimum feature power .40 and .45 under the same two
conditions. Null rejection for the maximin block is .045, .035, .045, and .020
at coefficient standard deviations 0, .02, .05, and .10.

## Round-nineteen independent-core audit

The overlap audit found a real adjacent result: Errore, Nachtsheim, and Li's
2013 conference abstract already studies maximin model-robust DCE designs for
main effects and interactions. The manuscript therefore drops any generic
“first maximin DCE” claim. The primary contribution is now the fibre quotient
completeness operator and its rank/null-space certificate; maximin selection is
a secondary conditioning rule. The full collision audit is in
`manuscript/literature_overlap_audit_round19.md`, and the proposition and
smart-device calculation are in
`manuscript/fibre_quotient_completeness_round19.md`.

The remaining submission gate is a preregistered paired DCE or documented
external dataset with independent calibration, semantic screening, and a
direct comparison against an efficient block and ordinary added-term/residual
diagnostics.

## Round-twenty-two sufficiency reformulation

The paper's main claim has been tightened to a fibre-conditional sufficiency
audit. The null is $Y\perp X\mid\phi(X)$, with the conditional response
function left unrestricted. A finite-support exposure matrix measures the
choice-probability-weighted conditional variance of a declared residual
direction. Directions that are functions of $\phi$ have zero exposure and are
kept inside the null; they are functional-form questions within the candidate
summary.

The proof, multinomial implementation, and finite-support counterexample are
now recorded in `manuscript/fibre_sufficiency_exposure_round22.md` and
`manuscript/fibre_sufficiency_exposure_proof_round22.md`. The counterexample
has candidate precision 1.00 and exposure 0.00 for an endpoint design, versus
precision 0.50 and exposure 0.50 for a full factorial and precision 0.25 and
exposure 0.75 for a middle-fibre allocation. This supplies the concrete
increment over candidate-precision design.

The eight-arm multinomial gate now preserves every pairwise candidate gap
exactly, has residual rank 3, and separates declared scale/framing from an
unlisted process in 200 replications. These are structural and planning
results. The empirical gate still requires a paired DCE or a matched public
dataset, and the Scopus/Google Scholar institutional exports remain open.

## Live readiness bar (2026-10-07)

```
fibre-sufficiency contribution      [#########.] 90%
finite-support proof/certificate    [########..] 85%
multinomial full-menu gate          [########..] 88%
overlap audit                       [#######...] 75%
Scopus/Google Scholar exports       [###.......] 30%
paired human DCE validation         [##........] 20%
full manuscript integration         [#####.....] 55%
submission package                  [###.......] 35%
overall submission readiness        [######....] 60%
```

The remaining 40% is evidence work: institutional literature exports, a
paired or matched DCE, and the final Word-manuscript integration. The current
results support a credible method-and-design contribution; they do not justify
claiming empirical confirmation or priority over every related test.
