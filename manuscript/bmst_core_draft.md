# Testing coarsened utility specification with anchored behavioral metamorphic relations

## Working title

**Anchored behavioral metamorphic specification testing for discrete choice
models: randomized utility fibres for coarsened-basis adequacy**

## Abstract

Choice modellers often compare out-of-sample fit after selecting a utility
specification. Similar fit can conceal different behavioral implications, and
single-task diagnostics do not provide a test oracle when the correct utility
function is unknown. This paper develops an anchored version of Behavioral
Metamorphic Specification Testing (BMST) for candidate bases that coarsen the
raw attribute space. A frozen candidate generates task versions with the same
full menu of candidate utility differences while the raw decomposition changes.
The instrument assigns one member of each pair to each respondent, includes a
zero-candidate-difference anchor, and holds pairwise raw geometry fixed in its
primary control arm. Respondent-cluster randomization supplies a finite-sample
reference distribution. The estimand is a violation of candidate-implied
choice-probability equivalence on a declared, nontrivial task fibre.

In a 100-replication anchored-fibre benchmark, the ordinary observational LR
test and a residual-learning proxy reject at .04 and .04 under an omitted
decomposition term because the original support is nearly flat in the omitted
direction. The anchored fibre test rejects at .83 on the zero-gap,
geometry-preserving arm and 1.00 on the geometry-changing nonzero-gap arm. In
the complexity-only control, the zero-gap and geometry-preserving arms reject
at .02 and .07, while the deliberately geometry-changing arm responds at
.41. These figures are planning evidence, not an empirical application. They
show why the paper reports a relation violation together with complexity
controls instead of assigning every rejection to utility misspecification. A
paired-task supplement is required for the empirical study.

## 1. Research problem

Choice-model specification is usually evaluated through likelihood,
information criteria, prediction, or a comparison between competing behavioral
models. These outcomes answer whether a fitted model predicts the observed
tasks. They do not directly test whether a candidate utility library treats
attribute decompositions that it considers equivalent in the same way.

The gap matters when the candidate fits aggregate choices well. A flexible
learner can absorb an omitted relation without explaining its behavioral
source. A richer structural model can improve fit while leaving the analyst
uncertain about which observable transformation changed the implied choice
probabilities. A specification test needs a relation that the candidate
itself entails and a design that evaluates that relation on new respondents.

BMST treats a candidate utility representation as a source of test oracles.
The method constructs a follow-up task whose candidate alternative differences
equal those of the original task. The candidate predicts exchangeability of
the two task versions. A systematic difference in observed choices is evidence
that the candidate basis is insufficient for the tested transformation.

## 2. Contribution and boundary

The paper makes one method contribution and two supporting contributions.

1. BMST transfers metamorphic test-oracle construction into discrete-choice
   specification and restricts it to candidate bases with nontrivial fibres.
   Its domain-specific relation is the Utility-Fibre Invariance Test (UFIT).
2. The anchored instrument uses a zero-candidate-difference task and a
   geometry-preserving common translation to separate a scale/complexity
   explanation from a decomposition-direction violation.
3. The benchmark compares the designed probe with an observational LR test and
   an out-of-fold residual learner, then reports scale, taste, order, and
   comparison-complexity controls.

BMST tests a declared choice-probability relation on a declared intervention
domain. It is not run when the candidate fibre is a singleton. It does not
introduce invariance theory, provide a universal MNL misspecification test,
recover a unique omitted term, or guarantee external validity beyond the tested
task fibre.

## 3. Candidate-preserving task relation

For alternative \(j\), let \(b_j(x)\in\mathbb{R}^p\) denote the candidate basis
and define the candidate utility

\[
V_j(x;\beta)=b_j(x)^\top\beta .
\]

The full menu coordinate is

\[
\phi(x)=\{b_j(x)-b_k(x):j<k\}.
\]

A transformation \(T\) is candidate-preserving when \(\phi(Tx)=\phi(x)\). The
candidate multinomial-logit probabilities are

\[
p_j(x;\beta)=\frac{\exp\{V_j(x;\beta)\}}
{\sum_k\exp\{V_k(x;\beta)\}}.
\]

The candidate-preserving condition implies
\(p_j(Tx;\beta)=p_j(x;\beta)\) for every \(j\) and every \(\beta\). The converse
also holds because equality of softmax odds implies equality of every
alternative-pair basis difference. The relation therefore comes from the
candidate representation rather than from an arbitrary matching rule. Before
the test, the feasible transformation set is screened for a nonzero direction
that preserves \(\phi\). A full raw-attribute basis with no such direction has
singleton fibres and is outside the primary estimand.

For the data-generating choice process \(P^*\), BMST targets

\[
\Delta_j(T)=P^*(Y=j\mid x)-P^*(Y=j\mid Tx).
\]

The null is \(\Delta(T)=0\) for the declared transformation. Rejection
establishes a violation of the candidate relation. It leaves the mechanism
unresolved when nonlinear utility, scale, random taste, presentation, or
sequence effects can produce the same violation.

## 4. Randomized one-member assignment

Each respondent receives one member of every focal pair. Let
\(Z_n\in\{-1,+1\}\) be a respondent-level orientation coin, held constant
across the focal pairs. The pair member is assigned before the respondent
chooses, and the respondent never sees both members of a pair. Candidate
parameters are estimated on a development fold and frozen before the test
fold is evaluated.

The primary binary arm adds two controls. A zero-candidate-difference pair has
\(\Delta V=0\); under symmetric binary errors its probability is one half for
every positive scale. A common raw-attribute translation applied to both
alternatives preserves their pairwise raw geometry and tests the candidate
relation without changing the similarity/dominance distance used by
comparison-complexity models. An alternative-specific decomposition move is
reported as a complexity arm, not as a direct utility diagnosis.

For focal tasks \(\mathcal Q_n\), define the cluster score

\[
S_n=\sum_{q\in\mathcal Q_n}Z_n
\{e(Y_{nq})-\widehat p_{nq}\}.
\]

Under the candidate null, the joint score distribution is invariant to
\(Z_n\mapsto-Z_n\). Conditional on the potential choices and the frozen
candidate, all respondent-level orientation assignments are equally likely.
Flipping the observed \(S_n\) vectors therefore supplies an exact
randomization reference for a statistic such as
\(\|\bar S\|^2\). Estimating the candidate on the test respondents or allowing
the fitted candidate to depend on orientation breaks this reference.

## 5. Simulation design

The original smart-device benchmark uses three-alternative tasks, additive
candidate attributes, alternative constants, and common shifts in time, cost,
and both attributes. The revised primary benchmark uses binary tasks with a
coarsened candidate burden \(s=a_1+a_2\) and an omitted decomposition coordinate
\(z=a_1-a_2\). Conditions vary one mechanism at a time:

- additive candidate;
- omitted quadratic utility;
- omitted attribute interaction;
- random linear cost sensitivity;
- task-specific error-scale drift;
- order-dependent choice bonus.

The observational support keeps \(z\) near zero. The fibre supplement moves
along \(s\)-constant directions and compares geometry-preserving and
geometry-changing decompositions. The LR comparator adds \(z^2\) to the
candidate likelihood. The residual comparator searches out-of-fold residuals
with a small random-feature basis in \(z\).

The main assignment benchmark uses 600 respondents, one member per pair,
three focal pairs, four nuisance tasks, 100 replications, and 499
randomization draws per replication. The test uses a respondent-grouped
development/test split. The negative-control benchmark uses 300 respondents
and adds the scale and order conditions.

## 6. Results

| Benchmark condition | Rejection rate |
|---|---:|
| Additive null, negative-control benchmark | .04 |
| Omitted nonlinearity, negative-control benchmark | .68 |
| Random linear taste, negative-control benchmark | .03 |
| Task-specific scale drift, negative-control benchmark | .23 |
| Order effect, negative-control benchmark | 1.00 |
| Additive null, one-member assignment | .05 |
| Omitted nonlinearity, one-member assignment | .79 |
| Random linear taste, one-member assignment | .05 |
| Omitted interaction, one-member assignment | 1.00 |

The null and random-taste conditions calibrate the original relation test. The
anchored-fibre benchmark adds the following planning result:

| condition | LR on observational support | residual learner | zero-gap GP | gap GP | zero-gap GC | gap GC |
|---|---:|---:|---:|---:|---:|---:|
| null | .04 | .03 | .06 | .07 | .03 | .03 |
| omitted decomposition | .04 | .04 | .83 | .56 | .50 | 1.00 |
| complexity only | .03 | .06 | .06 | .07 | .02 | .41 |

Here GP denotes geometry-preserving and GC geometry-changing. The table uses
100 replications with 400 respondents and 199 sign-flip draws. It is a design
comparison, not an empirical application. It shows the intended logic: the
observational diagnostics have little power on the omitted direction, the
geometry-preserving fibre detects the decomposition error, and the
complexity-only control is concentrated in the geometry-changing nonzero-gap
arm while the zero-gap anchor remains calibrated.

## 7. Empirical instrument

The current Swissmetro file does not provide enough exact randomized task
pairs. A broader public-data audit reached the same boundary: repeated tasks in
open DCE repositories do not document the candidate-preserving transformation
and one-member assignment required by the estimand. Those data can support
descriptive or predictive transfer, but they cannot replace the instrument.
The empirical study therefore needs a forced-choice supplement or a different
dataset with the required assignment structure. The supplement should include
an explicitly coarsened candidate basis, a pretest showing a nontrivial fibre,
at least one zero-gap anchor, a geometry-preserving translation arm, a
geometry-changing decomposition arm, a one-member-per-pair assignment, an
additive placebo, a random-taste condition, and an order or carryover placebo
in a separate arm. Response time, confidence, and the raw pairwise distance
index are recorded before the analysis plan is unlocked.

An opt-out requires a fully defined profile if the relation is intended to
include product-versus-opt-out differences. A forced-choice block provides the
cleaner first test of in-product utility specification.

## 8. Submission claim

The supported claim is that anchored BMST provides a finite-sample,
candidate-conditioned specification audit for a declared nontrivial fibre of a
coarsened utility basis. The empirical paper must
report usable-pair counts, the predeclared transformation, assignment balance,
null calibration, negative controls, and the action for a mixed violation.
Survival of the test supports the tested relation and task domain; it does not
establish global model correctness.
