# Testing utility specification with behavioral metamorphic relations

## Working title

**Behavioral metamorphic specification testing for discrete choice models:
randomized task fibres for utility-basis adequacy**

## Abstract

Choice modellers often compare out-of-sample fit after selecting a utility
specification. Similar fit can conceal different behavioral implications, and
single-task diagnostics do not provide a test oracle when the correct utility
function is unknown. This paper develops Behavioral Metamorphic Specification
Testing (BMST), a design-based audit for candidate utility bases. A frozen
candidate generates a pair of task versions whose alternative-specific
candidate utility differences are identical while the raw attribute
decomposition changes. The candidate therefore predicts equal choice
probabilities across the pair. We assign one member of each pair to each
respondent, aggregate respondent-cluster residual scores, and use the
randomized orientation to construct a finite-sample reference distribution.
The test estimand is a violation of candidate-implied choice-probability
equivalence on a declared task fibre.

In a 100-replication negative-control benchmark, the additive null rejection
rate is .04, omitted nonlinearity is detected at .68, random linear taste at
.03, task-specific error-scale drift at .23, and an order effect at 1.00. A
600-respondent assignment benchmark gives .05 null rejection, .79 power
against omitted nonlinearity, .05 rejection under random linear taste, and
1.00 power against an omitted interaction. The results establish a
specification relation test. They do not identify a unique omitted mechanism,
latent scale, or global model truth. A paired-task supplement is required for
the empirical application.

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
   utility specification. Its domain-specific relation is the Utility-Fibre
   Invariance Test (UFIT).
2. The paper gives a candidate-basis equivalence proposition and a
   respondent-cluster randomization reference for a one-member-per-pair
   instrument.
3. The benchmark separates functional-form departures from random linear
   taste, task-specific scale drift, and order effects.

BMST tests a declared choice-probability relation on a declared intervention
domain. It does not introduce invariance theory, provide a universal MNL
misspecification test, recover a unique omitted term, or guarantee external
validity beyond the tested task fibre.

## 3. Candidate-preserving task relation

For alternative \(j\), let \(b_j(x)\in\mathbb{R}^p\) denote the candidate basis
and define the candidate utility

\[
V_j(x;\beta)=b_j(x)^\top\beta .
\]

The vector of alternative differences is

\[
d(x)=\{b_j(x)-b_k(x):j<k\}.
\]

A transformation \(T\) is candidate-preserving when \(d(Tx)=d(x)\). The
candidate multinomial-logit probabilities are

\[
p_j(x;\beta)=\frac{\exp\{V_j(x;\beta)\}}
{\sum_k\exp\{V_k(x;\beta)\}}.
\]

The candidate-preserving condition implies
\(p_j(Tx;\beta)=p_j(x;\beta)\) for every \(j\) and every \(\beta\). The converse
also holds because equality of softmax odds implies equality of every
alternative-pair basis difference. The relation therefore comes from the
candidate representation rather than from an arbitrary matching rule.

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

The benchmark uses three-alternative tasks, additive candidate attributes,
alternative constants, and common shifts in time, cost, and both attributes.
Conditions vary one mechanism at a time:

- additive candidate;
- omitted quadratic utility;
- omitted attribute interaction;
- random linear cost sensitivity;
- task-specific error-scale drift;
- order-dependent choice bonus.

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

The null and random-taste conditions calibrate the relation test. The
nonlinear and interaction conditions show sensitivity to out-of-library
utility. Scale and order conditions demonstrate the interpretation boundary:
BMST detects a candidate-relation violation, and mechanism attribution needs
predeclared controls.

## 7. Empirical instrument

The current Swissmetro file does not provide enough exact randomized task
pairs. The empirical study therefore needs a forced-choice supplement or a
different dataset with the required assignment structure. The supplement
should include at least two directional shifts, one joint shift, a
one-member-per-pair assignment, an additive placebo, a random-taste condition,
and an order or carryover placebo in a separate arm.

An opt-out requires a fully defined profile if the relation is intended to
include product-versus-opt-out differences. A forced-choice block provides the
cleaner first test of in-product utility specification.

## 8. Submission claim

The supported claim is that BMST provides a finite-sample, candidate-conditioned
specification audit for a declared choice-task fibre. The empirical paper must
report usable-pair counts, the predeclared transformation, assignment balance,
null calibration, negative controls, and the action for a mixed violation.
Survival of the test supports the tested relation and task domain; it does not
establish global model correctness.
