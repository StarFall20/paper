# Paired-task supplement design

## Research question

The supplement asks whether a fitted utility specification makes choice
probabilities invariant to a transformation that preserves the candidate's
**full menu difference vector**. The question targets adequacy of a coarsened
candidate basis. It does not ask whether two arbitrary tasks look similar or
whether a flexible learner predicts better.

## Eligibility screen

Let \(b_j(x)\) be the candidate basis for alternative \(j\), and let

\[
\phi(x)=\{b_j(x)-b_k(x):j<k\}.
\]

Before generating tasks, search the feasible attribute domain for a nonzero
direction \(h\) with \(\phi(x+h)=\phi(x)\). A full raw-attribute linear basis
with no such direction has singleton fibres and is outside the primary
estimand. A composite burden such as \(s=a_1+\kappa a_2\) provides a simple
screening case; the fibre direction changes the decomposition while holding
\(s\) fixed.

For multinomial tasks, preserve every element of \(\phi\), not one chosen
alternative's gap. Preserve alternative labels, availability, and the
predeclared framing.

## Task arms

Use a binary forced-choice block first. For each base task, construct:

1. **Zero-gap anchor.** Set the candidate utility gap to zero. With symmetric
   binary errors, the candidate probability is \(1/2\) for every positive error
   scale. This arm anchors the scale explanation.
2. **Geometry-preserving translation (GP).** Apply the same raw shift to every
   alternative. Pairwise raw differences and any pairwise-distance complexity
   index stay fixed while the candidate-preserving relation is tested.
3. **Geometry-changing decomposition (GC).** Move along a candidate fibre that
   changes the raw decomposition or pairwise distance. Treat a response here as
   a comparison-complexity result unless the geometry control supports a
   utility interpretation.
4. **Additive placebo.** Use a candidate-preserving transformation generated
   under the additive data-generating process.
5. **Order/carryover placebo.** Reverse presentation order in a separate block.

Record raw pairwise distances, response time, confidence, order, and the exact
candidate vector for every task. Use at least three focal pairs so the power
curve is in the planned range. Keep a one-pair arm only as a burden or
sensitivity condition.

## Assignment

Assign one member of each focal pair to each respondent. Draw one orientation
coin \(Z_n\in\{-1,+1\}\) per respondent and hold it across focal pairs. Balance
the coin across respondent groups. A respondent never sees both members of a
focal pair in the primary arm, which removes within-respondent memory as the
primary explanation. A within-respondent paired block can be retained as a
sensitivity arm with a preregistered carryover placebo.

If the study includes an opt-out, give the opt-out a visible profile and apply
the same transformation to every alternative. A product-only shift changes
product-versus-opt-out differences and is not a candidate-preserving null for a
model that leaves the opt-out unprofiled.

## Estimation and randomization reference

Split respondents before estimation. Fit the candidate on a development fold,
freeze its parameters, and evaluate held-out respondents. For focal tasks
\(\mathcal Q_n\), define

\[
S_n=\sum_{q\in\mathcal Q_n}Z_n
\{e(Y_{nq})-\widehat p_{nq}\}.
\]

Under the candidate null, the joint score distribution is invariant to
\(Z_n\mapsto-Z_n\). Flip the complete respondent-level score vector to form the
randomization reference for a predeclared statistic such as
\(\|\bar S\|^2\). Do not estimate the candidate on the held-out orientation or
choose the transformation after seeing outcomes.

For a multinomial extension, stack the score components for all alternatives
and use the complete menu vector. Report one global test plus predeclared
shift-specific contrasts with multiplicity control. Do not interpret a single
pairwise contrast as a full-menu result.

## Comparators and power

Run the observational LR and out-of-fold residual proxy on the same simulated
and empirical respondent budgets. Construct the observational support so the
omitted direction is weakly represented. This makes the comparison
informative without claiming that the designed test is universally more
powerful.

Use the locked planning curve as the starting target: with one focal pair,
nonlinear power is .20, .42, and .70 at 150, 300, and 600 respondents; with
three focal pairs it is .90, 1.00, and 1.00. Re-estimate power after piloting
the actual response burden and report the Monte Carlo uncertainty.

## Falsification and interpretation

The supplement must include additive, random linear taste, task-specific
scale, order/carryover, and complexity-only conditions. Under the additive
placebo, rejection should be near the declared level. A rejection establishes
a violation of the candidate-conditioned relation on the tested fibre. It does
not identify a unique omitted term, and non-rejection does not establish global
sufficiency.

The preregistration must lock the candidate basis, feasible shifts, geometry
index, assignment, fold split, statistic, sign-flip draws, and action for a
mixed signature before outcomes are inspected.
