# Fibre quotient completeness: the independent core

## Central object

Let (x) denote a raw alternative profile and let (b(x)) be the candidate
sufficient-statistic vector. A fibre is a set of profiles with the same
(b(x)). For a reflected task (t=(A_+,A_-,B_+,B_-)), impose

\[
b(A_+)=b(A_-),\qquad b(B_+)=b(B_-).
\]

For every candidate coefficient vector \(\beta\), the candidate utility
difference is unchanged by the reflection. The task removes the
candidate utility from the odd contrast without calibrating to an estimated
\(\hat\beta\).

## Fibre-quotient operator

For a local departure (h) from the candidate basis, define

\[
D_t(h)=\frac12\big\{[h(A_+)-h(B_+)]-[h(A_-)-h(B_-)]\big\}.
\]

For a declared finite-dimensional departure space (H=\operatorname{span}
\{h_1,\ldots,h_r\}), the task block (S) induces the matrix

\[
D_S=\left[D_t(h_\ell)\right]_{t\in S,\ell=1,\ldots,r}.
\]

The **fibre quotient completeness** of (S) is

\[
\operatorname{FQC}(S;H)=\operatorname{rank}(D_S).
\]

The available-support benchmark is

\[
r_{\max}=\operatorname{rank}(D_{\mathcal X}),
\]

where \(\mathcal X\) contains every admissible structural fibre in the
attribute space. A block is **complete for (H)** when

\[
\operatorname{FQC}(S;H)=r_{\max}.
\]

The null space of (D_S) is the exact local blind space. If a direction lies
in the null space for every admissible task, it is structurally invisible on
the chosen attribute support and cannot be recovered by increasing the sample
size.

## Proposition and proof sketch

**Proposition.** Suppose the candidate model is linear in (b(x)), and each
reflected pair preserves (b(x)) alternative by alternative. Under a common
random-utility error law and a respondent-level random reflection sign, the
expected odd choice contrast is zero for the candidate model. For a local
departure (h\in H), the first-order expected odd contrast is proportional to
\(D_S(h)\). Consequently, two departures in the same coset of
\(\ker(D_S)\) are observationally equivalent to first order on block (S\),
and complete coverage of (H\) is equivalent to the full available rank
\(r_{\max}\).

**Proof sketch.** Equality of (b(x)) makes the candidate systematic utility
difference identical under the plus and minus reflection, for every \(\beta\).
The random sign reverses the odd component while preserving the even component,
so the candidate contribution cancels in expectation. A perturbation
\(V(x)=b(x)'\beta+\eta h(x)\) changes the logit probability by its derivative
\(p(1-p)\) times the reflected utility contrast; taking the plus-minus
difference yields \(D_t(h)\). Stacking tasks gives \(D_S h\). Linear algebra
then gives the rank and null-space statements.

This proposition is the paper's independent theoretical core. The maximin
criterion is applied only after the rank condition is checked, to improve
conditioning and power among complete blocks.

## Smart-device calculation

For the coarsened basis (b=(m+s,i+e,d,q)), the full structural pool contains
1,024 reflected tasks. The declared departure space has three directions:
total quadratic curvature, a midpoint mode-support interaction, and an
intelligence-cloud interaction. The full pool has rank 3. A three-task subset
(pool indices 39, 7, and 112) already has rank 3, with eigenvalues
approximately (0.483, 2.217, 5.055). The selected five-task maximin block
(indices 201, 59, 557, 997, 764) also has rank 3 and covers all five
probability-gap bins; its information eigenvalues are (1.648, 6.063, 7.352).

This separates two claims that should not be conflated. Rank completeness says
the declared directions are algebraically testable. The positive minimum
eigenvalue says the chosen block is well-conditioned for detecting the weakest
direction. The endpoint mode-support interaction is excluded because it is
constant on the (m+s) fibre; that exclusion is a structural result.

## Incremental comparator

The earlier paired-task benchmark supplies the required same-budget comparison
with ordinary diagnostics. When observational tasks are concentrated near the
centre of the decomposition coordinate, the added-term observational LR and
the out-of-fold residual proxy reject the omitted decomposition in .06 and .03
of 200 replications. The parity-fibre design rejects in 1.00. Under an
even-scale nuisance, the LR and residual rates are .03 and .06, while the
parity-fibre rate is .05. The comparison has a narrow interpretation: the
fibre design creates support for the declared odd departure, while the scale
negative control shows that an odd contrast is not a general mechanism
separator. These values come from
`results/parity_incremental_benchmark.csv` and should be reported alongside
the structural rank certificate.

## Why this is a single idea

The paper does not join unrelated tests. It starts from one question: which
departures remain observable after conditioning on a candidate utility basis?
The fibre creates the quotient, the odd contrast is its observable, and rank
defines completeness. Nuisance balance and maximin selection are constraints
and conditioning rules for that same operator. Complexity, presentation, and
ordinary efficient-design results enter as negative controls and comparators.
