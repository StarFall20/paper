# Local quotient theorem for NO-FQC

This note proves the nuisance-orthogonal extension. The primary finite-support
sufficiency and conditional-variance result is in
`fibre_sufficiency_exposure_proof_round22.md`.

## Setup

Let a reflected task block contain m task arms. For each arm, let Delta_t be a
vector of observed choice-probability contrasts. Stack the contrasts as Delta
in R^m. Let eta in R^r index local utility departures and nu in R^q index the
declared nuisance mechanisms. Around the candidate model, assume the mean map
is differentiable:

    E[Delta | eta, nu] =
        D eta + G nu + o(||eta|| + ||nu||).

The covariance matrix Omega of Delta is positive definite and W=Omega^(-1).
The columns of G are linearly independent after redundant nuisance columns
are removed.

The candidate-preserving fibre construction gives D. The nuisance library
gives G. These matrices are generated before the response outcomes are
observed.

## Theorem 1: local observational quotient

Define

    P_G = G (G' W G)^(-1) G' W,
    M_G = I - P_G.

Two local utility directions eta_1 and eta_2 are observationally equivalent
up to a declared nuisance change if and only if

    M_G D (eta_1 - eta_2) = 0.

Consequently, the locally distinguishable utility space has dimension

    rank(M_G D),

and its blind space has dimension

    r - rank(M_G D).

### Proof

The local mean difference between the two utility directions is
D(eta_1-eta_2). A nuisance change can reproduce that difference exactly when
there exists a vector a with

    D(eta_1-eta_2)=G a.

The right-hand side is the column space of G. Weighted projection removes
that column space, so the equality is equivalent to

    M_G D(eta_1-eta_2)=0.

The quotient map from R^r to the residual contrast space is M_G D. Its image
dimension is the rank, and rank-nullity gives the blind-space dimension.

## Theorem 2: locally optimal nuisance-orthogonal contrast

For a single departure direction g, consider all linear contrasts w' Delta
that satisfy

    G' w = 0,
    w' Omega w = 1.

If g' W M_G g > 0, the contrast with largest local signal is

    w_star = W M_G g / sqrt(g' W M_G g).

It has

    G' w_star = 0,
    w_star' Omega w_star = 1,
    w_star' g = sqrt(g' W M_G g).

### Proof

Because W M_G is symmetric and G' W M_G=0, the first equality holds. The
variance normalization follows from

    (W M_G g)' Omega (W M_G g)
      = g' M_G' W M_G g
      = g' W M_G g.

For any admissible w, nuisance orthogonality gives w' g = w' M_G g.
Cauchy--Schwarz in the Omega inner product bounds its absolute value by
sqrt(g' W M_G g). Equality holds at w_star.

The scalar kappa(g)=g' W M_G g is the local information remaining after the
declared nuisance projection. For N independent respondent clusters and a
local departure of size eta, the first-order noncentrality is
sqrt(N eta^2 kappa(g)), up to the cluster variance correction.

## Theorem 3: complete task blocks

Let D_X and G_X describe the full admissible structural pool. A selected block
S is complete for a departure space H under nuisance library N when

    rank(M_{G_S} D_S) = rank(M_{G_X} D_X).

No increase in respondent count can recover a direction in
ker(M_{G_X} D_X). A support expansion X' repairs a blind direction h only
when

    M_{G_{X'}} D_{X'} h != 0.

This gives an exact design-stage stop rule: a power calculation for h is
meaningful only after its residual signal is nonzero.

## Multinomial implementation

For a J-alternative menu, Delta must contain the full vector of independent
probability contrasts, or an equivalent multinomial score vector. The
covariance Omega must include within-task covariance across alternatives and
respondent-level clustering. Preserving one A--B gap does not preserve the
full menu prediction and does not satisfy the theorem.

The binary zero-gap fibre is a special case. Under a symmetric error law, zero
candidate gap fixes the binary probability at one half for every positive
common scale. It supplies a scale-insensitive anchor, while the multinomial
theorem supplies the general full-menu condition.

## Interpretation boundary

The theorem identifies local distinguishability modulo the declared nuisance
span. It does not prove that the nuisance library contains every behavioural
mechanism. An unlisted process direction outside the span of G can still
produce a residual rejection. The preregistered experiment must include at
least one such out-of-library mechanism and report it as a boundary result.
