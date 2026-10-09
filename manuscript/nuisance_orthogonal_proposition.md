# Local proposition for nuisance-orthogonal fibre contrasts

## Setup

Let (d=1,\ldots,m) index feasible paired task arms. For each arm, the two
versions preserve the candidate menu-difference vector

\[
\phi(x)=\{b_j(x)-b_k(x):j<k\}.
\]

The fibre is nontrivial when at least one raw attribute can change while
\(\phi\) remains fixed. Let \(\widehat\Delta\) collect the randomized,
respondent-clustered probability contrasts across the \(m\) arms. In a local
neighbourhood of the maintained candidate model, write

\[
E[\widehat\Delta]
 = \eta g_\eta + G_\nu\nu + r(\eta,\nu),
\qquad
\|r(\eta,\nu)\|=o(|\eta|+\|\nu\|),
\]

where \(\eta\) is the omitted utility direction and \(\nu\) contains the
declared nuisance directions: shared framing/order shifts, geometry-dependent
scale, or additional pre-specified processing terms. Assume the local
covariance \(\Omega\) is positive definite and set \(W=\Omega^{-1}\). The
nuisance columns must have full rank after redundant columns are removed.

## Proposition

Define

\[
P_\nu=G_\nu(G_\nu^\top W G_\nu)^{-1}G_\nu^\top W,
\qquad
r_\eta=(I-P_\nu)g_\eta.
\]

If \(r_\eta\ne0\), the unit-variance contrast that maximizes local signal while
remaining locally insensitive to every declared nuisance direction is

\[
w_\star
 = \frac{Wr_\eta}{\sqrt{r_\eta^\top W r_\eta}}.
\]

It satisfies

\[
G_\nu^\top w_\star=0,
\qquad
w_\star^\top\Omega w_\star=1,
\qquad
w_\star^\top g_\eta=\sqrt{r_\eta^\top W r_\eta}.
\]

Consequently, for a local nuisance perturbation \(\nu\),

\[
E[w_\star^\top\widehat\Delta]
 = \eta\sqrt{r_\eta^\top W r_\eta}
   + o(|\eta|+\|\nu\|),
\]

so the nuisance contribution vanishes to first order. The quantity

\[
\kappa_\eta=g_\eta^\top W(I-P_\nu)g_\eta
\]

is the local information available for the omitted direction after nuisance
projection. When the respondent-level contrast is approximately normal, a
two-sided level-\(\alpha\) test has local noncentrality

\[
\sqrt{N\,\eta^2\kappa_\eta},
\]

up to the cluster-variance correction. A planning approximation for target
power \(1-\beta\) is

\[
N\approx
\frac{(z_{1-\alpha/2}+z_{1-\beta})^2}
     {\eta^2\kappa_\eta},
\]

with \(\kappa_\eta\) estimated under the planned DCE and inflated by the
respondent-cluster design effect.

## Proof sketch

The weighted projection \(P_\nu\) maps any signal onto the column space of
\(G_\nu\). Weighted projection geometry gives
\(G_\nu^\top W r_\eta=0\). Therefore
\(G_\nu^\top w_\star=0\). The variance normalization follows from
\(W=\Omega^{-1}\). For any \(w\) satisfying \(G_\nu^\top w=0\) and
\(w^\top\Omega w=1\), Cauchy–Schwarz in the \(W\)-inner product gives

\[
|w^\top g_\eta|
 = |w^\top r_\eta|
 \le \sqrt{r_\eta^\top W r_\eta},
\]

with equality at \(w=w_\star\). Substituting the local mean expansion gives the
first-order insensitivity result.

## Scope conditions

The proposition covers the declared nuisance library. It does not certify that
the library is complete. An omitted complexity or framing mechanism with a
component outside the span of \(G_\nu\) can produce a residual rejection. That
case is a diagnostic for library expansion, not evidence that the candidate
utility basis alone is false. The main experiment must report sensitivity to
adding nuisance columns and must include at least one deliberately unlisted
mechanism.

For a multinomial menu, \(\phi\) must remain fixed as a whole vector. A single
pairwise utility gap is insufficient because a third alternative can change the
choice probabilities through the rest of the menu. The covariance matrix must
then be the multinomial covariance of the full probability-contrast vector or a
respondent-cluster robust estimate. The binary zero-gap result is a special
anchor: under symmetric errors, a zero candidate gap fixes the choice
probability at one half independently of a positive common scale. It is not a
general replacement for the full-menu fibre condition.

Failure to reject is compatible with low power. The paper must report the
planned effect size, \(\kappa_\eta\), sample-size calculation, and a power curve;
it must not interpret a non-rejection as proof of sufficiency.
