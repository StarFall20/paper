# Nuisance-orthogonal utility-fibre testing

## Research object

The candidate basis observes a composite burden \(s=a_1+a_2\). A feasible task
pair changes the raw decomposition coordinate \(z=a_1-a_2\) while preserving
the candidate gap \(s_A-s_B\). Let \(\Delta(d)\) be the vector of observed
choice-probability contrasts across feasible pair arms \(d\).

The target is an omitted decomposition direction \(\eta\). The nuisance library
contains a shared framing/order shift and a geometry-dependent error-scale
shift. The method chooses a contrast that is locally insensitive to the nuisance
library before testing the candidate-preserving relation.

For arm \(d\), let \(g_\eta(d)\) be the local derivative of the pair contrast
with respect to \(\eta\), and let \(G_\nu(d)\) collect the local derivatives
with respect to nuisance parameters. With \(W\) equal to the inverse local
Bernoulli variance,

\[
P_\nu=G_\nu(G_\nu^\top W G_\nu)^{-1}G_\nu^\top W,
\quad
g_\eta^\perp=(I-P_\nu)g_\eta.
\]

The selected contrast satisfies \(G_\nu^\top w=0\) and is proportional to
\(Wg_\eta^\perp\). A feasible task subset is selected by maximizing the
standardized residual signal subject to attribute-range, dominance, realism,
and respondent-burden constraints.

The local result is recorded as a proposition in
`manuscript/nuisance_orthogonal_proposition.md`. If
\(\kappa_\eta=g_\eta^\top W(I-P_\nu)g_\eta\), the approximate sample size for
two-sided level \(\alpha\) and target power \(1-\beta\) is

\[
N\approx\frac{(z_{1-\alpha/2}+z_{1-\beta})^2}
              {\eta^2\kappa_\eta},
\]

after applying the respondent-cluster design effect. A non-rejection is
interpreted through this power calculation; it is not evidence that the
candidate basis is sufficient.

This criterion differs from ordinary D-optimality, which targets parameter
precision, and from raw model-discrimination design, which maximizes separation
between specified rival models. It maximizes the part of a candidate
specification violation that remains after local nuisance projection.

## Instrument

The candidate-preserving condition is checked using the full menu vector

\[
\phi(x)=\{b_j(x)-b_k(x):j<k\}.
\]

The fibre must be nontrivial. Each respondent receives one member of every
focal pair. The candidate is estimated on a development fold and frozen on the
test fold. A respondent-level orientation coin supplies a cluster sign-flip
reference. The result is interpreted as a relation violation on the tested
fibre; it does not identify a unique omitted term.

For a multinomial menu, every component of \(\phi\) must be preserved. Keeping
one pairwise gap fixed leaves the remaining alternatives free to change the
choice probabilities. The multinomial implementation therefore uses the
covariance of the full probability-contrast vector and respondent-cluster
inference. The binary zero-gap result is a special anchor: symmetric errors
give probability one half at zero candidate gap for every positive common
scale. It does not replace the full-menu condition.

The common-translation arm remains useful because it can make a declared
pairwise geometry tangent zero. The nuisance-orthogonal construction combines
these controls into one pre-outcome contrast instead of treating them as
unrelated post-hoc checks.

## Planning benchmark

The proof-of-concept uses 200 replications, 600 respondents, 199 sign flips,
and five selected focal arms. The raw geometry index is squared Euclidean
distance. This choice is necessary: the L1 distance is constant for the
fixed-sum fibre and produces a zero scale tangent. The omitted decomposition
strength is set to a moderate \(\eta=0.10\). The naive comparator chooses arms
with the largest raw omitted signal and does not project nuisance tangents.

| condition | nuisance-orthogonal | naive raw-signal |
|---|---:|---:|
| null | .035 | .025 |
| omitted decomposition | .750 | 1.000 |
| geometry-dependent scale | .020 | .025 |
| shared framing | .035 | .020 |
| unlisted nonlinear scale | .035 | .040 |
| symmetric unlisted complexity | .035 | .100 |
| out-of-library complexity interaction | .035 | .140 |
| out-of-library geometry framing | .980 | 1.000 |
| combined omitted and scale | .385 | .850 |

The corrected benchmark shows the intended operating characteristic. Declared
scale and framing mechanisms remain near the 5% size, while the omitted
decomposition is detected at .750 and the combined signal at .385. The
out-of-library geometry-framing mechanism is rejected at .980. This is an
expected limitation: orthogonality protects against the declared nuisance
span, not an arbitrary processing rule. The remedy is to add a substantively
motivated nuisance column and report the sensitivity. The benchmark is
planning evidence, not an empirical application.

## Required stress tests

The main experiment must vary the nuisance library, not only the data
generating condition. Add an unlisted nonlinear scale function, asymmetric
framing, order carryover, random taste, and a geometry index not used in the
projection. Report the rank and condition number of \(G_\nu\), the residual
signal norm, calibration, power, and repair after adding the omitted term. The
empirical DCE must also compare the fibre statistic with an observational LR
test and an out-of-fold residual learner on the same simulated support. That
comparison supplies the incremental evidence that the designed fibre reaches
violations those diagnostics miss.

The contribution should be withdrawn if the nuisance projection removes all
useful signal, if the design criterion collapses to D-efficiency, or if an
ordinary LR test on the original support has the same calibrated power.
