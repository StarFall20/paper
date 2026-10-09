# Round nine: parity-separated utility-fibre testing

## Why a new round was necessary

The nuisance-orthogonal fibre design controls a declared tangent library, but an
unlisted geometry--framing interaction can still look like an omitted utility
term. The strongest unresolved objection is the one raised by comparison
complexity: two tasks can have the same candidate utility difference and
produce different probabilities because the comparison is harder.

The next idea changes the transformation itself. A task pair is embedded in a
symmetric candidate-preserving fibre trajectory. The trajectory is indexed by
\(t\), with

\[
z_A(t)=c+t,\qquad z_B(t)=c-t,
\]

while \(s_A\) and \(s_B\) remain fixed. The candidate gap is constant in \(t\).
For a squared pair geometry, \(d(t)=d(-t)\) is even. For an omitted quadratic
decomposition term,

\[
z_A(t)^2-z_B(t)^2=4ct
\]

is odd. A central parity contrast can cancel symmetric geometry and scale
changes before estimating any nuisance parameters.

## The proposed object

Let \(m(t)\) be the vector of choice-probability contrasts for the \(+t\)
version and let \(m(-t)\) be the corresponding vector for the reflected
version. Define

\[
m_{\mathrm{odd}}(t)=\frac{m(t)-m(-t)}{2},\qquad
m_{\mathrm{even}}(t)=\frac{m(t)+m(-t)}{2}.
\]

The **parity-separated utility-fibre test** uses \(m_{\mathrm{odd}}\) as its
primary falsification contrast and reports \(m_{\mathrm{even}}\) as a
complexity/scale diagnostic. Under a candidate basis that is constant on the
fibre, symmetric errors, and nuisance mechanisms that depend only on the even
geometry index, the candidate implies \(m_{\mathrm{odd}}(t)=0\). A directional
omitted utility term with an odd fibre response violates that relation.

The construction is stronger than a post-hoc complexity control. The symmetry
is imposed before outcomes, and the even nuisance cancellation is an algebraic
property of the instrument. Remaining odd nuisance mechanisms are explicitly
included as failure conditions and can be handled by the nuisance-orthogonal
projection from round eight.

## Proposition

Suppose the pair probability vector admits a local decomposition

\[
p(t)=p_0 + u_{\mathrm{even}}(t)+\eta q_{\mathrm{odd}}(t)
       +r(t),
\]

where \(u_{\mathrm{even}}(t)=u_{\mathrm{even}}(-t)\),
\(q_{\mathrm{odd}}(t)=-q_{\mathrm{odd}}(-t)\), and the remainder is of smaller
order. Then

\[
\frac{p(t)-p(-t)}{2}
 =\eta q_{\mathrm{odd}}(t)+\frac{r(t)-r(-t)}{2}.
\]

The even nuisance component cancels exactly. A weighted collection of the
central contrasts gives a respondent-cluster randomization test. When an odd
nuisance tangent \(G_{\nu,o}\) is substantively plausible, use the round-eight
projection on the parity contrast rather than treating the parity result as
universally identified.

The test has no content when the candidate fibre is a singleton, when \(c=0\)
for the chosen quadratic direction, or when the relevant omitted response is
even. These are design-screening conditions, not hidden assumptions.

## Literature boundary

The search found symmetric paired choice designs that optimize information under
MNL and use balanced attribute levels. It did not find a discrete-choice
specification test that treats a candidate-preserving fibre as a group orbit and
uses its odd/even decomposition to cancel complexity. Existing choice-set
specification tests change the menu; model-discrimination designs maximize
level separation between rival models; the present construction tests a
transformation parity relation inside one candidate-equivalence class.

The method should be presented as a new experimental diagnostic, not as a new
general utility theorem. Its nearest risks are ordinary contrast coding,
symmetric DCE design, and semiparametric orthogonal scores. The novelty claim
rests on the combined estimand: an odd response along a candidate-preserving
utility fibre after algebraic cancellation of even processing mechanisms.

## Planning benchmark

`analysis/parity_fibre_benchmark.py` uses 200 replications, 600 respondents,
199 respondent-level sign flips, and five selected arms. The symmetric arms
satisfy `d_plus - d_minus = 0` and `h_plus + h_minus = 0` exactly.

| condition | parity fibre | one-sided comparator |
|---|---:|---:|
| null | .050 | .040 |
| omitted odd decomposition | 1.000 | 1.000 |
| even geometry-dependent scale | .050 | 1.000 |
| shared framing | .055 | .625 |
| odd scale contamination | .325 | 1.000 |
| omitted decomposition + even scale | 1.000 | 1.000 |

The benchmark demonstrates the intended separation: symmetric scale is removed
by parity, while an odd scale mechanism remains detectable as a failure boundary.
The shared framing result is not a universal guarantee because the comparator's
one-sided task changes the effective baseline; the final implementation must
construct a matched framed pair and report both even and odd components.
The result is planning evidence, not an empirical application.

## Decision relative to round eight

Round eight remains the general fallback for arbitrary task pairs. Round nine is
a stronger primary instrument when the substantive omitted direction and the
main complexity mechanism admit a reflection symmetry. The preferred paper
architecture is now:

1. use the parity fibre as the main instrument;
2. use nuisance-orthogonal weighting for any residual odd nuisance library;
3. report the even component as a processing/complexity diagnostic;
4. compare against the ordinary one-sided fibre probe, observational LR, and
   out-of-fold residual learning;
5. reject the claim if the empirical instrument cannot implement the reflection
   without changing wording, order, or perceived attribute meaning.

The next required test is a multinomial implementation with the full menu
vector held fixed, plus a paired DCE in which presentation order and wording are
independently counterbalanced.

## Multinomial check

`analysis/parity_multinomial_check.py` implements a three-alternative version.
All three \(s_j\) values are fixed across the reflected tasks, so the complete
menu vector is preserved exactly: \(\phi=(10,20,10)\) in both versions. In a
200-replication, 600-respondent run, multinomial parity rejects at .040 under
the null, .035 under an even scale change, 1.000 under the omitted quadratic
direction, .055 under the deliberately asymmetric scale condition, and 1.000
under the combined condition. This confirms that the binary algebra is not
being used as a substitute for the multinomial menu condition.

Reference links used in this round:

- [Construction of symmetric paired choice experiments](https://www.nature.com/articles/s41599-023-02153-4)
- [Chicu and Masten, *A specification test for discrete choice models*](https://doi.org/10.1016/j.econlet.2013.08.024)
- [Optimal experimental design for model discrimination](https://pmc.ncbi.nlm.nih.gov/articles/PMC2743521/)
- [Shubatt and Yang, *Tradeoffs and Comparison Complexity*](https://arxiv.org/abs/2401.17578)

A small design-sensitivity grid (`results/parity_design_sensitivity.csv`, 40
replications per cell) shows why the movement budget must be predeclared. With
small (c=t=0.5), omitted-direction rejection is only .125; with (c=1,t=1)
it is .800; with (c=1.5,t=1) and (c=2.5,t=1.5) it reaches 1.000. The even
scale rows remain at .075 in these short runs, close to the coarse sign-flip
reference. These values are planning diagnostics, not a claim of calibrated
finite-sample size; the final power curve must use substantially more
replications and a prespecified feasible range.

## Incremental comparison against ordinary diagnostics

`analysis/parity_incremental_benchmark.py` places the same 600-respondent budget
in a narrow observational support and in the parity supplement. Across 100
replications, the observational LR and out-of-fold residual proxy reject at
.060 and .030 under the omitted decomposition, while parity rejects at 1.000.
Under an even scale nuisance, LR and residual rejection are .030 and .060,
whereas parity remains at .050; the one-sided fibre comparator rejects at 1.000.
Under the combined omitted-plus-scale condition, parity rejects at 1.000 while
LR and residual rejection remain .050 and .060. This is the required incremental
mechanism: the gain comes from the support and reflected assignment, not from
switching to a more flexible learner. The run is still a planning simulation;
the final paper must repeat it with the locked DCE generator and larger
replication counts.

## Extension beyond the logit link

The formal result can be stated for the moderate-utility class, where binary
choice probabilities have the form (F(\Delta V/D)) for a smooth increasing
link (F) and a product-difference index (D). On the reflected fibre,
(D(t)=D(-t)) and an omitted utility component (q(t)) is odd. A first-order
Taylor expansion gives an odd probability response proportional to
(F'(\Delta V_0/D(t))q(t)/D(t)). This makes the parity instrument a directional
specification test within a broader choice-probability class, rather than a
logit-specific scale trick.

`analysis/parity_link_robustness.py` checks the same design under logistic and
probit links. In 100-replication planning runs with 600 respondents, logit
rejection is .020, 1.000, .020, and .750 across null, omitted, even-scale, and
combined conditions; probit rejection is .010, 1.000, .020, and .990. The
short run is a link-robustness check, not a calibrated size or final power
claim.
