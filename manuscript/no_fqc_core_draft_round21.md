# Fibre-conditional sufficiency audits for discrete choice models

## Abstract

Choice-model diagnostics usually ask whether a fitted parametric model predicts
the observed tasks. This paper asks a different design question: does a
candidate menu summary retain all choice-relevant information in the raw
profiles? The null is the conditional sufficiency relation
\(Y\perp X\mid\phi(X)\), with the response function given the summary left
unrestricted. We construct finite, feasible DCE fibres that keep the complete
candidate menu vector fixed and randomize raw profiles inside each fibre. A
fibre exposure matrix gives a computable certificate for which declared
residual directions can be seen and which are structurally invisible. Its local
information is a choice-probability-weighted conditional variance, so a
residual that is a function of \(\phi\) correctly belongs to the null. A
nuisance-orthogonal extension removes declared framing and scale tangents before
the exposure rank is calculated. Structural and multinomial simulations verify
the full-menu condition, rank certificate, and separation of exposure from
sample-size power. The method is an audit of a coarsened candidate basis; it is
not a universal lack-of-fit test.

## 1. Motivation

Choice-modelling studies often compare likelihoods, information criteria,
holdout prediction, or richer specifications. These procedures can leave a
coarsened candidate basis unexamined when the design contains little or no
within-fibre variation. A design that estimates the candidate index precisely
can still give zero information about whether discarded raw attributes matter.

The paper makes the distinction explicit. Classical lack-of-fit and
T-optimal discrimination compare specified response surfaces. Here the
candidate response is held fixed inside a fibre and the alternative is a
residual dependence on raw profiles. The null allows any function of the
candidate summary, so the test targets information lost by the coarsening map.
This is the source of the additional design problem and the reason a fibre
support certificate is needed.

Framing, error scale, and task-processing effects can also change a reflected
response. The primary exposure certificate reports the structural signal
before outcomes are observed. The NO-FQC projection in Section 3 removes a
declared nuisance tangent space when a first-order choice-probability model is
used. Power calculations remain separate from both certificates.

## 2. Candidate-preserving fibres

Let $x$ be a raw alternative profile and let $b(x)$ be the candidate
sufficient-statistic vector. The candidate utility is

$$
V_0(x;\beta)=b(x)^\top\beta.
$$

For a reflected binary task
$t=(A_+,A_-,B_+,B_-)$, impose

$$
b(A_+)=b(A_-),\qquad b(B_+)=b(B_-).
$$

The complete candidate menu vector is then unchanged by the reflection. This
condition holds for every candidate coefficient vector. A respondent-level
orientation coin assigns one member of each reflected pair.

For a local departure $h$, define the odd utility contrast

$$
D_t(h)=\frac12\{[h(A_+)-h(B_+)]-[h(A_-)-h(B_-)]\}.
$$

For a declared departure space
$H=\operatorname{span}\{h_1,\ldots,h_r\}$, a task block $S$ gives the matrix

$$
D_S=[D_t(h_\ell)]_{t\in S,\ell=1,\ldots,r}.
$$

The candidate utility cancels from the odd contrast. This cancellation is a
design relation; it does not depend on an estimated coefficient vector.

### Sufficiency null and fibre exposure

The target null is

$$
Y \perp X\mid \phi(X).
$$

It leaves $P(Y\mid\phi(X))$ unrestricted. For a local residual direction
$h$, write the choice-probability expansion as

$$
p_\eta(x)=p_0(\phi(x))+\eta a_{\phi(x)}h(x)+o(\eta),
$$

where $a_z$ is a probability tangent. For a full-rank multinomial contrast
matrix $C$, set $b_z=Ca_z$ and let $V_z$ be the covariance of the independent
probability contrasts under $p_0(z)$. A symmetric pair sampled within a fibre
has local information

$$
\mathcal I_z(h)=\frac{\kappa_z}{2}\operatorname{Var}(h(X)\mid z),
\qquad \kappa_z=b_z'V_z^{-1}b_z.
$$

The design exposure is $\mathcal I(h)=E_z[\mathcal I_z(h)]$. A direction is
structurally invisible when $h(X)$ is constant within every supported fibre
with $\kappa_z>0$. Raw attribute variance is relevant only when it reaches the
choice-probability tangent. A direction $h=q(\phi)$ has zero exposure and
remains inside the sufficiency null; it is a functional-form question within
the summary space.

For finite discrete support, no derivative with respect to an attribute level
is needed. If $R_g$ contains a residual dictionary on fibre $g$ and
$H_g=I-\mathbf1w_g'$, the certificate is the finite matrix

$$
E=\sum_g\alpha_g\kappa_gR_g'H_g'W_gH_gR_g,
\qquad W_g=\operatorname{diag}(w_g),\quad
\kappa_g=b_g'V_g^{-1}b_g.
$$

Its rank and eigenvalues are calculated before sampling. Rank deficiency is a
structural blind direction. A positive eigenvalue says that a direction is
exposed; it does not establish finite-sample power.

## 3. Nuisance-orthogonal extension

Let $G_S$ collect the local response tangents for the nuisance mechanisms
declared before outcomes are observed. The columns can include signed
framing/order shifts, geometry-dependent scale, and pre-specified task
processing terms. Let $W$ be the inverse covariance metric of the observed
task contrasts. Define the weighted nuisance projection

$$
P_{N,S}=G_S(G_S^\top W G_S)^{-1}G_S^\top W
$$

and the residual departure operator

$$
D_S^\perp=(I-P_{N,S})D_S.
$$

The nuisance-orthogonal fibre quotient rank is

$$
\operatorname{rank}(D_S^\perp).
$$

Let $\mathcal X$ contain every admissible reflected task in the declared
attribute support. The available-support benchmark is

$$
r_{\max}^\perp=\operatorname{rank}(D_\mathcal X^\perp).
$$

A block is complete for $(H,N)$ when its residual rank reaches
$r_{\max}^\perp$. A direction in $\ker(D_S^\perp)$ is indistinguishable from
the declared nuisance span on that block. A direction in
$\ker(D_\mathcal X^\perp)$ is unavailable on the attribute support even before
sampling.

The residual information matrix is

$$
K_S=D_S^\top W(I-P_{N,S})D_S.
$$

Its smallest eigenvalue measures the weakest locally testable direction after
nuisance removal. For one target signal $g$, the maximum-signal unit-variance
contrast is

$$
w_\star=\frac{W(I-P_{N,S})g}
{\sqrt{g^\top W(I-P_{N,S})g}}.
$$

It satisfies $G_S^\top w_\star=0$. A task block is chosen in two stages:
first reach the full residual rank, then maximize the smallest residual
eigenvalue subject to gap, attribute-range, realism, and respondent-burden
constraints.

### Proposition

Assume a linear candidate utility, alternative-by-alternative fibre
preservation, a common random-utility error law, and respondent-level random
orientation. Under the candidate model, the expected odd response is zero.
For a local model

$$
V(x)=b(x)^\top\beta+\eta h(x),
$$

the mean contrast vector has the expansion

$$
E[\widehat\Delta]=D_S\eta+G_S\nu+
o(|\eta|+\|\nu\|),
$$

where $\nu$ is the declared nuisance parameter. Premultiplication by
$I-P_{N,S}$ removes the nuisance term to first order. The locally
distinguishable utility space and its blind space therefore have dimensions
$\operatorname{rank}(D_S^\perp)$ and
$r-\operatorname{rank}(D_S^\perp)$.

The proof follows from fibre cancellation, the first derivative of a smooth
choice probability, and weighted projection geometry. This projection is an
extension for declared nuisance mechanisms; the primary sufficiency claim is
the unprojected fibre exposure relation above. For a multinomial menu, the
covariance metric must use the full menu probability-contrast vector. A single
pairwise gap is insufficient because the remaining alternatives can change
choice probabilities.

## 4. Smart-device structural design

The testbed uses six attributes: use-management mode, intelligence,
professional support, clinical evidence, data management, and price. The
coarsened candidate basis is

$$
b(x)=(m+s,\ i+e,\ d,\ q).
$$

The generator groups raw profiles by this vector and forms reflected pairs.
The primary NO-FQC pool matches the number and sign of changed attributes while
allowing squared raw distance to vary. This creates a declared geometry-scale
tangent instead of eliminating it by construction. The pool contains 136,179
reflections with candidate gaps spanning five predeclared bins.

The departure library contains total quadratic curvature, a midpoint
mode-support interaction, and an intelligence-cloud interaction. The selected
five-task block uses pool indices 65101, 48808, 55951, 55734, and 10202. Its
residual information eigenvalues are 5.745, 6.909, and 11.492. A raw-signal
block selected under the same gap and nuisance-support constraints has
eigenvalues 0.172, 0.344, and 1.274. The difference is a conditioning result
after projection; it is not a claim that NO-FQC always has greater raw power.

The endpoint mode-support interaction is constant on the $(m+s)$ fibre. The
support audit therefore reports it as blind and returns a minimal repair:
adding level 3 to the mode/support domain produces a fibre with spread 1.
This is a direct support consequence of the same operator.

The support issue is separate from candidate precision. With
$\phi(a,b)=a+b$, a design concentrated on $(0,0)$ and $(1,1)$ estimates the
candidate index from widely separated values while every supported fibre is a
singleton. It has zero exposure for $h(a,b)=a-b$. Adding $(0,1)$ and $(1,0)$
creates a nonzero within-fibre contrast at $\phi=1$. This is the finite-support
counterexample used to explain why a D-efficient candidate design need not be
an informative model-audit design.

## 5. Simulation evidence

The structural benchmark uses 200 replications, 600 respondents, and 499
respondent-level sign flips. Rejection rates for the projected and raw
contrasts are:

| condition | NO-FQC | raw signal |
|---|---:|---:|
| null | .070 | .045 |
| omitted decomposition | .375 | .755 |
| declared scale | .035 | .085 |
| combined | .060 | .075 |
| signed framing | .045 | .215 |
| unlisted complexity | .055 | .095 |
| geometry framing outside the library | .995 | 1.000 |

The projected statistic controls declared nuisance directions at the cost of
power when the target departure lies close to those directions. The
out-of-library geometry-framing condition shows why the nuisance library must
be frozen and expanded through predeclared sensitivity analyses.

The departure-strength curve makes this trade-off explicit. For the omitted
decomposition, NO-FQC power is .375 at strength $\eta=0.10$ and .980 at
$\eta=0.30$, while the raw-signal comparator is .755 and 1.000. The projected
test remains close to size for the declared scale and framing conditions at
both strengths because those conditions do not contain the target utility
direction.

These are planning simulations. They do not establish a human preference
effect. A paired DCE must report usable-pair counts, semantic screening,
response time, task complexity, sample-size calculations, and the action for a
mixed or unresolved violation.

The multinomial implementation uses eight three-alternative arms and two
independent contrasts per arm. Every arm preserves all three pairwise
candidate gaps exactly (maximum deviation 0). The residual rank is 3 and the
smallest residual eigenvalue is 1.203; the constructed score satisfies
$\max|G'w_\star|=3.0\times10^{-18}$. Across 200 replications with 600
respondents and 499 sign flips, the NO-FQC rejection rates are .025 under the
null, .030 under declared scale, .040 under declared framing, and .045 under a
strong framing perturbation. The raw score rejects at .080, .545, .145, and
.835 in those same conditions. The omitted residual rejects at 1.000 for both
scores, while the deliberately unlisted process rejects at .470 after the
projection. These results verify the full-menu and nuisance-control gates;
they are not evidence of human choice behavior.

## 6. Positioning

Recent JOCM work studies assisted specification, model-choice effects,
sequential and Bayesian DCE design, scale and taste heterogeneity, varying
choice-set size, and respondent engagement. Those studies motivate the
problem. Model-robust maximin DCE design already exists, so maximin is used
only after residual rank completeness.

Fok and Paap provide outcome-side misspecification tests. Semiparametric
inference provides nuisance-orthogonal scores. The fibre-sufficiency audit
contributes a choice-specific bridge: a candidate-preserving observed fibre
supplies coefficient-robust cancellation, and the exposure matrix measures
which discarded raw coordinates can reach the choice probabilities. The
nuisance projection is a controlled extension for process tangents. The paper
does not claim a general invariance theory or a universal MNL test. Healy and Leo's
permutohedron framework characterizes experiments that test deterministic
preference rankings. NO-FQC addresses stochastic DCE contrasts for a
coarsened parametric utility basis and reports residual local rank and
information, so the paper should not use the broader phrase “which
experiments test a model.”

## 7. Scope and submission gate

The fibre-sufficiency audit requires a nontrivial candidate fibre. A saturated raw basis has
singleton fibres and gives no test content. A direction that is a function of
the candidate sufficient statistic is structurally invisible. A rejection
identifies a failure of the candidate relation on the tested support; it does
not identify one unique psychological mechanism.

The submission version requires a preregistered paired DCE or an external
dataset carrying the same candidate-preserving assignment. The preregistration
must freeze the candidate basis, departure library, nuisance tangents,
orientation assignment, gap bins, and support-repair rule. The analysis must
compare NO-FQC with a raw fibre statistic, an added-term LR test, an
out-of-fold residual learner, and a conventional efficient design. Until that
evidence is available, the paper is a simulation-validated design method.
