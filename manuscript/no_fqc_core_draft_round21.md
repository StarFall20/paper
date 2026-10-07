# Nuisance-orthogonal fibre quotient completeness for candidate-basis audits in discrete choice models

## Abstract

Choice-model diagnostics can miss a departure when the observed tasks contain
little variation in that direction. A reflected task can expose the problem by
holding the candidate model's full menu prediction fixed while changing the
raw attribute decomposition. Such a contrast still has an identification
problem: framing, error scale, and task-processing effects can move the
response at the same time. This paper introduces nuisance-orthogonal fibre
quotient completeness (NO-FQC). Candidate-preserving fibres generate an odd
contrast operator for a declared utility-departure space. A weighted projection
removes the tangent space of declared nuisance mechanisms before the operator's
rank and information eigenvalues are computed. The residual rank counts
locally testable utility directions; its null space gives a pre-outcome blind
direction certificate; its smallest eigenvalue measures conditioning after
nuisance removal. A smart-device structural pool contains 136,179 reflected
tasks when geometry is allowed to vary. A five-task block has residual
information eigenvalues 5.745, 6.909, and 11.492. In a 200-replication
benchmark with 600 respondents, the projected statistic keeps scale and
framing rejection near the 5% level while an unprojected signal shows
substantial over-rejection under framing. The method is a local design
certificate. It requires a nontrivial candidate fibre, a frozen nuisance
library, and a paired-task implementation.

## 1. Motivation

Choice-modelling studies often choose a utility specification by likelihood,
information criteria, holdout prediction, or comparison with a richer model.
Those diagnostics inherit the support of the observed tasks. A candidate basis
can fit that support while treating raw profiles as equivalent in a way that
respondents do not.

The problem becomes harder when the same reflected task can change an error
scale, a framing response, or a comparison process. A rejection then has no
clear interpretation. The design must create candidate utility cancellation
and control the nuisance directions before the test is interpreted.

NO-FQC answers one question: after a candidate basis is held fixed and a
declared nuisance tangent space is removed, which utility departures remain
locally testable on the available attribute support?

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

## 3. Nuisance-orthogonal fibre quotient completeness

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

The NO-FQC rank is

$$
\operatorname{NOFQC}(S;H,N)=\operatorname{rank}(D_S^\perp).
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
choice probability, and weighted projection geometry. For a multinomial menu,
the covariance metric must use the full menu probability-contrast vector. A
single pairwise gap is insufficient because the remaining alternatives can
change choice probabilities.

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

## 6. Positioning

Recent JOCM work studies assisted specification, model-choice effects,
sequential and Bayesian DCE design, scale and taste heterogeneity, varying
choice-set size, and respondent engagement. Those studies motivate the
problem. Model-robust maximin DCE design already exists, so maximin is used
only after residual rank completeness.

Fok and Paap provide outcome-side misspecification tests. Semiparametric
inference provides nuisance-orthogonal scores. NO-FQC contributes a
choice-specific bridge: a candidate-preserving observed fibre supplies the
coefficient-robust cancellation, and the nuisance projection supplies a
pre-outcome utility-versus-process boundary. The paper does not claim a
general invariance theory or a universal MNL test. Healy and Leo's
permutohedron framework characterizes experiments that test deterministic
preference rankings. NO-FQC addresses stochastic DCE contrasts for a
coarsened parametric utility basis and reports residual local rank and
information, so the paper should not use the broader phrase “which
experiments test a model.”

## 7. Scope and submission gate

NO-FQC requires a nontrivial candidate fibre. A saturated raw basis has
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
