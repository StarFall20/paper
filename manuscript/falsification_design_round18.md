# Structural-fibre maximin falsification design

## Contribution

The paper's primary contribution is a design rule for testing a coarsened
choice-model basis. It constructs a task block whose weakest detectable local
departure is as large as possible, subject to candidate preservation and
nuisance balance. The rule produces a coverage certificate before outcomes are
observed.

This changes the design objective from estimating a chosen utility model
efficiently to exposing departures that the chosen model cannot explain. The
two objectives select different tasks. A design can have a high determinant of
the candidate information matrix and still have zero support for a relevant
omitted direction.

## Coefficient-robust fibre

Let (b(x)) be the candidate sufficient-statistic vector for one alternative
and (V(x;\beta)=b(x)'\beta). A structural fibre pair satisfies

\[
b(x_{A+})=b(x_{A-}),\qquad b(x_{B+})=b(x_{B-}).
\]

Consequently, the candidate utilities and every candidate utility difference
are equal across the reflected versions for every (\beta). The construction
does not rely on a point estimate of the candidate coefficients. A pair that
only satisfies (V(x_{A+};\hat\beta)=V(x_{A-};\hat\beta)) has a weaker status:
coefficient estimation error can create a false parity signal.

The full menu condition is imposed alternative by alternative, including the
opt-out. Equality of one binary gap is not sufficient for a multinomial
claim.

## Falsification coverage criterion

Let (p) index an admissible reflected pair and let (h_\ell) be a frozen
local departure direction. The pair's odd signal is

\[
a_{p\ell}=\frac{1}{2}\{[h_\ell(A_+)-h_\ell(B_+)]-
[h_\ell(A_-)-h_\ell(B_-)]\}.
\]

At the candidate probability (p_p), define the standardized local signal

\[
\tilde a_{p\ell}=\sqrt{p_p(1-p_p)}a_{p\ell}/s_\ell,
\]

where (s_\ell) is a pre-outcome column scale. For a task block (S), the
coverage matrix is

\[
\mathcal I(S)=\tilde A_S'\tilde A_S,
\qquad
\kappa(S)=\lambda_{\min}(\mathcal I(S)).
\]

The structural-fibre maximin design solves

\[
S^*=\arg\max_{S\in\mathcal S}\kappa(S),
\]

where (\mathcal S) contains only structural fibres with balanced raw
comparison signatures and predeclared probability-scale bins. The value
\(\kappa(S^*)\) is the **falsification coverage certificate**.

The certificate is a design property. It does not turn the departure library
into a universal model class. A zero eigenvalue identifies an unsupported
direction and prevents the paper from claiming coverage of that direction.

## Smart-device proof of concept

The coarsened candidate basis is

\[
b(x)=(m+s,\;i+e,\;d,\;q),
\]

where (m) is use-management level, (s) support, (i) intelligence,
(e) evidence, (d) data management, and (q) price. The first two entries
represent care-intensity and clinical-quality composites. The task generator
enumerates raw profiles with the same (b(x)), then forms reflected pairs
whose raw comparison signature is equal across the two versions.

The resulting pool contains 1,024 admissible structural reflections. A
randomized exchange search over five probability-scale bins selects gaps
approximately 0.22, 0.36, 0.56, 0.66, and 0.82. The three declared
within-fibre directions have coverage eigenvalues 1.648, 6.063, and 7.352.
The selected block therefore has positive support for all three directions.

The direction library is deliberately limited to departures that can vary on
this structural fibre: a total quadratic term, a midpoint mode-support
interaction, and an intelligence-cloud interaction. The endpoint
mode-support interaction already used in the original simulation is constant
on the (m+s) fibre. The certificate correctly excludes it from the testable
library. This is a substantive result about support, not a missing power
calculation.

With 600 respondents, 499 respondent-level sign flips, 200 replications, and
a weak departure strength (\eta=0.30), the maximin block's lowest feature
power is 0.67 at zero coefficient perturbation and 0.65 when the candidate
coefficients are perturbed with standard deviation 0.05. A random structural
block has lowest feature power 0.40 and 0.45 under the same conditions. Null
rejection for the maximin block is 0.045, 0.035, 0.045, and 0.020 when the
coefficient perturbation standard deviation is 0, 0.02, 0.05, and 0.10.
These are simulation results, not a human-choice finding.

## Relation to the earlier BMST/UFIT work

The earlier candidate-preserving parity test remains the inference layer. The
new contribution determines whether the inference layer has support against a
declared local departure library. The task-ranking rule is no longer a
hand-picked information score for one known interaction. It is a maximin
coverage rule with a failure certificate.

The old coefficient-calibrated grid remains useful as a comparator. Its
coverage eigenvalue for a five-direction library is zero; its price-threshold
power is 0.045 at 600 respondents, while the maximin calibrated grid raises
that power to 0.94 in the matched 200-replication benchmark. This comparison
shows why candidate-preserving equality and coverage design are necessary
together.

## Boundaries and empirical gate

The method has four declared boundaries:

1. A full raw-attribute basis can have singleton structural fibres.
2. A departure that is a function of the candidate sufficient statistic is
   structurally invisible on that fibre.
3. An odd presentation, order, or attention effect can still appear as a
   relation violation; the nuisance screen must be measured and reported.
4. A rejection locates failure of the candidate relation; it does not identify
   a unique omitted mechanism.

The empirical submission gate is a preregistered paired DCE that implements
the structural fibre, one-member-per-pair assignment, raw signature balance,
the departure library, and the sign-flip reference. The human study must also
compare the maximin block with a conventional efficient block and with
added-term LR/residual diagnostics under the same respondent budget.

## Positioning against adjacent methods

Classical optimal-design work optimizes parameter precision or discrimination
among specified rival models. Fok and Paap's JOCM test uses pairwise moments to
test MNL misspecification. Breitmoser's invariance results establish observable
foundations for conditional logit. The present method uses these as boundaries
and develops a different object: a coefficient-robust, candidate-preserving
task block with a pre-outcome certificate for the weakest supported local
departure. It should be presented as a bounded DCE design method, not as a
new general invariance theorem or a universal specification test.
