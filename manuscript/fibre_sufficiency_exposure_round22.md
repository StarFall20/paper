# Fibre-conditional sufficiency as the primary contribution

## Decision after the overlap audit

The paper should present one framework: a **fibre-conditional sufficiency
audit** for discrete choice designs. The earlier names BMST and UFIT are
removed from the main story. Fibre quotient completeness is a proposition and
its rank calculation, not a second framework.

The central object is a candidate summary \(\phi(x)\) for a raw alternative or
menu \(x\). The null is

\[
        Y \perp X \mid \phi(X),
\]

where \(Y\) is the choice outcome. This is a conditional sufficiency claim.
It leaves the function \(P(Y\mid\phi(X))\) unrestricted. The paper therefore
does not test whether a particular linear index, coefficient vector, or
functional form is correct inside the summary space.

For a menu, \(\phi\) is the complete candidate menu object. In a three-option
task it contains enough coordinates to recover all independent candidate
probability contrasts. Matching one chosen-versus-rejected gap is not a
multinomial fibre.

## Relation to lack-of-fit designs

Classical lack-of-fit and T-optimal discrimination designs compare a specified
candidate response surface with a specified rival or a finite rival family.
They can use points at which the candidate predictions differ. The present
design creates a different comparison: candidate predictions are held fixed
inside each fibre, and raw profiles are randomized within that fibre. The test
then asks whether the outcome distribution changes with the discarded raw
coordinates. The alternative is a residual function on the fibre, potentially
an infinite-dimensional object.

This distinction is the contribution claim. Its target is conditional
invariance under a coarsening map: the candidate response surface is held
fixed inside each fibre, and the admissible support can be a constrained finite
DCE support. The claim remains bounded: the implementation tests a declared
residual dictionary or a local utility tangent.

In a standard T-optimal comparison, a design maximizes an integrated squared
difference between two specified response surfaces. The fibre criterion instead
maximizes the conditional residual information
\(E[\kappa_Z\operatorname{Var}(h(X)\mid Z)]\). A rival that changes only
\(q(\phi)\) can have large ordinary lack-of-fit distance while remaining inside
the sufficiency null. A raw-coordinate residual can have zero ordinary signal
on a singleton design and positive fibre exposure after support is added. The
two criteria answer different design questions.

## Fibre exposure theorem

Let \(z=\phi(x)\) index a fibre \(\mathcal F_z\). A reflected task samples
\(X^+,X^-\) from a symmetric within-fibre design measure \(\pi_z\). Let
\(p_0(z)\) be the common \(J\)-alternative choice-probability vector under the
null. For a scalar local residual direction \(h\), assume

\[
 p_\eta(x)=p_0(z)+\eta a_z h(x)+o(\eta),
 \qquad z=\phi(x),
\]

where \(a_z\) is a probability tangent with entries summing to zero. Let \(C\) be a full-row-rank
contrast matrix for the multinomial menu and let

\[
 b_z=C a_z, \qquad
 V_z=C\{\operatorname{diag}(p_0(z))-p_0(z)p_0(z)^\top\}C^\top.
\]

For a signed reflected contrast, the first-order mean is

\[
 \mathbb E[\Delta_z\mid X^+,X^-]
   =\frac{\eta}{2}b_z\{h(X^+)-h(X^-)\}+o(\eta).
\]

If the two members are independently drawn from the same fibre measure, the
local Fisher exposure for direction \(h\) is

\[
 \mathcal I_z(h)
 =\frac{\kappa_z}{2}\operatorname{Var}(h(X)\mid z),
 \qquad \kappa_z=b_z^\top V_z^{-1}b_z.
\]

The design-level exposure is \(\mathcal I(h)=\mathbb E_z[\mathcal I_z(h)]\).
Thus structural invisibility means

\[
 \mathcal I(h)=0
 \quad\Longleftrightarrow\quad
 h(X)\text{ is constant within every supported fibre with }\kappa_z>0.
\]

Raw attribute variance alone is not the criterion; only variation that reaches
the choice-probability tangent is exposed. For a vector residual dictionary,
apply the scalar result to every linear combination or use the block Jacobian
version.

The result is local in the departure amplitude \(\eta\), not in the attribute
levels. It therefore applies to finite discrete fibres. No derivative with
respect to a discrete attribute is required.

## Computable certificate

For a finite support, fibre \(g\) contains profiles \(x_{g1},\ldots,x_{gK_g}\)
with weights \(w_g\). Let \(R_g\) contain the declared residual features in
its rows and let

\[
 H_g=I_{K_g}-{\bf 1}w_g^\top
\]

be the weighted within-fibre centering matrix. The finite-support exposure
matrix is

\[
 E=\sum_g \alpha_g\kappa_g R_g^\top H_g^\top W_g H_gR_g,
 \qquad W_g=\operatorname{diag}(w_g),\quad
 \kappa_g=b_g^\top V_g^{-1}b_g,
\]

where \(\alpha_g\) is the design mass on fibre \(g\). The certificate is

\[
 \operatorname{rank}(E)=r
\]

for a declared \(r\)-direction residual dictionary. A zero eigenvalue marks a
direction that no increase in respondent count can reveal on the selected
support. The smallest positive eigenvalue supplies the structural conditioning
input to a sample-size calculation. The certificate is checked before data are
collected; statistical power is a separate calculation under a specified
effect amplitude and response noise.

If \(h(x)=q(\phi(x))\), then \(H_gR_g=0\). Such a direction is inside the
conditional-sufficiency null. It represents a functional-form or parameter
question within \(\phi\), not a violation caused by discarded raw
coordinates. The paper must keep this case separate from structural
invisibility.

## Design implication

A design optimized for precision of the reduced candidate basis can place all
mass on singleton fibres. It can estimate the candidate index precisely while
having \(E=0\) for every residual direction. A fibre-aware design must first
require non-singleton support and then maximize a criterion such as
\(\lambda_{\min}(E)\), \(\log\det(E+\epsilon I)\), or a preregistered
directional exposure. This is a support constraint on model criticism, not a
replacement for D-efficiency.

The simplest counterexample uses \(\phi(a,b)=a+b\). A two-point design on
\((0,0)\) and \((1,1)\) is informative about the candidate index and has
singleton fibres, so it cannot expose \(h(a,b)=a-b\). Adding the two middle
profiles \((0,1)\) and \((1,0)\) creates a nonzero fibre contrast while
leaving the candidate index value fixed at one. The same logic carries to
constrained smart-device profiles and multinomial menus.

The finite check sets the positive response metric \(\kappa_g\) to one to
isolate support geometry. It gives candidate-index precision \(1.00\) and fibre exposure
\(0.00\) for the endpoint design. A full factorial orthogonal design gives
\(0.50\) and \(0.50\), while shifting mass toward the middle fibre gives
\(0.25\) and \(0.75\). In the two-direction dictionary \((a-b,ab)\), the
endpoint exposure matrix has rank 0 and the two designs with middle-fibre
support have rank 1: the interaction \(ab\) is still structurally blind on the
available fibre. These numbers are design quantities, not estimated effects.
They make the structural separation visible without relying on a
linear-regression residual plot.

## Boundary with nearby literature

Sufficient-dimension-reduction and conditional-moment testing establish the
general statistical language of conditional invariance. Wilcox's binary
choice experiment tests dependence on previous choices, which is a process
dependence question rather than dependence on raw attributes after conditioning
on a candidate menu summary. Fok and Paap's JOCM tests target MNL/IIA
misspecification through pairwise composite likelihood and GMM moments. The
present object fixes the candidate menu summary by design and measures
within-fibre residual exposure before fitting a richer model.

Optimal model-discrimination designs and T-optimality provide the design
optimization precedent. The present criterion is a conditional residual
criterion: it optimizes variation after projecting out the candidate summary,
under feasible discrete DCE support. The paper should cite this relationship
directly and avoid claiming that optimal design or conditional-independence
testing is new in general.

The nearest conceptual collision is Healy and Leo's 2026 graph-theoretic
characterization of experiments that test deterministic preference rankings.
That work classifies orderings using menus. The present result concerns
stochastic probability contrasts, local choice tangents, and the exposure of
raw coordinates discarded by a candidate summary. It does not characterize
all experiments that test a preference model.

## Evidence required for a submission claim

The current structural and multinomial simulations establish the certificate,
the full-menu preservation condition, nuisance orthogonality, and the
separation between structure and power. A submission-ready version still needs
one of the following empirical validations:

1. a new paired DCE with the preregistered fibre assignment and a frozen
   candidate summary; or
2. a public DCE containing repeated or deliberately matched fibres with enough
   raw-profile support to estimate conditional contrasts.

The empirical comparison must report the fibre support table, the exposure
eigenvalues, sample-size planning, null controls, complexity and order placebos,
and a benchmark against a candidate-fit or LR lack-of-fit procedure. The
conclusion should say which residual directions were exposed and which were
structurally unavailable.
