# Nuisance-balanced utility-fibre audit for discrete choice models

> **Evidence boundary after Round 27.** The finite support-complete score is
> inferentially equivalent to adding unrestricted within-fibre profile terms;
> it should be presented as a design/exposure certificate unless a stronger
> design objective is added. The public Swissmetro results are observational
> diagnostics, not randomized validation.

## Research problem

Choice models are usually evaluated on the variation already present in the
data. A likelihood ratio test, a flexible residual learner, or a richer utility
specification can reveal structure when the observed attributes provide enough
leverage. A candidate model can still appear adequate when the data occupy a
narrow part of the attribute space. This creates a design problem: how can a
researcher test the sufficiency of a utility basis when ordinary observations
contain little variation along the omitted direction?

We address this problem with a candidate-preserving choice design. The design
constructs pairs of attribute profiles that the candidate model treats as
identical at the menu level. It then changes the attribute decomposition while
keeping the candidate prediction fixed. A systematic change in choice
responses across the pair violates the candidate's sufficiency relation.
The construction also matches a predeclared nuisance signature. The signature
includes the value-weighted \(L_1\) comparison distance, the number of changed
attributes, the raw level-change load, and the signs of component changes.
Equal candidate gaps alone do not imply equal choice probabilities under
comparison-complexity models.

## Reflected utility fibres

Let (\phi(T)) denote the complete vector of candidate menu utilities for task
(T), including the opt-out. A utility fibre is

\[
\mathcal F_{\phi} = \{T: \phi(T)=\phi\}.
\]

A reflected pair (T_{+},T_{-}\in\mathcal F_{\phi}) changes the attribute
composition inside the same candidate fibre. Let \(g(T)\) denote the declared
nuisance signature. We require

\[
g(T_+)=g(T_-),
\]

with tolerances fixed in an independent calibration stage. The signature can
include the pairwise geometry used by the declared processing model:

\[
D(T_{+})=D(T_{-}).
\]

For a binary A--B contrast, write (m_{+}) and (m_{-}) for the oriented
choice response. The primary statistic is the odd component

\[
m_{\mathrm{odd}}=\frac{m_{+}-m_{-}}{2},
\]

and the even component

\[
m_{\mathrm{even}}=\frac{m_{+}+m_{-}}{2}
\]

is reported as a processing diagnostic. Under the candidate model, the matched
nuisance mechanisms are equal across the reflected pair and cancel from
\(m_{\mathrm{odd}}\). An omitted utility direction with an odd response
remains in the primary contrast. An unmatched process effect is treated as a
failed nuisance-balance condition and blocks a utility-specific interpretation.

The result extends beyond logit. In the moderate-utility form

\[
P(A\succ B\mid T)=F\!\left(\frac{\Delta V(T)}{D(T)}\right),
\]

with a smooth increasing (F), an even (D), and an odd omitted component
(q(T)), the local parity response is

\[
\frac{P_{+}-P_{-}}{2}
=\eta\frac{F'(\Delta V_0/D)}{D}q(T)+o(\eta).
\]

The test targets a directional failure of the candidate basis. It does not
identify a unique omitted term from rejection alone.

## Task design criterion

The design searches the finite attribute space before outcomes are observed. A
candidate task receives an odd-information score

\[
I_{\mathrm{odd}}(T)=p_T(1-p_T)s_T^2,
\]

where (p_T) is the candidate A--B choice probability and (s_T) is the
magnitude of the predeclared odd omitted-gap signal. The primary task block
also requires exact antisymmetry of the omitted gap, full-menu preservation,
and equality of the nuisance signature. In the smart-device attribute space,
enumeration finds 677 candidate-preserving reflections, 648 pure-odd
reflections, and 58 reflections that also match the full declared nuisance
signature. Five predeclared tasks cover candidate gaps near 0.32, 0.38, 0.57,
0.80, and 0.98.

A matched five-task simulation shows the design value under a weak omitted
signal. With 150 respondents and a respondent-level orientation coin, the
ordinary candidate grid rejects at .490 and the fibre-optimal grid at .870.
With an even-scale nuisance added, the rates are .335 and .785. Null rejection
is .015 and .055, and even-scale rejection is .040 and .045. Strong omitted
signals produce power of 1.000 for both designs, so the design claim concerns
weak signals. A separate binary moderate-utility stress test shows why the
nuisance signature is required: a loose grid has .185 rejection under a
complexity-only condition at 600 respondents, while the balanced grid has .050;
with the omitted direction added, both reach 1.000.

## Support-complete finite certificate

The task criterion does not depend on a hand-picked residual dictionary when
the feasible support is finite. For a fibre with (K_g) profiles, let (Q_g)
span all (K_g-1) contrasts orthogonal to the constant profile vector. The
support-complete exposure block is

\[
 E_g^{\mathrm{sat}}=\alpha_g\kappa_g Q_g^\top
 \{\operatorname{diag}(w_g)-w_gw_g^\top\}Q_g.
\]

The direct sum has dimension (sum_g(K_g-1)), so its rank and smallest
eigenvalue certify whether every non-constant response pattern on the declared
finite support is visible. This removes a reviewer-dependent choice of omitted
polynomial terms. It remains a finite-support certificate and does not claim
coverage of unlisted continuous functions. On the (3\times3) counterexample,
the candidate-precision endpoint design has rank 0, while the support-complete
design has rank 4 and minimum eigenvalue 0.1222. The reproducible calculation
is in `analysis/support_complete_fibre_design.py`.

## Fibre violation profile

The five-task block also produces a vector of task-specific odd responses. The
global average tests whether the candidate relation fails somewhere in the
block. A max-(T) sign-flip statistic reports where the failure is concentrated
while controlling the family-wise error rate. In planning simulations with 600
respondents, 200 replications, and 999 sign flips, a localized omitted
interaction gives global power .330 and profile power .885; the profile selects
the active task in .990 of replications. Under the null, global and profile
rejection are .035 and .050. A diffuse omitted interaction produces power 1.000
for both statistics. The profile is a diagnostic layer of the same
nuisance-balanced fibre audit. It does not identify a unique omitted term.

## Assignment and heterogeneity boundary

The primary randomization unit is the respondent block. One orientation coin is
drawn once for each respondent and shared across the five tasks; the reference
distribution flips respondent-level cluster signs. This assignment rule is used
in the profile, complexity, and task-comparison benchmarks. A separate negative
control draws respondent coefficients around the calibrated population vector.
With 600 respondents and 200 replications, rejection is .085, .095, and .085
when coefficient standard deviations are 0, .1, and .2, compared with 1.000 for
the declared omitted direction. The modest finite-sample elevation under random
taste is a boundary result: the fibre is built from the calibrated candidate,
so rejection under unmodelled individual heterogeneity is evidence against the
candidate relation and cannot be assigned to a particular omitted interaction.

## Smart-device implementation

The supplied smart-device attributes provide an application-specific audit. The
selected pair uses use-management mode, intelligent functionality, professional
support, clinical evidence, data management, and price. The additive candidate
assigns utilities (0.23) and (-0.17) to A and B in both reflected tasks, and
assigns (-0.42) to the opt-out. The candidate menu vector is fixed at
((0.23,-0.17,-0.42)). The squared A--B level-index distance is 6 in both
versions.

The data-generating intelligence-by-cloud interaction changes the A--B gap from
(-0.55) to (+0.55). A 200-replication A/B/opt-out simulation with 400
respondents gives .025 null rejection and .995 rejection when this interaction
is active. A calibration-error audit shows why an independent pilot is needed:
coefficient noise with standard deviation .02 gives planning rejection near
.030, while noise with standard deviation .05 gives .150.

## Contributions and boundaries

This study makes four contributions. First, it introduces a candidate-preserving
DCE specification audit that probes omitted utility directions inside a
nontrivial attribute fibre and supplies a support-complete finite rank
certificate. Second, it adds nuisance-signature balance so that comparison
complexity is tested as a negative control instead of being folded into the
utility conclusion. Third, it supplies a pre-outcome information criterion for
selecting tasks that improve weak-signal sensitivity after the balance
constraints are imposed. Fourth, it provides an application-specific protocol
with calibration, semantic, ordering, and multiplicity controls.

The design requires a nontrivial fibre. A saturated candidate basis has no
within-fibre variation. An odd order, wording, attention, or semantic effect
survives the primary contrast and requires a nuisance audit. An even omitted
response can be invisible to the primary statistic. Non-rejection supports the
candidate relation on the tested fibre and does not establish global utility
sufficiency.
