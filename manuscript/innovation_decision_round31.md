# Innovation decision after the finite-support and policy-path audits

## Recommended primary contribution

The manuscript should be centred on a **support-complete fibre audit for a
coarsened utility representation in a discrete choice experiment**. The
candidate object is the complete menu summary
\(\phi(X)\), and the estimand is the finite-support conditional relation
\(Y\perp X\mid\phi(X)\). The design constructs non-singleton feasible fibres,
holds the candidate menu prediction fixed within each fibre, and reports a
pre-outcome rank and eigenvalue certificate for the response contrasts that
the declared support can expose.

This is one framework. The fibre creates the conditional comparison, the
support-complete contrast space supplies the certificate, and the design
optimizer repairs or reallocates support when the certificate is deficient.
Nuisance projection and policy-path sensitivity remain controls or extensions
inside this framework; they should not appear as separate headline methods.

## What the computations establish

The exact finite-support construction uses nine profiles \((a,b)\in\{0,1,2\}^2\)
and \(\phi(a,b)=a+b\). The two-point candidate-precision design puts half its
mass on \((0,0)\) and half on \((2,2)\). It has candidate-summary variance
4.00, support-complete rank 0, and minimum exposure eigenvalue 0. A
support-complete allocation gives rank 4, minimum eigenvalue 0.1222, and
positive mass on every profile. These are structural design quantities, not
estimated preference effects.

The finite benchmark was rerun with 30 replications, 400 observations, and
199 conditional randomization draws. Under the hidden direction that is
orthogonal to the two-feature hand-written dictionary, the dictionary score
rejected in 0.067 of runs and the support-complete score in 1.000. Under the
null the corresponding rates were 0.133 and 0.033. The result demonstrates
why a finite dictionary can miss an exposed direction. It does not establish
that the support-complete score is inferentially different from a saturated
fibre likelihood ratio.

The saturated-LR comparison is therefore an explicit boundary test. The two
procedures use the same finite fibres and conditional randomization law and
show the same rejection pattern. The publishable increment is the design
certificate: before outcomes are collected, the analyst can calculate the
available rank, identify a structural blind direction, and search for the
smallest feasible support repair. The paper must state this contribution
directly and avoid a claim of a new universal lack-of-fit test.

## Why this is not the policy-path contribution

The counterfactual policy-path audit is useful as a negative control. It shows
that model disagreement can be large when omitted curvature is visible and
zero when two models share an omitted off-support term. Its generic form is
close to the existing literature on counterfactual sensitivity, robustness,
and model equivalence. It should not be the title contribution. The common-mode
witness can remain as a limitation or a short robustness section because it
prevents an invalid interpretation of model agreement.

## Claim boundary for submission

The paper can claim a design-based finite-support audit of a coarsened utility
representation. It can claim that candidate precision and representation
refutability are distinct design objectives, and that support rank, null space,
and eigenvalues give an auditable pre-outcome certificate. It can claim that
the certificate separates structural visibility from sample-size power.

It should not claim global nonparametric sufficiency, a universal MNL
misspecification test, a new conditional-randomization principle, or human
preference evidence from the public Swissmetro file. The Swissmetro run is an
implementation check; the smart-device and finite-support runs are planning
simulations. A randomized candidate-preserving DCE would be needed to make a
human-data rejection claim.

## Submission recommendation

Use the title **Testing Utility-Basis Sufficiency in Discrete Choice
Experiments: A Support-Complete Fibre Design**. Keep one acronym at most, or
write the method name in full throughout. Put the exact rank theorem and the
candidate-precision counterexample before the simulation results. Report the
saturated-LR equivalence as a deliberate scope check. Treat the policy-path
oracle and the Swissmetro audit as robustness and external implementation
evidence.
