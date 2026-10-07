# Independent innovation search, round 23

## Search objective

The search asks whether the current fibre-conditional sufficiency audit should
be replaced by a stronger contribution, or whether another method, variable,
or reverse-design operation can deepen it without producing a stitched theory.
The screening rule is strict: a candidate must have one primary estimand, one
identification mechanism, a design that can be fixed before outcomes are seen,
and a failure result that would make the idea less credible.

The recent Journal of Choice Modelling agenda makes this rule necessary. Recent
papers study modelling workflow, model-choice sensitivity, choice-set size,
decision-rule heterogeneity, level overlap and attribute attendance, and
assisted utility search. They show that process, design, and specification are
active concerns, while a new algorithm-versus-MNL comparison would enter a
crowded stream. The relevant boundaries are [Nova, van Cranenburgh and Hess
(2025)](https://doi.org/10.1016/j.jocm.2025.100562), [Quainoo et al.
(2024)](https://doi.org/10.1016/j.jocm.2024.100524), [Akinc, Street and
Vandebroek (2024)](https://doi.org/10.1016/j.jocm.2024.100493), [Kazagli and de
Lapparent (2023)](https://doi.org/10.1016/j.jocm.2023.100413), [Jonker
(2024)](https://doi.org/10.1016/j.jocm.2024.100494), and [Ortelli et al.
(2021)](https://doi.org/10.1016/j.jocm.2021.100285).

## Idea DNA used for screening

The parent problem is candidate misspecification under sparse or constrained
DCE support. The mutation is to move the object of design from parameter
precision to the visibility of candidate violations. The locked mechanism is
conditional sufficiency on a complete menu summary. The reverse operation is
to search for designs that are excellent for the candidate and deliberately
blind to a raw-coordinate departure, then repair that blind spot. The method
transfer comes from metamorphic testing and efficient-score projection; neither
is imported as a ready-made choice model.

The following locks prevent accidental theory stitching:

1. The candidate summary is frozen before held-out outcomes are formed.
2. A fibre keeps the complete menu object, not one selected pairwise gap.
3. The null leaves \(P(Y\mid\phi)\) unrestricted.
4. A residual \(q(\phi)\) is inside the null and cannot be counted as a
   violation.
5. Structural rank and statistical power are separate outputs.
6. A positive result is a relation violation on declared support; it is not a
   unique label for an omitted term.

## Candidate screen

| candidate | transferred or changed object | closest collision | scientific increment | decision |
|---|---|---|---|---|
| Fibre-conditional sufficiency audit | Candidate-preserving DCE fibre, efficient residual exposure, finite rank certificate | General misspecification tests, T-optimal design, conditional-moment tests | Shows that candidate-efficient support can have zero visibility for raw-coordinate departures; gives a pre-outcome certificate and a repair algorithm | **Primary** |
| Policy-weighted fibre audit | Weight the exposure matrix by the derivative of a declared WTP or policy functional | Robust policy analysis and decision-aware uncertainty | Ranks invisible directions by policy consequence, not only detectability | Extension after a policy estimand is frozen |
| Sequential adaptive fibre audit | Active-learning style wave-by-wave selection of the next candidate-preserving fibre | Bayesian/adaptive DCE design | Allocates later tasks to unresolved blind directions while preserving a holdout randomization reference | Future work unless a preregistered two-wave design is added |
| Decision-rule perturbation | Pair fibres with attendance, inertia, or non-trading signatures | Kazagli--de Lapparent; attribute non-attendance literature | Useful negative controls, but it does not create a new estimand | Control layer only |
| Conformal choice sets | Transfer conformal calibration to choice probabilities or WTP | Generic conformal prediction and calibrated classification | Coverage is not a candidate-basis sufficiency test; a direct transplant would be decorative | Reject as primary |
| Cross-context transport audit | Domain adaptation or covariate-shift correction for choice models | Large ML/DCM transfer benchmarks | Important for external validity, but too broad and weakly tied to the current paper | Reject as primary |
| Adversarial support sabotage and repair | Deliberately maximize candidate precision subject to zero exposure, then add minimal repair profiles | D-optimal and T-optimal designs | Makes the structural blind spot falsifiable and visually clear | Key experiment inside primary |

## Why the primary object survives

Classical lack-of-fit designs compare a specified candidate with a specified
rival. A flexible specification test compares fitted predictions with observed
outcomes on the support already available. The fibre audit fixes the candidate
menu summary and changes only raw profiles within its fibre. The departure is a
conditional residual function, with the null response surface left unrestricted
inside the summary. The design criterion is therefore

\[
  \mathcal I(h)=\mathbb E\left[\kappa_Z
  \operatorname{Var}\{h(X)\mid Z\}\right],
\]

and the finite certificate is the rank and spectrum of its support matrix.
This is a different target from coefficient precision or distance between two
pre-specified response surfaces.

The reverse-design experiment gives the most compact evidence. On the declared
finite support, the candidate-index D-optimal endpoint design has candidate
variance 4.00 and residual exposure rank 0. A fibre-aware design with at least
10% mass in each non-singleton fibre has residual rank 2 and minimum exposure
eigenvalue 0.1805. This is a structural counterexample to the assumption that
ordinary efficiency implies model-criticism coverage. It is not a claim that
all D-efficient designs are blind.

## Boundaries found in the new literature pass

Fok and Paap's JOCM paper provides a formal MNL/IIA misspecification boundary
through pairwise composite likelihood and GMM moments
([link](https://doi.org/10.1016/j.jocm.2024.100531)). Their moments are fit-based
and observational. The fibre audit creates a candidate-preserving task relation
before outcomes and uses the design assignment as the reference.

The JOCM work on decision-rule heterogeneity and attribute attendance shows why
choice-share differences can reflect processing, inertia, or task burden. The
current method treats these as declared nuisance directions and negative
controls. It does not claim to identify them from a single rejection.

Model discrimination and optimal-design work establishes D-, T-, and related
criteria. The present contribution is a domain restriction and certificate:
the alternative space is the residual quotient inside a candidate-preserving
fibre, with feasible discrete support and full-menu preservation. The paper
must cite this lineage and avoid claiming a new generic optimal-design
criterion.

## Final decision from this round

No replacement candidate currently clears the novelty and identification gates
more strongly than fibre-conditional sufficiency. The policy-weighted and
sequential versions are meaningful extensions, but promoting either one now
would make the manuscript broader before the empirical DCE gate is closed. The
primary story should remain one coherent object:

> A discrete-choice design can certify whether a coarsened candidate summary is
> sufficient by holding its complete menu prediction fixed, measuring the
> residual variation exposed within feasible fibres, and repairing structural
> blind spots before data collection.

The innovation is independent at the contribution level, while its primitives
are acknowledged as borrowed tools. A priority claim remains inappropriate
until the institutional Scopus and Google Scholar exports are deduplicated.
The empirical claim remains open until a paired or otherwise documented
candidate-preserving DCE is fielded.

## Next falsification gate

The next technical check should add a policy-functional weighting to the same
finite exposure matrix and verify whether it changes the selected support. If
the policy weighting is arbitrary, unstable across plausible policies, or
collapses to the ordinary exposure criterion, it will stay out of the main
paper. This gate tests whether the extension adds scientific content instead
of another label for the same eigenvalue calculation.

### Gate result

The policy-weighted trace criterion was run on the same $3\times3$ support
for a uniform deployment distribution, an interaction-focused distribution,
and a decomposition-focused distribution. All three selected a support with
structural exposure rank 1, even when the policy-direction matrix had rank 2.
The unconstrained policy score improved its target-direction exposure but lost
the second structural direction. The structural log-determinant reference
retained rank 2 and minimum eigenvalue 0.1805. This is a useful failure result:
policy weighting cannot replace the structural certificate. It will remain a
secondary sensitivity analysis only if the final paper imposes a full-rank
constraint and freezes the policy functional before design search.

The reproducible check is in `analysis/fibre_policy_weighted_design.py`, with
outputs in `results/fibre_policy_weighted_design.csv`.
