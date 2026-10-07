# Round 26: support-complete fibre audit and independent novelty screen

## What changed in the innovation search

The previous version used a finite residual dictionary (R). That was a real
weakness: a reviewer could say that the proposed test only detects the
directions chosen by the authors. The new refinement keeps the same candidate
summary and the same complete-menu fibre, but replaces the hand-picked
dictionary with the full finite contrast space of each supported fibre.

For a fibre (g) with (K_g) feasible profiles and conditional design weights
(w_g), let (Q_g) be any orthonormal basis for the (K_g-1) vectors
orthogonal to the constant vector. The support-complete exposure block is

\[
 E_g=\alpha_g\kappa_g Q_g^{\top}
       \{\operatorname{diag}(w_g)-w_gw_g^{\top}\}Q_g.
\]

The direct sum (E=\oplus_g E_g) has one coordinate for every non-constant
response pattern on every non-singleton fibre. Its rank is therefore a
support certificate, while its smallest eigenvalue is a conditioning measure.
If all profiles in the declared finite fibre grid receive positive mass,
\(\operatorname{rank}(E)=\sum_g(K_g-1)\). A zero block means that the design
cannot distinguish some raw-profile response pattern regardless of sample
size. This is stronger than a certificate for a chosen polynomial or
interaction dictionary. It remains a finite-support statement; it does not
claim that a finite DCE tests arbitrary functions outside the declared
support.

The implementation is `analysis/support_complete_fibre_design.py`. On the
same (3\times3) support with \(\phi(a,b)=a+b\), the candidate-precision
endpoint design has candidate-summary variance 4.00 and support-complete rank
0. The support-complete design has rank 4 (the exact dimension of the four
non-singleton-fibre contrasts) and minimum exposure eigenvalue 0.1222, while
retaining positive mass on every profile. The result is in
`results/support_complete_fibre_design.csv`.

## Why this is an independent increment

The contribution is no longer “add a residual term to a logit model.” The
object being designed is the **refutability of a representation**: after
conditioning on the complete candidate menu vector, the experiment allocates
support to every observable direction left in the finite fibre. The candidate
probability surface remains unrestricted. A conventional lack-of-fit test can
compare a fitted model with a rival response surface, but it does not force
the candidate prediction to be held fixed while spanning the discarded raw
coordinates. A T-optimal design can be excellent for a specified rival and
still have rank zero for the conditional residual space. The support-complete
certificate makes this distinction auditable without selecting a rival
function.

The refinement also clarifies the scope. It is a complete test on a declared
finite support, not a claim of nonparametric identification on a continuous
attribute domain. For a continuous domain or an enormous discrete support,
the paper must return to a declared residual class (for example, a bounded
RKHS subspace) and report its effective rank.

## Independent literature screen

I screened 222 Crossref records for Journal of Choice Modelling from 2021--2026
and then read the closest open abstracts or full texts. The title corpus has
no paper using the terms “sufficiency”, “invariance”, “randomization” or
“adversarial” for a choice-model specification audit. This is a discovery
signal, not a priority claim.

The closest works establish different objects:

| work | what it contributes | boundary for this paper |
|---|---|---|
| Ortelli et al. (2021), [assisted specification](https://doi.org/10.1016/j.jocm.2021.100285) | multi-objective combinatorial search over utility specifications | searches candidate models on observed support; it does not design candidate-equivalent tasks or test conditional sufficiency |
| Parady, Ory and Walker (2021), [validation review](https://doi.org/10.1016/j.jocm.2020.100257) | documents the field’s heavy reliance on fit statistics and scarce external validation | motivates a design-based validation target; it does not supply one |
| Liu et al. (2023), [ICLV versus multi-task DNN](https://doi.org/10.1016/j.jocm.2023.100431) | compares theory-driven and data-driven prediction and interpretation | does not test whether a reduced utility representation is sufficient |
| van Cranenburgh et al. (2024), [decision-rule assumptions in designs](https://doi.org/10.1016/j.jocm.2023.100465) | shows that the assumed decision rule used to build an efficient design affects preference recovery | studies design-induced behavioural effects; it does not hold candidate menu predictions fixed to test discarded attributes |
| Fok and Paap (2025), [MNL misspecification tests](https://doi.org/10.1016/j.jocm.2024.100531) | pairwise composite-likelihood and GMM overidentification tests for broad MNL/IIA misspecification | fit-based and observational; it does not create a candidate-preserving fibre or a support-rank certificate |
| Mao et al. (2025), [Bayesian optimal designs by simulated annealing](https://doi.org/10.1016/j.jocm.2025.100551) | improves parameter-precision design search | optimizes information for a specified model, not refutability of a coarsened representation |
| Salas, De la Fuente and Astroza (2025), [Shapley attribute importance](https://www.sciencedirect.com/science/article/pii/S1755534525000016) | decomposes MNL fit and predictions into attribute contributions | interprets a fitted model; it does not test invariance after conditioning on its predictions |
| Hoshino and Yanagi (2026), [conditional randomization tests for exposure mappings](https://doi.org/10.1002/jae.70076) | tests whether a coarsened exposure mapping is adequate under known treatment assignment | the closest methodological collision; its exposure is network treatment interference, while this paper’s object is a complete menu utility representation and raw-attribute fibre in a DCE |
| Shubatt and Yang (2024/2026), [tradeoff complexity](https://arxiv.org/abs/2401.17578) | models choice errors generated by difficult tradeoffs and shows robustness when complexity is held fixed | supplies a nuisance mechanism and control principle; it does not test utility-basis sufficiency or construct a support certificate |

The Hoshino--Yanagi comparison prevents an inflated claim. The manuscript must
say that conditional-randomization logic is a transferable ingredient. The
independent increment is the DCE-specific construction of a complete-menu
candidate fibre, the support-complete contrast certificate, and the separation
of structural visibility from behavioral power.

## Candidates rejected as primary contributions

1. **Generic conditional randomization test.** It now overlaps too closely
   with Hoshino and Yanagi (2026). Keep it as the inferential implementation,
   not as the novelty label.
2. **Graph or cycle terminology.** A graph Laplacian is an equivalent way to
   represent the same within-fibre contrasts. It adds notation without a new
   estimand.
3. **Policy-weighted exposure.** The earlier audit showed that an unconstrained
   policy score can collapse structural rank. It remains a sensitivity
   analysis after a full-rank constraint, not a replacement theory.
4. **A neural adversarial critic as the headline.** A critic trained on the
   same outcomes would reintroduce data-dependent search and selective
   inference. The support-complete certificate is safer and stronger; an
   RKHS or adversarial extension belongs in future work unless its class and
   holdout are frozen before outcomes.

## Current novelty decision

The strongest defensible contribution is now:

> A design-based, support-complete conditional-sufficiency audit for a
> coarsened utility representation in a multinomial DCE. It constructs
> non-singleton fibres that preserve the complete candidate menu prediction,
> spans every non-constant response contrast on the declared finite support,
> and reports a rank/eigenvalue certificate before estimation.

This is a meaningful advancement over a generic lack-of-fit comparison, while
remaining honest about its finite-support scope. The primary empirical claim
still requires a randomized candidate-preserving DCE. The public Swissmetro
run is an external implementation check only.
