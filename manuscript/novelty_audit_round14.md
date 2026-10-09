# Round fourteen: anti-stitch novelty audit and complexity-balance correction

## Decision

The paper should make one methodological claim:

> A discrete-choice specification can be audited by randomizing reflected
> tasks inside a candidate-preserving utility fibre while matching a declared
> nuisance signature; the remaining odd response tests the candidate relation
> along a direction that ordinary support does not identify.

This is an experimental-design method. It is not a new utility theory, a new
invariance group, or a general theory of cognitive complexity.

## Why the method is a single idea

The object that generates every component is the candidate fibre
\(\mathcal F_\phi=\{T:\phi(T)=\phi\}\).

1. The fibre supplies profiles with identical candidate menu predictions.
2. A reflection supplies two orientations of the same candidate relation.
3. The nuisance signature specifies which declared process features must also
   remain fixed.
4. The odd contrast is the only response component that changes sign under the
   reflection.
5. The information criterion chooses the most sensitive pairs from the same
   constrained pool.

The reflection, nuisance balance, odd contrast, and task criterion are thus
successive steps of one identification argument. They are not independent
features collected from unrelated papers.

## What is inherited and what is added

| Element | Existing knowledge used | Added contribution in this paper |
|---|---|---|
| Reflection and odd/even algebra | Symmetric-design and local contrast arguments | Applied to an observed attribute fibre defined by a candidate's full menu vector |
| Moderate-utility response | Choice probabilities of the form \(F(\Delta V/D)\) | Uses the distance term as a predeclared nuisance balance condition |
| Comparison complexity | Weighted \(L_1\) distance can alter choice noise | Converts that distance into a design constraint and a negative-control benchmark |
| Model-testing experiments | Experimental designs can separate preference models | Provides a stochastic DCE implementation with candidate-menu preservation, calibration, and respondent-level randomization |
| Efficient design | Information criteria choose informative tasks | Optimizes the odd signal only after fibre and nuisance constraints are imposed |

The added contribution is the design object and its operating protocol. The
paper should cite the parent ideas and state the boundary directly.

## Nearest-neighbour collision audit

### Healy and Leo (2026)

“[Which Experiments Test a Model?](https://doi.org/10.1016/j.jet.2026.106191)”
characterizes experiments that test or classify deterministic preference
models using a labeled permutohedron and a minimal-experiment algorithm. It is
the closest general model-testing paper.
The present design addresses a different object: a stochastic DCE with a
parametric candidate utility basis, observed profiles that share a complete
candidate menu vector, and process nuisance that must be balanced. It does not
claim to characterize all testing experiments or to find a globally minimal
experiment.

### Shubatt and Yang (2024, current 2026 version)

Their [comparison-complexity model](https://arxiv.org/abs/2401.17578) writes binary choice as
\(G((U(x)-U(y))/d_{L1}(x,y))\), with a value-weighted \(L_1\) distance. This
creates a direct failure mode for a simple fibre design: equal candidate gaps
do not imply equal choice probabilities when \(d_{L1}\) differs. The present
paper uses that result as a design constraint. It does not propose their
complexity model or claim to explain cognitive errors.

### McGranaghan et al. (2024)

Their [paired-task results](https://doi.org/10.1257/aer.20221535) show that differential noise can make equal latent
values produce unequal choice rates. This is a direct reason to avoid treating
any paired-choice difference as an omitted utility term. The nuisance-balanced
design adds a complexity-only negative control and restricts the utility
interpretation to pairs with a matched declared process signature.

### Latent permutation invariance and earlier DCE symmetry work

The 2024 Journal of Econometrics paper studies latent utility and permutation
invariance. Earlier health DCE work studies symmetry and task-order effects.
Those papers do not construct candidate-preserving observed attribute fibres
or an odd/even randomized DCE audit. The present paper must avoid “first
invariance test” language and call its transformation an observed design
device.

## Falsification of the earlier loose design

The first smart-device grid matched candidate utilities and raw squared
distance. An audit against the weighted \(L_1\) comparison metric found that
none of the five selected pairs matched the full declared signature. The
corrected design requires equality of:

- candidate menu vector;
- weighted \(L_1\) distance;
- number of changed attributes;
- raw attribute-level load;
- counts of positive and negative component changes;
- odd interaction antisymmetry.

The strict pool contains 58 pure-odd reflections. A five-task selection has
candidate gaps approximately 0.32, 0.38, 0.57, 0.80, and 0.98, with odd
signals 0.90, 0.90, 0.90, 0.90, and 0.55.

In a binary moderate-utility stress test with 600 respondents, 200
replications, and 999 respondent-cluster sign flips, the loose design rejected
the candidate relation in 18.5% of complexity-only samples. The balanced
design rejected in 6.0%. When the omitted interaction was added, both designs
rejected in 100% of samples. These results do not show that complexity has been
eliminated in every setting. They show why the added balance condition is
necessary for the utility-specific interpretation.

## Claim that survives the audit

The defensible claim is:

> We develop a nuisance-balanced, candidate-preserving DCE design for testing
> utility-basis sufficiency. The design searches observed attribute fibres,
> matches a predeclared process signature, and uses a randomized odd response
> as the confirmatory contrast. A complexity-only stress test demonstrates the
> cost of omitting the balance condition.

The following claims must be removed:

- first general symmetry or invariance test for choice;
- proof that any rejection identifies a particular omitted interaction;
- proof that the design controls all cognitive complexity;
- universal power superiority over standard designs;
- empirical preference conclusions for the smart-device market.

## Remaining evidence gate

The design-level novelty is now coherent and independently motivated. A
投稿级 paper still needs a human or external-data DCE with independent
calibration, semantic screening, and the frozen assignment protocol. Until
that gate is closed, the manuscript should describe the smart-device results
as design validation and planning evidence.
