# Round six independent innovation audit

## Starting point

The supplied Word manuscript asks when XGBoost improves prediction over MNL in
a synthetic smart-medical-device choice experiment. Its main result is a
complexity-dependent performance crossover. That question is useful as a
benchmark, but it is below the current Journal of Choice Modelling contribution
threshold on its own. Recent JOCM work already covers assisted utility search,
machine-learning comparisons, attention and process data, context-dependent
heterogeneity, model-choice sensitivity, and new behavioural models.

The manuscript needs one research object that changes what can be learned from
choice data. A new algorithm label or a new attribute context would leave the
central problem unchanged.

## What the recent JOCM literature leaves open

| recent direction | established advance | unresolved problem relevant here |
|---|---|---|
| Assisted specification and ML comparisons | Search algorithms and flexible predictors improve the analyst's ability to explore utility forms and compare holdout performance. | A better holdout score does not show that the candidate utility library treats behaviorally equivalent tasks consistently. |
| Attention, non-trading, inertia, and experience | Process data and latent decision rules explain departures from static utility maximization. | A specification audit needs a way to detect a relation violation before assigning it to a particular latent mechanism. |
| Context-aware and latent-class models | Preference parameters can vary with context, segmentation, and error structure. | Scale, taste, process, and functional-form changes remain difficult to separate from a single observational fit. |
| Model-choice and welfare studies | Different specifications can produce materially different loss-aversion or welfare quantities. | Analysts need a falsifiable diagnostic for the candidate utility representation before reporting a policy quantity. |
| Bayesian optimal DCE design | Designs can maximize parameter information or discriminate among prespecified competing models. | Standard criteria do not construct a test oracle from the equivalence classes implied by the fitted candidate basis. |

These gaps motivate a diagnostic relation, not another flexible predictor.

## Candidate routes considered

### Route A: BMST/UFIT — retain as the main contribution

For a candidate basis `b_j(x)`, define the alternative-difference vector

$$
d(x)=\{b_j(x)-b_k(x):j<k\}.
$$

The single new object is the **candidate utility fibre**: the set of task
versions with the same `d(x)`. A candidate Logit model assigns the same
choice-probability vector to every member of a fibre. BMST makes that model
implication testable by assigning one member of each pair to each respondent
and using respondent-cluster randomization. UFIT is the formal relation.

The contribution is a candidate-conditioned exchangeability test. It does not
combine a learner, a process model, and a policy rule. A rejection means that
the observed choice process violates the candidate relation on the declared
intervention domain. The negative controls determine whether the violation is
consistent with scale, taste, order, or functional-form changes; they do not
pretend to identify a unique mechanism.

### Route B: fibre-aware falsification design — integrate into Route A

Standard DCE design targets parameter precision or discrimination among a set
of fully specified models. BMST can choose feasible transformations after
freezing the candidate by maximizing the weakest predicted separation across a
declared family of out-of-library utilities, subject to the exact
candidate-fibre constraint and a movement budget. This is useful as a design
criterion for the BMST instrument. It remains part of the one contribution,
not a second independent claim.

The existing maximin audit supports this role: the constrained design raised
the weakest probe score from 1.142 to 1.632, kept null rejection at 0.06, and
raised nonlinear power from 0.82 to 1.00 in the frozen simulation. The
repeated-shift solution is retained as a failure boundary. The manuscript must
describe the alternative family and movement budget before treating the design
as informative.

### Route C: choice-set conformal prediction — reject as the main route

Conformal prediction is a plausible transfer from distribution-free machine
learning. A choice-set-specific prediction set could provide coverage for the
chosen alternative and support abstention. A 2025 mode-choice paper already
wraps tree models with inductive Mondrian conformal prediction. A generic
conformal layer would therefore be an incremental application. It may become a
technical extension after BMST, but it cannot carry the independent novelty
claim.

### Route D: model-averaged welfare stability — reject as the main route

Reporting a WTP or policy interval over multiple plausible specifications would
address model-choice uncertainty. Model averaging, robust welfare bounds, and
recent JOCM work showing model-dependent loss-aversion estimates already occupy
this space. The route would add a useful reporting practice, but it would not
replace the missing falsifiable relation.

### Route E: adversarial counterfactual stress testing — retain only as a
diagnostic comparison

A constrained perturbation could search for the smallest change in attributes or
choice sets that changes a model's policy ranking. Robust counterfactual and
partial-identification literatures already study this problem. The route is
valuable for assessing fragility, but its output is a sensitivity map rather
than evidence that the observed choice process violates a candidate utility
relation.

### Route F: environment-invariant WTP — reject as the main route

Learning preference functions that remain stable across datasets or time would
connect invariant prediction with choice modelling. Context-aware Bayesian
mixed Logit and the scale-identification literature create a difficult boundary:
stable WTP can coexist with changing error scale, and unstable coefficients can
reflect scale alone. The idea needs stronger identification than the current
data provide.

## Independent novelty statement

The paper should make one precise claim:

> BMST turns the equivalence relation induced by a frozen candidate utility
> basis into a randomized, finite-sample test on newly assigned choice tasks.
> The test asks whether observed choice probabilities are exchangeable across
> task versions that the candidate treats as utility-difference equivalent.

This advances existing work in a specific way. Axiomatic invariance results
characterize model properties. Pairwise MNL tests use within-task moments or
alternative comparisons. Metamorphic testing supplies the general idea of
using a transformed input as an oracle. BMST supplies the choice-specific
transformation, the one-member task assignment, the clustered reference
distribution, and the tested-domain estimand in one coherent design.

## Alignment with the supplied manuscript

The smart-medical-device generator should become a controlled testbed. Its
attributes and λ complexity gradient can generate the candidate, omitted
nonlinearity, interaction, scale, and order conditions needed for the BMST
negative controls. The original XGBoost-versus-MNL crossover becomes a
descriptive benchmark in the appendix. It should not remain the title claim or
the main contribution because its numerical crossover is specific to the
chosen generator.

## Submission gates

The innovation passes the conceptual gate and the simulation feasibility gate.
It has not passed the full empirical gate. Before submission, the paper must:

1. state the fibre-equivalence proposition and assignment assumptions;
2. preregister the transformation grid, movement budget, and one-member
   assignment;
3. report null size, power, repair, scale, random-taste, order, and combined
   mechanism controls at the target replication count;
4. field a paired-task supplement or obtain a dataset with the same assignment
   structure;
5. report usable-pair counts and the action for an unresolved or mixed
   violation;
6. keep the claim local to the tested fibre and avoid calling the test a
   universal MNL or invariance test.

The independent innovation is therefore selected, while the manuscript remains
below complete-submission status until the empirical instrument is available.

## Papers used for the boundary audit

- [Assisted specification of discrete choice models](https://doi.org/10.1016/j.jocm.2021.100285)
- [Predicting choices of street-view images](https://doi.org/10.1016/j.jocm.2024.100470)
- [Discrete choice experiments with eye-tracking](https://doi.org/10.1016/j.jocm.2024.100478)
- [A discrete choice modeling framework for non-trading behavior](https://doi.org/10.1016/j.jocm.2023.100413)
- [Context-aware Bayesian mixed multinomial logit model](https://doi.org/10.1016/j.jocm.2024.100536)
- [Understanding the decision-making process of choice modellers](https://doi.org/10.1016/j.jocm.2025.100562)
- [Model choice and framing effects](https://doi.org/10.1016/j.jocm.2024.100524)
- [Constructing Bayesian optimal designs for discrete choice experiments](https://doi.org/10.1016/j.jocm.2025.100551)
- [New misspecification tests for multinomial logit models](https://doi.org/10.1016/j.jocm.2024.100531)
- [Enhancing mode-choice models with conformal prediction](https://doi.org/10.2478/ttj-2025-0027)
