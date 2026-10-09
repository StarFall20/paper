# Literature-overlap audit: candidate-preserving utility fibres

**Audit date:** 2026-10-06  
**Scope:** Journal of Choice Modelling revision; candidate-preserving DCE fibre test  
**Purpose:** determine whether the proposed contribution is a direct duplicate, a stitched combination of existing ideas, or a distinct bounded method.

## Search protocol

The search used exact-title and concept queries in Google Scholar, publisher pages, DOI records, author manuscripts, and OpenAlex-style discovery. Exact-title Scholar checks were performed for:

- `"Which experiments test a model"` — 5 results; the Healy–Leo article was the exact lead result.
- `"Discrete Choice with Presentation Effects"` — 1 exact lead result; 4 Scholar citations.
- `"Tradeoffs and Comparison Complexity"` — about 37 results; the Shubatt–Yang article was the exact lead result; 34 Scholar citations.
- `"Latent utility and permutation invariance"` — 1 exact lead result; 2 Scholar citations.
- `"Specification testing of discrete choice models"` — 1 exact lead result; 20 Scholar citations.
- `"paired valuation tasks" discrete choice noise` — 21 results; the McGranaghan et al. papers were the leading paired-task/noise results.
- `"utility-preserving" "choice experiment"` — 8 results; the leading hit concerned utility-preserving trade-offs in valuation, not a DCE fibre test.
- `"model testing" "discrete choice"` — about 1,420 results; broad screening was required because the query mixes specification tests, model selection, and applications.

Direct Scopus access was attempted through the public site and Elsevier API. The public site returned **403 Forbidden** and the API returned **401 Unauthorized** without an institutional/API key. No Scopus record count is reported. The audit therefore uses the accessible scholarly records above and records the Scopus limitation explicitly.

## The proposed object being audited

The contribution is one design-and-test object:

1. Map every choice task to the complete candidate menu utility vector \(\phi(T)\).
2. Search for a non-trivial fibre \(F_\phi=\{T:\phi(T)=\phi\}\).
3. Pair reflected tasks inside the same fibre while balancing a pre-declared nuisance signature (weighted comparison distance, changed-attribute count, raw level-change load, and sign counts).
4. Assign one respondent-level orientation coin to the full task block.
5. Use the odd response \(m_{odd}=(m_+-m_-)/2\) as the primary candidate-relation contrast; report the even response \(m_{even}=(m_++m_-)/2\) as a scale/process diagnostic.

The estimand is a violation of the candidate relation on an observed, pre-outcome support. The method does not claim general invariance, unique omitted-term identification, or a new theory of complexity.

## Closest literature and collision assessment

| Neighbour | Shared ingredient | What the neighbour actually does | What remains distinct here | Collision verdict |
|---|---|---|---|---|
| Fosgerau (2008), *JOCM*; Chicu & Masten (2013), *Economics Letters* | Specification testing in discrete choice | Uses smoothed residuals, nonparametric statistics, or choice-set variation to detect misspecification | Creates candidate-preserving observed task fibres and tests a relation that is invisible to a standard fit or residual check | **Partial overlap: target, not design or estimand** |
| Healy & Leo (2026), *JET* | Experiments designed to test a model | Characterizes deterministic preference-model testing/classification with a labeled permutohedron and finds minimal experiments | Stochastic DCE test on utility fibres; nuisance-balanced reflections; cluster sign-flip inference; odd/even diagnostic | **Closest high-level neighbour; no direct duplication** |
| Breitmoser (2017), *Discrete Choice with Presentation Effects* | Presentation/process effects and alternative permutations | Models UI, ordering, and labeling effects as a second focality component and estimates it from experiments | Uses process variables as a declared nuisance signature and a negative-control diagnostic; no second utility theory is introduced | **Substantial boundary overlap; keep as an explicit identification limit** |
| Shubatt & Yang (2024), *Tradeoffs and Comparison Complexity* | Comparison difficulty can change responses at equal utility gaps | Builds a behavioral theory of pronounced tradeoffs and cognitive noise | Does not model complexity; balances a transparent signature so complexity is less able to explain the odd contrast | **Confound to control, not a competing theory** |
| McGranaghan et al. (2024), *AER* | Paired tasks and differential noise | Shows paired choice tasks can create apparent preference effects through task-specific noise; proposes paired valuations | Applies pairing to candidate-preserving DCE relations, adds respondent-level orientation, and separates odd signal from even scale/process signal | **Design-validity precedent; not the same estimand** |
| Lindberg, Eriksson & Mattsson (1995); Mattsson, Weibull & Lindberg (2014); de Palma & Kilani (2007) | The word “invariance” and random-utility restrictions | Characterize latent achieved-utility or conditional maximum-utility distributions, including MNL/GEV implications | Tests observable task-level relation violations; no latent distribution characterization is claimed | **Terminology/theory neighbour; do not call this a new invariance theorem** |
| Allen & Rehbeck (2024), *Journal of Econometrics* | Latent utility and permutation invariance | Uses invariance of unobservables to obtain revealed-preference inequalities and partial identification | Uses observed task construction to test a candidate map; no latent exchangeability or partial-ID result | **Conceptual overlap only** |
| Singh, Liu & Yoganarasimhan (2023/2024) | Permutation-invariant choice functions | Builds set-function/neural demand estimators invariant to product permutations | Provides a falsification/diagnostic design, not a demand estimator | **No direct collision** |
| Boxebeld (2024), *JOCM*; Carlsson et al. (2012), *JOCM*; Pedersen et al. (2011), *JOCM* | Ordering, learning, cost inclusion, and forced/unforced design effects | Documents design artefacts and parameter/scale changes under different presentations or cost attributes | Uses these effects to define nuisance controls and to state when the fibre contrast is not interpretable | **Empirical motivation and threat literature** |
| Ortelli et al. (2021), *JOCM* | Utility-specification search | Treats specification search as multi-objective combinatorial optimization | Searches task space for a falsification support after the candidate map is fixed; does not optimize fit or parsimony | **No direct collision** |

## Does the idea look stitched together?

There is no direct duplicate in the screened literature that combines all four defining conditions: (i) equality of the complete candidate menu utility vector, (ii) a reflected within-fibre DCE task pair, (iii) balance on an explicit presentation/complexity signature, and (iv) a respondent-clustered odd/even sign-flip decomposition.

The idea would look stitched together if the manuscript presents these ingredients as separate “innovations” borrowed from different literatures. The defensible unit is one estimand and one design algorithm: **a nuisance-balanced, candidate-preserving fibre audit**. The paired-task literature, presentation-effects literature, complexity literature, and invariance literature should be cited as constraints that define the audit's validity conditions.

The wording must avoid four overclaims:

- “first invariance test for DCMs” — false because latent achieved-utility and conditional-maximum invariance are established literatures;
- “complexity-free identification” — false because an odd process effect can remain;
- “unique diagnosis of omitted utility terms” — false because random taste, attention, framing, and task-specific noise can also violate the relation;
- “general model test” — false because a non-trivial fibre requires a coarse candidate map; a full attribute-linear map can have singleton fibres.

## Novelty verdict

**Direct duplication:** not found in the screened records.  
**Partial overlap:** material and must be acknowledged, especially Healy–Leo, Breitmoser, Shubatt–Yang, and McGranaghan et al.  
**Independent contribution:** plausible as a bounded DCE design-and-diagnostic method, provided the paper treats the full fibre construction, nuisance balance, orientation randomization, and odd/even estimand as one inseparable object.  
**Current publication risk:** medium. The risk is rhetorical and evidentiary, not a discovered duplicate. A reviewer may still call it a recombination unless the paper includes a preregistered human/external DCE and a head-to-head benchmark against an added-term LR test and ML residual diagnostics.

## Required changes after this audit

1. Add the nearest-neighbour table to the introduction or online appendix and state the four defining conditions in one paragraph.
2. Rename “invariance” claims as **candidate-relation audit** or **utility-fibre diagnostic** except when discussing prior invariance theory.
3. Keep comparison complexity and presentation effects as nuisance threats/negative controls, not as additional mechanisms in the theory.
4. Report the null, localized omitted-term, diffuse omitted-term, random-taste, complexity-only, and combined-threat simulations with finite-sample power and a clear non-identification statement.
5. Add the external DCE comparison against an added-term LR test and residual-based ML check. This is the remaining novelty gate for a submission-level claim.

## Sources

- [Healy & Leo, *Which experiments test a model?*](https://doi.org/10.1016/j.jet.2026.106191)
- [Fosgerau, *Specification testing of discrete choice models*](https://www.sciencedirect.com/science/article/pii/S1755534513700216)
- [Chicu & Masten, *A specification test for discrete choice models*](https://www.sciencedirect.com/science/article/pii/S0165176513003972)
- [Breitmoser, *Discrete Choice with Presentation Effects*](https://epub.ub.uni-muenchen.de/58047/)
- [Shubatt & Yang, *Tradeoffs and Comparison Complexity*](https://arxiv.org/abs/2401.17578)
- [McGranaghan et al., *Distinguishing Common Ratio Preferences from Common Ratio Effects Using Paired Valuation Tasks*](https://doi.org/10.1257/aer.20221535)
- [Lindberg, Eriksson & Mattsson, *Invariance of Achieved Utility in Random Utility Models*](https://doi.org/10.1068/a270121)
- [Mattsson, Weibull & Lindberg, *Extreme values, invariance and choice probabilities*](https://doi.org/10.1016/j.trb.2013.10.014)
- [de Palma & Kilani, *Invariance of conditional maximum utility*](https://doi.org/10.1016/j.jet.2005.05.010)
- [Allen & Rehbeck, *Latent utility and permutation invariance*](https://doi.org/10.1016/j.jeconom.2024.105844)
- [Singh, Liu & Yoganarasimhan, *Choice Models and Permutation Invariance*](https://arxiv.org/abs/2307.07090)
- [Boxebeld, *Ordering effects in discrete choice experiments*](https://doi.org/10.1016/j.jocm.2024.100489)
- [Pedersen et al., *Does the Inclusion of a Cost Attribute in Forced and Unforced Choices Matter?*](https://www.sciencedirect.com/science/article/pii/S1755534513700447)
- [Ortelli et al., *Assisted specification of discrete choice models*](https://doi.org/10.1016/j.jocm.2021.100285)
