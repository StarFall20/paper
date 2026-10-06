# Novelty and journal fit audit

## Editorial verdict

The topic fits the Journal of Choice Modelling because it studies utility specification, behavioural interpretation, model validation, and the use of data-driven methods in choice analysis. The current contribution is publishable only as a mechanism-controlled validation study. A claim that the paper introduces a new ML specification algorithm would overlap with established JOCM work and recent specification agents.

The independent innovation search identifies a stronger candidate: a
model-equivalent choice-pair test. It tests a model-implied equality created
by the task design, so it can reject a good aggregate fit when the utility
basis is wrong. This direction has a clear novelty claim only after a valid
paired-task instrument is confirmed; otherwise it remains a simulation method
and the manuscript should retain the narrower benchmark framing.

The candidate is adjacent to Fok and Paap's 2025 JOCM tests, which use
alternative pairs and overidentifying moments to diagnose MNL/IIA
misspecification. The paper must state this overlap explicitly and limit any
novelty claim to the cross-task equality of utility-difference vectors after a
systematic search confirms that the construction is not already established.

## Closest prior work

| Study | What it already contributes | Required distinction in this paper |
|---|---|---|
| Ortelli et al. (2021) | Multi-objective variable-neighbourhood search for interpretable specifications | Test mechanism recovery and decision consequences under controlled misspecification |
| Hernandez et al. (2023) | Association rules and random forests that assist a portfolio-choice specification | Use respondent-grouped nested validation and separate observable structure from latent heterogeneity |
| Beeramoole et al. (2023) | Bi-level extensive hypothesis testing for nonlinearities, heterogeneity, and correlation in Mixed Logit | Evaluate whether a smaller diagnostic workflow recovers the right mechanism and WTP consequences |
| Delphos (Nova et al., 2026 preprint) | Deep-RL agent that learns sequential specification policies and compares with VNS on Swissmetro and DECISIONS | Provide a mechanism benchmark with explicit failure boundaries and decision-regret outcomes |
| Sfeir et al. (2026) | LLM-generated specifications evaluated for fit, behavioural plausibility, and complexity | Keep the diagnostic learner inside a reproducible, non-generative feature-selection pipeline and test structural recovery |
| Zhao et al. (2020) | Direct ML-versus-logit comparison of prediction and behavioural outputs | Refit a behavioural model after diagnostics and measure recovery, calibration, and regret |
| Fok and Paap (2025) | Composite-likelihood and GMM overidentification tests using pairs of alternatives for MNL/IIA misspecification | Test equality across different choice tasks with equal candidate utility differences and different attribute decompositions; do not claim a general MNL misspecification test |

## Main overlap risks

1. “ML-assisted utility specification” alone is too broad and is already represented in JOCM.
2. A simulation with preselected candidate terms can show workflow mechanics, but it cannot establish broad discovery ability.
3. Matching a fully structured MNL does not demonstrate that ML improves an informed analyst; the value is parsimony, recovery, and robustness under limited prior knowledge.
4. The original implementation labelled MNL forward selection as ML-assisted. The selector now uses a respondent-level Random Forest diagnostic to rank candidate terms, followed by nested behavioural refitting.

## Revised contribution statement

The paper develops a mechanism-controlled benchmark for deciding when data-driven utility diagnostics are useful in discrete choice modelling. It separates three sources of complexity: observable nonlinear and interaction terms, discrete segmentation, and continuous latent taste heterogeneity. It evaluates the workflow with predictive accuracy, coefficient and WTP recovery, calibration, specification stability, choice-share error, and decision regret under grouped validation. The output is a mechanism-specific selection rule that tells analysts when term discovery is sufficient and when the model must represent heterogeneity directly.

If the paired-task condition is met, the revised contribution statement becomes:
the paper develops a candidate-preserving paired-task randomization test for
utility specification. The test preserves candidate-model utility differences
across randomized task pairs while changing the attribute decomposition. Its
equivalence violation measures whether the observed choice process supports
the candidate basis and can stratify predeclared transformations. It does not
identify a unique omitted term. The existing ML and process analyses serve as comparison and boundary

## Journal and reader alignment

The primary readers are choice modellers who need a disciplined specification workflow, researchers comparing behavioural and data-driven models, and applied analysts who report policy or welfare consequences. The manuscript should lead with the modelling decision and its behavioural consequence. The medical-device simulation should remain a controlled example; LPMC and Swissmetro must carry the evidence of transferability.

## Evidence required to close the novelty gap

- Report RF candidate ranking, true-positive and false-positive term recovery, selected-term count, and stability across replications.
- Add a candidate-library coverage or omitted-term stress test so the workflow has an explicit failure boundary.
- Compare against at least one established assisted-search baseline and a pure flexible learner under the same grouped validation budget.
- Complete full Mixed Logit, WTP recovery, calibration, 100 replications, and the LPMC/Swissmetro external checks.

## Decision

The direction is worth pursuing for JOCM. The paper should be presented as a validation and decision-support contribution, with algorithmic novelty claims removed. Acceptance prospects depend on demonstrating that the benchmark yields information unavailable from a single fit comparison or a new search heuristic.

Sources: [JOCM aims and scope](https://shop.elsevier.com/journals/journal-of-choice-modelling/1755-5345); [Ortelli et al.](https://doi.org/10.1016/j.jocm.2021.100285); [Hernandez et al.](https://doi.org/10.1016/j.jocm.2022.100397); [Beeramoole et al.](https://doi.org/10.1016/j.jocm.2023.100409); [Delphos](https://arxiv.org/abs/2506.06410); [Sfeir et al.](https://doi.org/10.1016/j.jocm.2026.100623); [Zhao et al.](https://doi.org/10.1016/j.tbs.2020.02.003).
