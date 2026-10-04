# Method choices informed by relevant choice-modelling papers

## Assisted specification

Ortelli et al. (2021), *Assisted specification of discrete choice models*, formulates utility specification as a search over candidate models and makes the fit-parsimony trade-off explicit. This supports the current design choice to evaluate a candidate utility set inside validation and report selected terms, predictive fit, and decision regret together. The present project uses a simpler forward-selection prototype; it does not claim to reproduce their multi-objective metaheuristic.

Source: https://doi.org/10.1016/j.jocm.2021.100285

## Data-driven diagnostics

Hernandez et al. (2023), *Data-driven assisted model specification for complex choice experiments data*, uses association rules and random forests to support behavioural model specification and to contrast behavioural assumptions with flexible predictions. This motivates using ML as a diagnostic layer that proposes candidate terms for a behavioural refit. It also motivates reporting interpretation and model fit together.

Source: https://doi.org/10.1016/j.jocm.2022.100397

## Validation discipline

Hillel et al. (2021), *A systematic review of machine learning classification methodologies for modelling passenger mode choice*, identifies recurring problems in performance estimation, hyperparameter selection, and model analysis. The revised protocol addresses these risks with respondent-level grouping, nested candidate selection, fixed data partitions, and probability-based metrics.

Source: https://doi.org/10.1016/j.jocm.2020.100221

## Design consequence for this paper

The main result should answer a model-selection question: under which mechanism conditions does a diagnostic learner identify utility structure that improves a refitted behavioural model and the resulting product decision? Accuracy remains a descriptive outcome. The primary evidence is the joint pattern of term recovery, calibration, choice-share error, and decision regret under the stated validation regime.
