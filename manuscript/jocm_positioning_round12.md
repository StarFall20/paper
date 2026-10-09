# JOCM positioning audit round twelve

## What the recent journal signals show

Recent Journal of Choice Modelling papers place methodological contributions in
four connected areas:

1. **Design algorithms.** [Mao, Kessels, and van der Zanden (2025)](https://www.sciencedirect.com/science/article/pii/S1755534525000144)
   use simulated annealing to improve Bayesian DCE design over coordinate
   exchange. The paper values a transparent design criterion and extensive
   computational evidence.
2. **Processing validity.** [Boxebeld (2024)](https://www.sciencedirect.com/science/article/pii/S1755534524000216)
   reviews ordering effects across 85 studies, while [Jonker (2024)](https://www.sciencedirect.com/science/article/pii/S1755534524000265)
   evaluates level overlap and color coding with randomized DCEs. These papers
   make task processing and attribute attendance central design concerns.
3. **Model and data flexibility.** [Łukawska, Jensen, and Rodrigues (2025)](https://www.sciencedirect.com/science/article/pii/S175553452400068X)
   combine a neural network with interpretable mixed logit parameters for
   context-dependent heterogeneity. [Salas et al. (2025)](https://www.sciencedirect.com/science/article/pii/S1755534525000016)
   use Shapley-based decomposition for attribute importance in MNL.
4. **Model-building practice and prediction.** [Nova, van Cranenburgh, and Hess (2025)](https://www.sciencedirect.com/science/article/pii/S1755534525000259)
   study specification workflows with a serious game, while [Zhu and Si (2024)](https://www.sciencedirect.com/science/article/pii/S1755534524000034)
   compare machine learning and discrete choice prediction for image choices.

## Fit of the current contribution

The parity-fibre contribution fits the journal most strongly as a **design-based
specification audit**. It supplies a pre-outcome task-construction rule, a
falsifiable candidate-basis restriction, and a computational design criterion.
The smart-device attributes are a transparent testbed. The paper should not be
positioned as a general medical-device preference study or as another XGBoost
versus MNL race.

The recent JOCM evidence makes three controls mandatory in the main design:

- order and wording counterbalancing, with response-time and certainty measures;
- level overlap or an equivalent semantic-comprehension screen, because
  attribute non-attendance can create an odd process response;
- a full design comparison against a conventional Bayesian/D-efficient design,
  reporting candidate utility balance, odd information, task burden, and size.

The current repository already has order controls, an even-component diagnostic,
calibration uncertainty, and a fibre information criterion. The conventional
design comparison and a human pilot remain open.

## Journal-facing contribution paragraph

> We propose a design-based specification audit for discrete choice models. The
> method searches for nontrivial attribute fibres on which a candidate model
> predicts the same complete menu, then pairs reflected profiles so that an
> explicit even processing component is held fixed. A pre-outcome information
> criterion selects tasks that maximize sensitivity to an odd omitted utility
> direction. Simulation and an application-specific smart-device design show
> how the audit can reveal structure that ordinary likelihood and residual
> diagnostics miss, while calibration, ordering, and semantic controls define
> its empirical limits.

## Positioning decision

The original predictive crossover remains a benchmark. The submission story
should lead with the specification audit and the DCE design criterion, then use
the smart-device generator to demonstrate recoverability. This positioning
matches the journal's recent interest in efficient designs, processing validity,
interpretable machine learning, and transparent model-building workflows.
