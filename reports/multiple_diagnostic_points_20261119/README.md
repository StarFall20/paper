# Finite estimation at several diagnostic checkpoints

The [proof](proof.md) extends the fixed-weight, fixed-predictor result to diagnostics at an arbitrary finite list of preassigned coefficient vectors. Equal-weight auxiliary rows can agree on all requested objective derivatives and empirical score covariances at those vectors. The three corrected objective functions have different existence outcomes even with the same observed choices, predictor, weight, auxiliary count and bounded attributes.

The construction keeps symmetric pairs and uses the independence of the constant function from the derivatives of even diagnostic functions. This isolates the mean absolute contrast that governs the recession slope. The setup concerns diagnostics computed from objective derivatives and score moments. The observed menus still give an exact support check. A finite coefficient with exactly zero gradient is also a sufficient certificate under full contrast rank.

Run `python verify.py --out replay.json` with Python 3.9 or later. It uses the standard library. The program solves the two-checkpoint construction using 85-digit arithmetic and recomputes the diagnostics and recession directly on all individual rows. [Numerical results](results.json) retain the nodes and errors. The maximum matched diagnostic difference is below 1e-78. The analytic proposition establishes exact matching for sufficiently small perturbations. The numerical nodes are approximations.

The first reference is the origin, where the three designs share the first Newton update. The second reference is that common update. This example establishes one checkpoint selected by the diagnostics. The general theorem allows arbitrary finite preassigned checkpoints. An unrestricted adaptive search or access to the raw design is outside the theorem's information restriction.

A separate Newton run uses an absolute slope score tolerance of 1e-6 in all three cases. Every run stops at iteration 13. Its returned slopes are 263.426, 264.217 and 265.018. The slope in the genuinely finite case is 367.840 when solved to higher precision. These are two-parameter constructed examples with an exactly profiled intercept.

[General derivative-oracle lower bounds](https://arxiv.org/abs/1710.11606) address the cost of finding approximately stationary points over broad smooth function classes. This example concerns finite existence for the particular prediction-corrected logit objective at common labelled choices and a fixed predictor. The proof uses an inverse function argument on bounded menu contrasts.
