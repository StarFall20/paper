# Component clipping changes the corrected estimating equation

The inspected source is [run_gai_label.py, lines254–276](https://github.com/ChengLu1999/generative-augmented-inference/blob/d18717206867c7a12ff470879a0dd5d488b300c2/GAI_Conjoint_Study/label_z/run_gai_label.py#L254-L276). This note concerns that pinned implementation, not all implementations or results associated with the paper.

Let g be any probability vector, y a one-hot observed choice, w its label indicator, and 0<e≤1. The corrected target is τ=g+(w/e)(y−g). For a labelled record with chosen component k,

τ_k=1+(1/e−1)(1−g_k)≥1,

τ_j=(1−1/e)g_j≤0 for j≠k.

Component clipping to [0,1] gives y exactly; its row sum is one, so normalization leaves it unchanged. For an unlabelled record τ=g already belongs to the simplex. The resulting loss pools observed labels with unlabelled soft predictions. The inverse-propensity correction disappears from the labelled targets. This identity holds for every number of alternatives and every g in the simplex.

The source then evaluates its uncertainty formula with the unmodified corrected score. Convergence of the clipped loss does not guarantee that score vanishes. Our fixed-design binary audit holds g and every label allocation fixed and checks both gradients. It does not evaluate confidence-interval coverage or reproduce the complete article tables.

The identity does not imply that soft self-training always gives poor point estimates or that corrected inference is generally inconsistent. It identifies the objective being optimized. In the current benchmark, clipped soft training has smaller point error than the unpenalized corrected root; this result is retained alongside the score discrepancy and finite-root failures.
