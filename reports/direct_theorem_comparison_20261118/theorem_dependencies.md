# Which steps are inherited and which require calculation?

| Step | Authoritative source or current argument | Decision |
| --- | --- | --- |
| Compensated two-sample loss, scalar power and regular covariance | Complete supplied PPI++ v2, body and Appendices A–C; pinned code | Existing objective and criterion; no new estimator claim. |
| Finite moment support, recession and curvature rank | Existing convex support theory; current Proposition 1 | Diagnostic specialization, not a new existence principle. |
| Nearest row-category threshold | Koriyama–Bellec formal ICML 2025, Assumption 2.5, equation (4), Theorem 2.6, Lemma A.2 | Finite individual infima fail. A fixed-predictor pair proves that its category statistic cannot determine this compensated boundary after removing that assumption. No contradiction to the original theorem. |
| General Gaussian transition | Amelunxen et al. 2014, Theorem I and metric statistical dimension | Directly supplies the conditional kernel/cone transition. Explicitly inherited. |
| Gaussian absorption and relative interior | Götze et al. 2020, Theorem 3.1, Lemma 5.1, Corollary 5.6 | Existing. Corollary 5.6 directly supplies the relative-interior statement currently proved again in G3. |
| Gaussian random projections and constrained ordinary logit | Han–Ren examined v2, Theorems 1.1, 1.3 and 2.1 | General geometric and ordinary-logit results are established; no new general projection theorem. The pre-image theorem states nontrivial-cone hypotheses, so a zero target cone must not be inserted without checking its scope. |
| Current corrected interval cone | Appendix A1–A2, including its low-dimensional subspace constraint | Problem-specific representation, evaluated with inherited polar and projection identities. |
| Explicit proportional boundary | Appendix A3–A5 | Uniform support cancellation, multiplier and scale bounds, ordered extrema and conditional expectation limit require their stated checks. Candidate technical calculation. |
| Fixed-score/full-covariance examples, positive-power limit, probability-loss divergence | Existing Appendices B and D | Controlled consequences of the same objective, evaluated against their own prior margin and regularization results. Generic local/global distinctions are established. |
| Bounded known scale range | Separate verified bounded-scale supplement | Subordinate corollary using a classical inequality. No independent core innovation. |

## Direct geometric reduction

Let n=m+N. Conditional on the low-dimensional observations, A is a q×n independent standard Gaussian matrix and C is the fixed closed polyhedral cone specified in Appendix A. The random kernel is independent of C and has dimension n−q when q<n. The established kinematic theorem compares δ(C)+(n−q) with n. Thus a strict linear gap between δ(C) and q supplies the two probability limits. If q≥n the Gaussian kernel is zero almost surely. Neither the randomness of the original cone nor the two sample labels creates a new kinematic theorem: conditioning resolves that randomness.

The model-specific task is to calculate δ(C)/m. For the manuscript's cone, polar calculus expresses it as the conditional Gaussian expectation of an infimum over intercept/signal multipliers followed by a supremum over one shared cone scale. The scale is shared across all rows; allowing separate scales would change the cone. The observed prediction terms cancel in the multiplier support bound only under the same fixed joint population law for both samples. Positive labelled residuals bound the scale at each fixed power. These arguments localize both extrema before the uniform limit and conditional expectation step. This is the exact location of the candidate calculation beyond the general transition.

## Scope of the comparison

The direct application of general geometry is acknowledged rather than treated as a novelty obstacle that must be hidden. The row-category pair rules out a specific simpler reduction. It does not prove that no alternative response encoding, design-dependent reparameterization or broader theorem can produce an equivalent formula. The current A7 argument excludes only affine reallocations measurable with respect to the conditioned low-dimensional observations; it does not exclude arbitrary transformations. The literature screen remains bounded, and no global priority or external peer-review certification follows.

Primary sources: [formal Koriyama–Bellec record](https://proceedings.mlr.press/v267/koriyama25a.html), [Amelunxen et al.](https://arxiv.org/pdf/1303.6672), [Götze et al. v2](https://arxiv.org/pdf/1911.04184v2), [Han–Ren v2](https://arxiv.org/html/2212.05545v2).
