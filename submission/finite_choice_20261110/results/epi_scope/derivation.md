# Empirical-likelihood scope on the existing separated design

The source is Wang, Wen and Zou (2025), [arXiv v1](https://arxiv.org/html/2512.16363v1), equations (2), (4) and (5), Theorem 1, and Supplements S.1–S.2. The constraints retain a supervised score weighted by normalized nonnegative labelled weights. A positive empirical likelihood requires each weight to be strictly positive. This structure holds for the ideal, pooled-centred and cross-fitted auxiliary constructions in those equations. Their regular efficiency results have their stated assumptions.

Our inference below applies those constraints to the ordinary logistic supervised score. It is a consequence of classical positive-weight separation, not a new EPI theorem or an implementation of the authors' algorithm.

Write the supervised row score as $s_i(\beta)=x_i\{\sigma(x_i^T\beta)-y_i\}$. If a direction $d$ obeys $(2y_i-1)x_i^Td\geq0$ for all labelled rows, with at least one strict inequality, then

$$d^Ts_i(\beta)=-(2y_i-1)x_i^Td\,\sigma\{-(2y_i-1)x_i^T\beta\}\leq0.$$

For finite $\beta$, the inequality is strict on every row having a positive margin. Every strictly positive weighting therefore gives $d^T\sum_iw_i s_i(\beta)<0$. The supervised moment constraint cannot hold. Additional auxiliary constraints cannot restore feasibility. Thus complete or quasi-complete human separation excludes a positive-likelihood feasible point for this logistic-score specialization. Auxiliary constraints may introduce further obstructions; the converse is not asserted. A numerical return with zero likelihood is not a positive-likelihood solution or an identified regular estimator.

The current manuscript's Appendix B5 already uses labelled features $(2,4)$ and choices $(1,1)$, with auxiliary features $(1,1)$. Here both supervised scores, $-2/(1+e^{2\beta})$ and $-4/(1+e^{4\beta})$, are strictly negative for every finite coefficient. Even normalized weights allowing zero entries cannot make their sum zero. This is an obstruction independent of the auxiliary function selected for EPI.

With the existing base predictions $(0.9,0.9)$ and $(0.5,0.5)$ and correction power $15/88$, the compensated moment is $21/8$, while the attainable upper endpoint is $117/44$. Their gap is $3/88>0$. The gradient is negative at zero and positive at $\log 8$, with positive curvature, proving a unique finite root. Independent 80-digit arithmetic locates the root at about 2.0354709351 and checks its original-loss derivative.

At the previously retained path endpoint, the labelled predictions are $(25/44,25/44)$ and power is one. Its corrected moment is $79/44$, beyond the upper endpoint one; recession in direction one is $-35/44$. The compensated loss is unbounded below. The gradient at the zero reference remains $-57/44$ in both comparisons. These are the same design and endpoint already used for the complete-GMM comparison; no additional cohort is generated.

This audit separates the empirical likelihood constraints, the existing complete-GMM objective and the compensated loss. It establishes no ranking of forecast accuracy, interval coverage or general statistical efficiency. The sign argument and positive-weight separation are established tools, and the EPI source supplies a distinct existing method. The paper's specific proportional-dimensional boundary and matched-information consequences remain the contribution candidates. Formal PPI++ 2026 body and Appendices A–E still require direct comparison.
