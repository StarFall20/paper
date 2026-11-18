# Proposed text for the existing article

This text is not yet integrated into the reviewed Word release. It stays subordinate to the existing main boundary and uses its assumptions.

## Suggested addition to Appendix G

### Bounded scale ranges

The auxiliary-budget reversal can occur with a bounded scale contrast. Retain the balanced model, the calibrated proxy with error e, an independent known scale D and the population rule λτ=τ/(1+τ). Suppose 0<a≤D≤b and write r=b/a. The classical bounded-variable variance inequality gives

$$\frac{(ED)^2}{ED^2}\geq\frac{4r}{(1+r)^2}.$$

Indeed, with μ=ED, the inequality (D−a)(D−b)≤0 implies ED²≤(a+b)μ−ab, and

$$\mu^2-\frac{4ab}{(a+b)^2}\{(a+b)\mu-ab\}=\left(\mu-\frac{2ab}{a+b}\right)^2.$$

Equality holds for the endpoint law with P(D=b)=a/(a+b). The inequality is established prior work (Bhatia and Davis, 2000).

Combining this bound with (G.14), a strict decline between the two iterated auxiliary endpoints is possible among scale laws of support ratio at most r exactly when

$$r>r_*(e):=\frac{1+\sqrt{1-\pi e(1-e)}}{1-\sqrt{1-\pi e(1-e)}}.$$

The equality law proves sufficiency, and the moment bound proves necessity. The proportional sample limit is taken at each fixed law and auxiliary ratio first. This criterion characterizes the endpoint mechanism. Finite-ratio boundaries can dip below both endpoints, so r≤r*(e) does not provide a finite-budget safety guarantee.

For a finite-ratio illustration, take D=1/2 with probability 4/5 and D=2 otherwise. Then ED²=1 and (ED)²=16/25. At proxy error e=2/5, the boundaries at τ=1/40 and τ=1 are enclosed around 0.4855662367 and 0.3982976460. The dimension ratio κ=0.46 lies strictly between them. Along the variance rule the first corrected objective has a finite estimate with probability tending to one, and the second has no finite estimate with probability tending to one. Ordinary human separation remains unchanged under positive whole-row scaling, with threshold 1/2. The regular covariance factor falls from approximately 3.9961 to 3.92, a small reduction. The boundary subsequently increases; no monotone deterioration is claimed. These are Gaussian, zero-signal model results, not a population failure-rate estimate.

## Suggested related-work paragraph

Backward-corrected classification losses can be unbounded below, and existing lower-bounded proper-loss theory gives convex-conjugate criteria for that property (Yoshida et al., 2021). General high-dimensional convex-risk theory also studies when minimizers exist, with explicit conditions on row losses and their infima (Taheri et al., 2021; Koriyama and Bellec, 2025). Our object is the existing two-sample compensated logit criterion. The proposed result calculates its proportional-dimensional existence boundary and separates that boundary from the regular covariance criterion used to select power. Generic unboundedness and the underlying convex geometry are established inputs.

## References to add or check

Bhatia, R., Davis, C., 2000. A better bound on the variance. American Mathematical Monthly 107(4), 353–357. https://doi.org/10.1080/00029890.2000.12005203

Yoshida, S. M., Takenouchi, T., Sugiyama, M., 2021. Lower-bounded proper losses for weakly supervised classification. Proceedings of Machine Learning Research 139, 12110–12120. https://proceedings.mlr.press/v139/yoshida21a.html

Taheri, H., Pedarsani, R., Thrampoulidis, C., 2021. Sharp guarantees and optimal performance for inference in binary and Gaussian-mixture models. Entropy 23(2), 178. https://doi.org/10.3390/e23020178
