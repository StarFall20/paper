# Finite estimation and label budgets in related work

The search combined prediction-powered inference with logistic and multinomial logit, finite minimum, recession, quasi-separation, auxiliary features, and sample size. It also searched Journal of Choice Modelling for separation, mode choice with unbalanced classes, and choice-experiment sample size. The result-level comparison covers primary articles and author manuscripts.

| Paper | Relevant result | Implication for this analysis |
| --- | --- | --- |
| Angelopoulos, Duchi and Zrnic, PPI++ | The corrected loss is convex for GLMs at powers in [0,1]. The stated regular inference uses nonsingular limiting information. | The audit uses this objective, with a fixed estimated predictor. Finite attainment is checked from the observed support. |
| Mani, Xu, Lipton and Oberst, ICML 2026 | Proposition 4.1 gives exact variance for the PPI++ mean under joint Gaussianity. Related sections study finite-sample estimation error and correlations needed for risk improvement. | A generic claim of finite-sample harm is already established. The choice claim is specifically about existence of a minimum and subsequent predictive loss at a given design and power. |
| Vadlamani et al., 2026, Section 5 and Appendix F | Quasi-separation and unstable power estimation appear at low budgets for predicted pairwise comparisons. | This is the direct prior observation of the symptom. The manuscript evaluates general menu support and tests when the observed failure disappears in nested label draws. |
| Assele, Meulders and Vandebroek, JCM 2023 | The authors build sample-size guidance from choice-design features under mixed multinomial and multinomial logit. | It establishes the journal's practical interest in label requirements. The present arm tracks finite-solution and prediction changes in the same released menus with correction. The observed crossovers are design-specific. |
| Kessels, Jones and Goos, JCM 2019 | Firth bias reduction supplies finite estimates for separated choice data. | The paper has a supervised separation comparator in its supplement. The new arm uses a fixed ridge fallback and accounts for the predictor's 600 training labels in the direct MNL comparison. |

- [PPI++ v2 and embedded appendices](https://arxiv.org/pdf/2311.01453v2)
- [Mani et al. in PMLR](https://proceedings.mlr.press/v306/mani26b.html) · [accessible author full text](https://arxiv.org/pdf/2505.20178)
- [Vadlamani et al.](https://arxiv.org/html/2609.34156v1)
- [Assele et al. in JCM](https://www.sciencedirect.com/science/article/pii/S1755534523000374)
- [Kessels et al. in JCM](https://www.sciencedirect.com/science/article/pii/S1755534518300836)

Database coverage on 19 November 2026. The school browser inventory and direct Scopus reconnect both timed out this round. Direct public Google Scholar query links returned unavailable in the browsing service. The official PPI++ Supplement A URL was rechecked and yielded no readable text. This round adds the public primary-source comparisons above. The institution-based search and official-supplement comparison remain open.

[Sablica, Hornik and Rusch, v2](https://arxiv.org/html/2610.08100v2), Theorem 1 and Section 3.2, provide an overlap criterion for a product of observed-category likelihood factors, including multinomial logit. The corrected choice target in this study includes the difference of predicted moments across two samples. Its position can be exterior even when the supervised likelihood has overlap. This three-way support comparison and the matched-checkpoint result concern the corrected criterion.

[Fuh, Kao and Pang](https://link.springer.com/article/10.1007/s00362-026-01902-2) study positivity constraints on the parameters of multinomial choice models. The publisher's abstract identifies an additional monotonicity condition. This source was screened from its abstract. Its scope differs from unconstrained choice-coefficient estimation with a prediction correction.
