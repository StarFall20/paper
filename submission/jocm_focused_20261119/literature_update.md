# Result-level literature check

Date 19 November 2026

## The contribution to assess

Classical exponential-family support yields the fixed-design membership criterion. Standard linear programming implements it. The contribution concerns what changing the auxiliary design does to prediction-corrected choice likelihoods. With the labelled choices, predictor, weight and sample sizes fixed, matching local derivatives through any preassigned finite order still permits a unique finite optimum, an unattained finite infimum or unbounded decrease. The proportional-dimensional results identify binary and whole-menu thresholds with a finite auxiliary ratio. The held-out analysis connects the approach to the weight boundary with the loss on actual choice menus.

## Closest results rechecked

| Source and section | What it establishes | Implication for the manuscript |
| --- | --- | --- |
| Angelopoulos, Duchi and Zrnic, PPI++ v2, §3, Corollary 1 and appendices | The corrected GLM loss, convexity for weights in [0,1], and regular asymptotic inference | Use the same correction and attribute the criterion. The finite global minimum requires a separate support calculation. |
| Vadlamani et al., Transfer Calibrated Prediction Powered Inference v1, §5 and Appendix F | Quasi-separation in low-label Bradley–Terry fits and ill-conditioning of the estimated correction weight | Credit the direct observation of prediction-assisted separation. The present claim concerns the three finite-design outcomes and the matched local information result. |
| Chen, Bai and Zhu, Covariance-Calibrated Prediction-Powered Inference v2, §3 and Supplement S2.10 | The same regular limit can coexist with different conditional linear risks. The structured logistic risk result conditions on a finite pilot. | Cite this precedent for differences hidden by regular asymptotics. Here the comparison uses the same corrected logit objective and changes attainability through the auxiliary menus. |
| Xu and Sun, Filtered Prediction-Powered Inference v1, §3 and Theorem 7 | A margin condition controls recovery of a region in which predictions improve efficiency | The margin concerns predictor and regression functions relative to a population reference. The choice-likelihood boundary concerns the attainable empirical feature moment. |
| Fisch et al., Stratified Prediction-Powered Inference | Stratum-specific corrections and label allocation exploit heterogeneity for more efficient inference | Credit stratification for controlling differences between strata. The food example changes within-menu model moments after matching empirical menu frequencies. |
| Kessels, Jones and Goos, JCM 2019, and Sfeir et al., JCM 2026 | The journal has published both finite-estimate remedies for separated human choice data and experiments using LLMs in choice modelling | The manuscript addresses a choice estimation problem under the journal's existing methodological scope. The supplement retains the human-only Firth comparison. |

- [PPI++ v2 and its embedded appendices](https://arxiv.org/pdf/2311.01453v2) · [formal article DOI](https://doi.org/10.1214/26-AOAS2215)
- [Vadlamani et al., §5](https://arxiv.org/html/2609.34156v1)
- [Chen et al., v2](https://arxiv.org/pdf/2610.21837)
- [Xu and Sun, v1](https://arxiv.org/pdf/2602.10464)
- [Fisch et al.](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c9fcd02e6445c7dfbad6986abee53d0d-Abstract-Conference.html)
- [Kessels et al.](https://doi.org/10.1016/j.jocm.2018.12.002) · [Sfeir et al.](https://www.sciencedirect.com/science/article/pii/S1755534526000291) · [journal scope](https://shop.elsevier.com/journals/journal-of-choice-modelling/1755-5345)

## Coverage

Scopus showed a homepage after the reported login. Two following attempts to attach to the search tab timed out. This session contributes zero verified Scopus result sets. The broader [institutional literature audit](../../reports/institutional_literature_audit_20261113/closest_prior_matrix.md) records earlier Web of Science and Google Scholar checks. This follow-up used public searches combining prediction-powered inference with logistic separation, recession, finite minimizers, and matching local diagnostics, followed by section-level comparisons of primary sources.

The PPI++ v2 main text and embedded appendices are covered. The separate final supplements identified by [SUPPA](https://doi.org/10.1214/26-AOAS2215SUPPA) and [SUPPB](https://doi.org/10.1214/26-AOAS2215SUPPB) still need a final-version comparison. The named works do not establish the three-way matched-derivative result in the passages checked. This is a bounded priority comparison. An independent proof review and formal text-similarity review remain open.

## Follow-up on multiple diagnostic points

[General derivative-oracle lower bounds](https://arxiv.org/abs/1710.11606) concern approximate stationarity over broad smooth function classes. The Appendix B7 result uses equal-weight rows from the specified prediction-corrected logit objective. It controls finite attainment via the mean absolute attribute contrast while matching the selected local diagnostics. The general statement allows preassigned checkpoints. The numerical example also matches diagnostics at the jointly determined first Newton update.

The publisher's direct [Supplement A PDF](https://projecteuclid.org/journals/supplementalcontent/10.1214/26-AOAS2215/aoas2215suppa.pdf) and [Supplement B ZIP](https://projecteuclid.org/journals/supplementalcontent/10.1214/26-AOAS2215/aoas2215suppb.zip) remained inaccessible on retry. [The author's bibliography](https://web.stanford.edu/~jduchi/publications.html) gives the journal citation and links to the arXiv file.

A local exact-phrase comparison against the user-supplied v2 file, including its embedded appendices, found zero contiguous matches of 10 words or longer in the scanned prose of the article and supplement. The comparison excluded the manuscript's references and formulae. It covers one source and exact wording.

## Nested budget and recent literature follow-up

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


## Priority at the unit endpoint

[Cominetti and San Martín](https://www.ceremade.dauphine.fr/~carlier/Cominetti-SanMartin.pdf) derive a limiting optimal-face center for the exponential penalty of a linear program. The present unit-weight path first minimizes supervised recession and then uses both human utility gaps and auxiliary utility gaps shifted by one. The gap scales arise from the corrected choice objective. The proposition includes an explicit held-out choice-loss rate. [Weighted supervised logistic existence](https://doi.org/10.1080/03610926.2016.1260742) is prior for ordinary case weighting; the publisher's abstract was screened. [Liu et al.](https://link.springer.com/article/10.1007/s11424-026-5075-1) compare predictive performance for model specification in a prediction-powered setting. Their publisher's abstract concerns testing fit to the data-generating process.

The browser connection to Scopus timed out. The public result-level comparisons above leave formal institutional priority searching and the final PPI++ supplement check open.


## Release verification and current database coverage

The release checks the identity of 67 distinct linked sources. Appendix C17 now cites [Hoffman (1952)](https://nvlpubs.nist.gov/nistpubs/jres/049/4/V49.N04.A05.pdf) for its linear error bound.

[Song et al. (2026), Sections 2–4](https://hdsr.mitpress.mit.edu/pub/dwrc1xcp/release/2) review prediction-powered diagnostics for sample alignment, missingness, prediction provenance and data-adaptive weights. The fixed-menu calculation in this manuscript addresses the existence of the corrected logit optimum. The review's empirical example evaluates efficiency and coverage.

The logged-in Scopus page displayed Scopus Preview and offered author and source searches. The institution flow returned to the university login. The library advertises Web of Science SCIE and SSCI coverage from 2010. Its off-campus link proceeded through school authentication, then Web of Science returned institution error 90013. This session contributes no document result set from Scopus or Web of Science. [University database list](https://lib.sdufe.edu.cn/wxzy/dzzy.htm), [university Web of Science entry](https://lib.sdufe.edu.cn/info/1162/1383.htm)

Google Scholar returned about 106 results for the broad query from 2023 onward. The first 30 were screened by title and available excerpts. A narrower query returned five, all screened. A query targeting matched derivatives returned zero. Exact strings and screened titles are in the [search log](../../reports/submission_release_audit_20261119/google_scholar_search.md). The most direct overlap remains the observed quasi-separation in a prediction-assisted Bradley–Terry application. A working paper on synthetic respondents also addresses quasi-separation in fitting synthetic preferences. In the available primary material, neither gives the fixed-power comparison of a finite optimum, an unattained finite infimum and unbounded decrease under matched derivatives. [Vadlamani et al., Section 5](https://arxiv.org/html/2609.34156v1), [Leng working paper record](https://ssrn.com/abstract=6835019)

The final PPI++ supplement links returned web pages without readable supplement content. The supplied v2 and its embedded appendices were reviewed, including exact phrase screening of the manuscript prose. The screen found zero matching ten-word windows. Formal similarity review and review of the separately published supporting files remain open.


## Bounds from local diagnostics

[Bach (2010), Proposition 2](https://www.di.ens.fr/~fbach/bach_ejs_self_concordance.pdf) gives a bounded-feature sufficient condition for a unique global minimum from the Newton decrement. Appendix B8 gives an exact range of compatible limits in the symmetric binary design. The finite number of equal-weight contrast pairs is part of the information.

[Leng, Sang and Agarwal, revised SSRN abstract](https://ssrn.com/abstract=4802019) optimize mixture weights over alternative LLM elicitation outputs to match an aggregate human benchmark. Their guarantee concerns calibration loss within the feasible mixtures. The directly relevant synthetic MNL passage in Leng's separate local-validity paper remains available as a Scholar excerpt. The publisher and author's publication page did not provide readable full text in this session.


# Result-level comparison for the empirical and theoretical claims

19 November 2026

| Source | What it establishes | Relation to this manuscript |
| --- | --- | --- |
| Angelopoulos, Duchi and Zrnic, PPI++ v2, Section 3 and Appendix A.2 | Corrected loss, regular inference and conditions for consistency. The fixed-dimensional consistency proof bounds the objective on a sphere around the target. | The manuscript cites the loss and the regular results. The finite-design calculation classifies the corrected target for the observed auxiliary menus and compares what local derivatives reveal. The separately published appendix and code downloads returned web responses. [v2 with embedded appendices](https://arxiv.org/pdf/2311.01453v2), [formal Supplement A](https://doi.org/10.1214/26-AOAS2215SUPPA), [formal Supplement B](https://doi.org/10.1214/26-AOAS2215SUPPB) |
| Vadlamani et al., Transfer Calibrated Prediction Powered Inference, Section 5 and Appendix F | Reports quasi-separation and unstable power estimates in the 19-parameter Chatbot Arena fit at low label budgets. | This is a direct observation of the practical symptom and is cited as such. The fixed-label comparison here changes only the auxiliary choice menus, and the mathematical constructions match local diagnostics while producing three distinct outcomes. [Author full text](https://arxiv.org/html/2609.34156v1) |
| Ao et al., PPI-SVRG, Equation (3), Theorems 5.2 and 5.5 | Develops control-variate methods for stochastic optimization. The bounds use objective error relative to an empirical-loss minimizer. | The estimator and its optimization method are different questions. The comparison in Appendix B9 checks attainment of the specified corrected logit objective before an optimizer is chosen. [Author full text](https://arxiv.org/pdf/2601.21470) |
| Nguyen, Attainment Boundaries and Escape Rates in Asymptotically Conic Optimization, Assumption 4.1 and Theorem 4.3 | Gives directional selection and power-law escape near finite nonattainment for a fixed epigraph under an inward linear tilt. | The general idea of approaching a nonattainment boundary has precedent. The choice-model calculation uses exponentially decaying menu probabilities and derives the logarithmic rate and the coefficient on held-out log loss from the menu utility gaps. [Author full text](https://arxiv.org/pdf/2608.10341v1) |
| Bach, Self-Concordant Analysis for Logistic Regression, Proposition 2 | A sufficiently small scaled Newton decrement guarantees a global minimizer under a bound on the third derivative. | Appendix B8 uses a known paired design to give exact compatible limits. The descriptive ranking of raw Newton decrements in Appendix B9 concerns a different use of that statistic, outside the claim made by the sufficient condition. [Published article](https://www.di.ens.fr/~fbach/bach_ejs_self_concordance.pdf) |

The contribution should be described as a result about a specific corrected choice likelihood. The classical support classification and general idea of a boundary at which attainment fails are established. The evidence reviewed here supports the narrower claim about matched local diagnostics, the effect of auxiliary menu composition on a fixed labelled design, and the rates for actual forecasts under the stated assumptions. The source-specific text screen reports contiguous matches within the downloaded texts. Broader phrase, paraphrase and priority assessment requires further coverage.

The local prose screen compared both manuscript files with 8 named source PDFs and found 0 exact matches of ten or more consecutive normalized words. The results identify the versions and exclude candidate equations, references and tables. A broader similarity assessment remains open.


November 19 follow-up. Web of Science displayed institutional login error 90013. The coverage count uses the nine archived institutional searches. The November 6 version of Chen, Bai and Zhu conditions its grouped logistic risk theorem on the existence of the pilot, as specified in Theorem 4.2. The author-linked PPI++ research commit was independently retrieved and executed. Subscription Scopus and the formally deposited PPI++ appendices remain outstanding. [Covariance-calibrated PPI, version 2](https://arxiv.org/html/2610.21837v2#S4.SS2)


Appendix H gives the correlated two-task benchmark, the exact ordinary-logit comparison and an independent check of all 2280 corrected outcomes. The November 20 institutional report records eight Web of Science Topic queries, 71 exported records and cross-checks in EconLit, EBSCO Business and Google Scholar.

The exact text screen compared 14 named primary PDFs, covering 611 pages. No contiguous matching passage of ten or more words was detected under the recorded extraction rules.

See the November 20 author review report for explicit scope and result-level comparisons.
