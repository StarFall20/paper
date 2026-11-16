# Primary sources and contribution scope

[Kessels, Jones and Goos (2019)](https://doi.org/10.1016/j.jocm.2018.12.002), Journal of Choice Modelling 31, 1–21, studies Firth estimation for small choice datasets and separation. The publisher metadata and abstract were inspected; the full article endpoint returned 403. This directly motivates a human-only Firth comparator. The present replay is not a replication of their experiments.

[Kosmidis and Firth (2021)](https://doi.org/10.1093/biomet/asaa052), Sections 1–2 and Corollary 1, provides the full-rank finite-estimate result and the conventional inverse-information Wald convention. The new implementation uses the established half-log-determinant penalty. This method is prior work; its finite output is not a contribution of the present paper.

The authors' [brglm2 source](https://github.com/ikosmidis/brglm2/blob/f1e21ee94292d75422e2141f6cc9344fd880a26e/R/brglmFit.R), commit f1e21ee94292d75422e2141f6cc9344fd880a26e, was read at the mean-adjustment and Jeffreys-adjustment functions and information-covariance documentation. In binary logit these give the same adjusted score. No external R code was copied or executed. The independent Python implementation checks the determinant objective and adjusted score separately.

[Sablica, Hornik and Rusch (2026)](https://arxiv.org/html/2610.08100v1), Theorem 1, Section 4.1 and Theorem 4, supplies categorical-likelihood overlap, LP detection and uniqueness on the estimable span. The full primary HTML is now accessible, resolving the preceding access gap. These results reinforce attribution of the generic diagnostic to prior work. The current manuscript's entry is genuine but uncited in its body; the prepared revision supplies its precise use.

[Koriyama and Bellec (2025)](https://arxiv.org/pdf/2501.03163), v3 Assumption 2.5 and Theorem 2.6, retains finite individual row infima and strict convexity. The corrected labelled rows fail those conditions. This establishes a bounded applicability distinction, not originality by assumption failure. The manuscript's explicitly versioned v2 comparison should be updated to the verified v3 numbering.

[Geyer (2009)](https://www.stat.umn.edu/geyer/gdor/), Electronic Journal of Statistics 3, 259–289, develops exponential-family recession and computational diagnosis. The author's paper page and linked primary draft were inspected. Classical support/recession geometry should receive an explicit source alongside the existing attribution to Albert and Anderson.

The 37-entry author-year screen finds 36 entries with candidate citations in the body or appendices and one bibliography-only entry, Sablica et al. This is a citation-connectivity screen, not a fresh metadata audit of every reference, independent proof review, Scopus/Scholar review or priority certificate.

The recent journal articles by [Hess et al. (2026)](https://doi.org/10.1016/j.jocm.2026.100611) and [Sfeir et al. (2026)](https://doi.org/10.1016/j.jocm.2026.100623) connect uncertainty reporting and assisted modelling to its readership. Their publisher abstracts support that topical connection; they do not establish the novelty of the compensated boundary. The directly relevant Firth comparison is the stronger change to the present evidence.
