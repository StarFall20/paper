# Closest-result audit and contribution decision, 18 November 2026

## The supplied PPI++ source is complete

The user-supplied 29-page March 2024 v2 contains the body and Appendices A–C. Its SHA-256 is 5c9dc26aa1a537785d34b1d690a7387cf21e0493f0a3f945905dc556d30e70bd. The [preceding page-bound map](https://github.com/StarFall20/paper/blob/b0782bb99ba36761145e6676690312f067b6bd21/reports/available_ppi_prior_20261118/source_and_claim_map.json) is the working comparison. The objective, GLM and multiclass formulation, fixed-dimensional consistency/coercivity argument, variance tuning, one-step alternative and odds-ratio inference are established there. The current candidate concerns the explicit proportional-dimensional existence boundary and its controlled consequences. Identity with the final published article remains unverified; this is a source-version limitation and does not stop substantive work.

## Further primary sources checked in this turn

**Yoshida, Takenouchi and Sugiyama (2021), “Lower-Bounded Proper Losses for Weakly Supervised Classification.”** Read the introduction, §5.1 Example 12, §5.2 Theorem 13 and Proposition 14, §5.3 Corollary 15, and Supplement C.5's conjugate proof. Backward-corrected cross-entropy can already be unbounded below; lower boundedness has a convex-conjugate characterization. Their generalized logit squeezing changes the potential and its link. It is not automatically the same coefficient target as the present unmodified compensated logit. Generic unboundedness and conjugate geometry are established. Sources: [official body](https://proceedings.mlr.press/v139/yoshida21a/yoshida21a.pdf), [official supplement](https://proceedings.mlr.press/v139/yoshida21a/yoshida21a-supp.pdf).

**Taheri, Pedarsani and Thrampoulidis (2021), “Sharp Guarantees and Optimal Performance for Inference in Binary and Gaussian-Mixture Models.”** Theorem 1 assumes bounded minimizers; Remark 3 discusses boundedness. Section 4.2 examines separation for ordinary nonnegative vanishing-margin losses. These inspected results do not directly supply the present two-sample compensated boundary. This is a scoped comparison, not a claim that the entire literature omits it. [Official article](https://www.mdpi.com/1099-4300/23/2/178).

**Koriyama and Bellec (2025), “Phase transitions for the existence of unregularized M-estimators in single index models.”** Assumption 2.5 and the surrounding threshold theorem require strict row convexity and integrability of the rowwise infimum. Those requirements need checking for compensated rows that can have infimum minus infinity or become affine. General convex-loss phase-transition machinery is established; naming a compensated loss does not itself prove an extension. The manuscript already attributes this work. [Examined arXiv v3](https://arxiv.org/html/2501.03163v3).

**Bhatia and Davis (2000), “A Better Bound on the Variance.”** Publisher metadata confirms American Mathematical Monthly 107(4), 353–357, DOI 10.1080/00029890.2000.12005203. Original full text was not read in this turn. The bounded-variable inequality is established, and [proof.md](proof.md) supplies every algebraic step used here. The new endpoint application must not be described as a new inequality. [Publisher metadata](https://www.tandfonline.com/doi/full/10.1080/00029890.2000.12005203).

## Decision on independent contribution

There is one scientific object: the existing unpenalized compensated choice-logit objective. Its proportionally growing dimension, two finite sample blocks, predictor geometry and common menu coefficients lead to the candidate explicit boundary. Matched local information, positive powers tending to zero and auxiliary-budget consequences are tests of that boundary's meaning. The bounded-scale calculation strengthens one consequence. It is an immediate corollary of that boundary and a classical moment bound; it cannot carry a separate independent-innovation claim.

Any generic corrected-loss unboundedness claim would be redundant with existing weak-supervision work. The current claim must say what the explicit calculation adds beyond established separation, convex conjugates and conic methods. A proposed citation paragraph is included with the appendix text. The added bounded-scale result removes one dependence on extreme scale contrasts. It does not establish an estimable scale specification, a nonzero-signal or panel result, or an improved prediction method.

The released food DCE and official-package diagnostics remain operational evidence. Small-label failures are visible; at larger label budgets finite fits are common, ordinary human-label MNL has lower held-out loss, and descriptive price equivalents are close. These adverse findings limit a claim of broad practical superiority. No evidence is discarded to make the new corollary look stronger.

The material is more defensible after this check, but sufficient JCM novelty and substantive choice-analysis value remain unresolved. A complete theorem-equivalence comparison against the closest general phase-transition results and final author review are still required. Repeating the same simulations cannot establish priority. No internal review is represented as an independent review or as proof of meeting the journal's threshold.

## Search scope

[search_log.json](search_log.json) records 22 ordinary web queries in this turn, primary units actually read, and metadata-only limitations. There were zero Scopus, Google Scholar or Web of Science queries in this turn. Results do not constitute a systematic database review, a global non-overlap guarantee, or a plagiarism scan. No original article PDF is redistributed.
