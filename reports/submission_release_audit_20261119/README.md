# Release verification for Journal of Choice Modelling

19 November 2026

## Reproduction

Extract Reproduction_Materials.zip and follow its README. The archive includes a single verification command, with optional checks for the original RADIANT food workbook and Swissmetro source. The sources remain at their upstream locations. The runner checks their SHA-256 values before running the relevant verifiers. The top-level requirements now pin `openpyxl==3.1.5`, which pandas uses to read the RADIANT workbook.

The [release receipt](release_receipt.json) records 14 passing checks from a clean extraction with Python 3.12.14 and the pinned versions. The public-data checks reconstruct the original menus and recompute reported quantities from saved coefficients and allocations. The archive README links to the commands for refitting and the broader simulation. Repeated allocations describe the released menus and their fixed holdouts. A fresh virtual environment installed the dependencies from PyPI and passed 14 checks. Its 11 earlier result reports matched the original replay. [Portability check](portability.json)

## Attribution

[Reference metadata and fallback evidence](reference_metadata.json) records the title and leading-author audit for every distinct linked reference in the submitted Markdown. Automated registry or primary-site checks matched 59 links. Seven links required a primary-source lookup. The later supplement cites the original linear error bound used in Appendix C17 and the Newton-decrement guarantee discussed in Appendix B8. Its inequalities have fixed normals because the supervised recession is a finite maximum of linear functions. [Hoffman (1952)](https://nvlpubs.nist.gov/nistpubs/jres/049/4/V49.N04.A05.pdf)

## Priority check

The finite-sample research claim concerns three different outcomes for the same corrected choice criterion under matched local derivatives and fixed labelled choices, prediction rule, power and sample sizes. The supporting menu geometry draws on standard exponential-family support. The unit-weight limit uses known error-bound and successive-slack techniques and derives the gap scales from the logit criterion. [Geyer (2009)](https://www.stat.umn.edu/geyer/gdor/), [Cominetti and San Martín (1994)](https://doi.org/10.1007/BF01582220)

Song et al. connect prediction-powered inference with missing-data and survey methods. Their diagnostics cover sample alignment, prediction provenance, missingness and data-adaptive weights. Their application studies interval precision and coverage. The sections checked give methodological context for the present fixed-menu test of whether the corrected logit optimum exists. [Song et al. (2026), sections 2–4](https://hdsr.mitpress.mit.edu/pub/dwrc1xcp/release/2)

The [institutional coverage record](institutional_coverage.md) distinguishes verified access from authentication errors. The [Google Scholar log](google_scholar_search.md) records three queries, 34 distinct screened titles and follow-up comparisons with the closest sources.

The separate final PPI++ supplements returned webpages with no readable supplement content on the publisher download routes. The local v2 includes its appendices and has been compared. The release still needs institutional priority searching and an independent full-manuscript similarity assessment. [PPI++ v2](https://arxiv.org/pdf/2311.01453v2), [formal Supplement A identifier](https://doi.org/10.1214/26-AOAS2215SUPPA), [formal Supplement B identifier](https://doi.org/10.1214/26-AOAS2215SUPPB)

[Fixed-label empirical and source-level checks](../matched_auxiliary_diagnostics_20261119/README.md) accompany the manuscript version with Appendix B9.

[Author branch replay](../author_ppi_branch_audit_20261119/README.md) verifies the stored call records as part of the release.

Appendix H gives the correlated two-task benchmark, the exact ordinary-logit comparison and an independent check of all 2280 corrected outcomes. The November 20 institutional report records eight Web of Science Topic queries, 71 exported records and cross-checks in EconLit, EBSCO Business and Google Scholar.
The exact text screen compared 14 named primary PDFs, covering 611 pages. No contiguous matching passage of ten or more words was detected under the recorded extraction rules.

Section 4.4 and Supplementary Appendix I compare four estimators on fixed public choices and matched probabilities. The code recomputes 8,000 fits in the extracted release, verifies 7,993 finite scores and checks the 32 fixed-menu limits through primal feasibility and dual supporting directions. The clean archive passed 17 release checks. Revised prose was bound to those outputs through a final inventory and headline check.

Section 3.2 and Supplementary Appendix H6 extend the paired task result to a fixed weight below one. Both samples retain positive log partition weights. The current extraction reconstructs all 400 new fit configurations and rechecks the archived headline results. Fifteen earlier checks retain byte-identical evidence.

Appendix I4 compares the half-limit correction, fixed multipliers and all-soft criteria, with all fallbacks retained. The 24,000 configurations use row folds and released five-row block folds. The current release reruns 4,000 residual limits and independently recomputes all 48 summaries.

Section 3.4 and Appendix J verify the diagnostic limitation while holding the entire discrete profile support and all sample sizes fixed. Changing only the auxiliary frequencies at weight 0.5 produces all three existence outcomes at identical local derivatives through order five.

Appendix J verifies an exact three-state frequency comparison and checks the same mechanism on the public RADIANT menus. A fixed auxiliary forecast for each of 34 menus leaves 70 finite-versus-unbounded pairs at unit weight and none at weight 0.5.

Appendix J verifies per-allocation correction-weight limits for 384 previously used food allocations. With all auxiliary menus and their forecasts fixed, 12 pairs span a finite fit and an unbounded objective at weight 0.75, with ordinary labelled Logit finite.
