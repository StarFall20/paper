# Release verification for Journal of Choice Modelling

19 November 2026

## Reproduction

Extract Reproduction_Materials.zip and follow its README. The archive includes a single verification command, with optional checks for the original RADIANT food workbook and Swissmetro source. The sources remain at their upstream locations. The runner checks their SHA-256 values before running the relevant verifiers. The top-level requirements now pin `openpyxl==3.1.5`, which pandas uses to read the RADIANT workbook.

The [release receipt](release_receipt.json) records 13 passing checks from a clean extraction with Python 3.12.14 and the pinned versions. The public-data checks reconstruct the original menus and recompute reported quantities from saved coefficients and allocations. The archive README links to the commands for refitting and the broader simulation. Repeated allocations describe the released menus and their fixed holdouts. A fresh virtual environment installed the dependencies from PyPI and passed 13 checks. Its 11 earlier result reports matched the original replay. [Portability check](portability.json)

## Attribution

[Reference metadata and fallback evidence](reference_metadata.json) records the title and leading-author audit for every distinct linked reference in the submitted Markdown. Automated registry or primary-site checks matched 59 links. Seven links required a primary-source lookup. The later supplement cites the original linear error bound used in Appendix C17 and the Newton-decrement guarantee discussed in Appendix B8. Its inequalities have fixed normals because the supervised recession is a finite maximum of linear functions. [Hoffman (1952)](https://nvlpubs.nist.gov/nistpubs/jres/049/4/V49.N04.A05.pdf)

## Priority check

The finite-sample research claim concerns three different outcomes for the same corrected choice criterion under matched local derivatives and fixed labelled choices, prediction rule, power and sample sizes. The supporting menu geometry draws on standard exponential-family support. The unit-weight limit uses known error-bound and successive-slack techniques and derives the gap scales from the logit criterion. [Geyer (2009)](https://www.stat.umn.edu/geyer/gdor/), [Cominetti and San Martín (1994)](https://doi.org/10.1007/BF01582220)

Song et al. connect prediction-powered inference with missing-data and survey methods. Their diagnostics cover sample alignment, prediction provenance, missingness and data-adaptive weights. Their application studies interval precision and coverage. The sections checked give methodological context for the present fixed-menu test of whether the corrected logit optimum exists. [Song et al. (2026), sections 2–4](https://hdsr.mitpress.mit.edu/pub/dwrc1xcp/release/2)

The [institutional coverage record](institutional_coverage.md) distinguishes verified access from authentication errors. The [Google Scholar log](google_scholar_search.md) records three queries, 34 distinct screened titles and follow-up comparisons with the closest sources.

The separate final PPI++ supplements returned webpages with no readable supplement content on the publisher download routes. The local v2 includes its appendices and has been compared. The release still needs institutional priority searching and an independent full-manuscript similarity assessment. [PPI++ v2](https://arxiv.org/pdf/2311.01453v2), [formal Supplement A identifier](https://doi.org/10.1214/26-AOAS2215SUPPA), [formal Supplement B identifier](https://doi.org/10.1214/26-AOAS2215SUPPB)

[Fixed-label empirical and source-level checks](../matched_auxiliary_diagnostics_20261119/README.md) accompany the manuscript version with Appendix B9.
