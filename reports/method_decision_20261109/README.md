# Reproducible DCE design audit and research decision, 2026-11-09

Read 最终研究去留报告.md, classical_equivalence_proof.md and survey_and_sample_decision.json. The current independent-new-method claim is not supported. The output is a completed classical-method diagnostic resource; it is not a submission-ready new theory or deployable new questionnaire.

## Reproduce

Use Python 3.12 and requirements.txt. Official public inputs are downloaded to a separate directory. No DH36 raw exports are included or needed to run the public design audit.

```sh
python fetch_published_dce_inputs.py --out-dir INPUTS
python published_dce_decision_audit.py --route-data INPUTS/apollo_swissRouteChoiceData.csv --vehicle-data INPUTS/vehicle_data.txt --out RESULTS --replications 500 --draws 499
python classical_kl_scope_audit.py --route-data INPUTS/apollo_swissRouteChoiceData.csv --vehicle-data INPUTS/vehicle_data.txt --out RESULTS
python design_support_audit.py --input teaching_example_design.json --out RESULTS/teaching_example_audit.json
```

The final decision report and bibliography are curated artifacts. The report-generation helper reads their frozen decision context and analysis outputs; run it against this complete result directory if rebuilding its figures. The figures compare different null representations explicitly and do not imply they are interchangeable.

All 9,000 simulation records are computer generated and each has one response. They are not respondents and are not field sample planning. Diagnostic probe feasibility is limited to the declared ticks and marginal bounds. Candidate representations and nominal weights are audit choices, not hypotheses attributed to the source authors. No original choice outcomes enter a new behavioral rejection claim.

## Generic support audit

Each JSON menu contains all alternatives, a declared candidate vector for each, and a nonnegative design weight. The representation is either absolute_menu_summary or utility_difference_vector (all non-reference candidate differences). Numeric values are compared as exact rational numbers. Repeated identical raw menus are merged; zero-mass support is omitted. A raw menu cannot have conflicting candidate values. Positive structural rank is not evidence that the representation is sufficient or that a statistical test has power.

## Evidence limits

The two data archives and their exact checksums are in published_data_sources.json. The release audits the displayed catalogue, not every original design-generation constraint. Scopus/Scholar access was unavailable as recorded in search_and_access_log.json. No exhaustive literature or global priority guarantee is made. Previous internal submission-threshold statements are archived judgments and are superseded by this decision.

Private DH36 raw snapshots and their exploratory analysis remain preserved; only their hashes are referenced. There are no new human records, contact details or private response vectors in this package.

SVG figures are the canonical figures in the repository. Optional PNG previews are included in the local ZIP but omitted from the canonical MANIFEST.sha256 and GitHub release directory; the same plotting helper recreates both formats.
