# Public choice records under increasing auxiliary budgets

This fixed-plan audit keeps human labels, a finite ridge pilot and nuisance predictions fixed within each allocation while expanding a nested auxiliary pool. It tests the existing auxiliary-only compensated objective and source-style power rule. It adds no estimator or theory.

The [complete findings](progress_zh.md) retain all 500 source-convention permutations, 20/50 label budgets, two prediction routines, six auxiliary budgets and three power paths. There are 36000 method records and 25792 actual support calls after within-cell duplicate-power elimination. Objectives can repeat across cells. The complete record file uses lossless gzip compression; readers accept both the original CSV and its compressed form. These are repeated allocations on one released file, not new respondents. Every auxiliary human label is masked before fitting or objective evaluation.

The retuned 20-label finite counts increase from 156 to 161 (AI aided) and 142 to 150 (attributes only) between the first and last pools. Four AI-aided and three attributes-only allocations lose attainment: one shared allocation becomes exterior, while three AI-aided and two attributes-only records select zero power on separated human designs. These seven method records involve six distinct allocations. At 50 labels, retuned and fixed-first-N powers are finite in every allocation at every pool size. This is evidence against a frequent practical availability loss on this file. Unit-power comparisons and all boundaries remain.

## A rare verified case

The [case](case/exact_case.json) was selected after the full primary study using the smallest trial among first-to-last finite-to-exterior transitions. Both affected prediction routines are retained. It is one existing allocation, not two independent examples. The before-pool roots have direct numerical score and information checks. For the enlarged pool, integer attributes and stored dyadic predictions permit exact rational recession signs: human recession is zero and the compensation term is negative. Every positive power is then unbounded below. The zero-power human fit also has separation. The smaller uniform prediction-perturbation radius is about 0.002058.

These exact signs concern the frozen numerical predictions. Before-pool roots are numerical checks. The [secondary case plan](case/secondary_case_plan.json), [frozen predicted probabilities](case/frozen_case_predictions.json) and [independent exact check](verification/exact_case_verification.json) retain scope and timing. No raw human response vector is shipped. Reproduction fetches the pinned source numeric input separately.

## What was verified

The [verification](verification/verification.json) reconstructs all 36000 records, checks 1000 human-design moment programs, 625 preselected positive-power interval programs, 3602 method-level original negative directions and all 120 paired transition rows. Directions can repeat across methods. The independent influence calculation agrees with source powers within 8e-15. This is not a second complete primary support run.

The PSD score trace is an estimated diagnostic at a fixed ridge pilot. It is not the final estimator's true variance, prediction risk or confidence interval. The separate pooled tuning quadratic is the source-style power surrogate, not the published PPI++ covariance estimator: its matrix is non-PSD in 84 records even though every reported trace is nonnegative. No intervals were issued in this study.

The [figure](figures/public_auxiliary_budget.png) shows all 72 prespecified cells. Error bars are 95% Wilson Monte Carlo intervals conditional on the file. Budgets and methods share allocations; connecting lines only join measured points. The [vector PDF](figures/public_auxiliary_budget.pdf) has embedded fonts. All findings, including improved aggregate availability and the rare reverse event, are retained.

## Reproduce

Install the exact scientific versions in requirements.txt with Python 3.12. From this package directory:

```sh
python verify_package.py
python analysis/fetch_public_conjoint_audit.py --out local_inputs
python run_reproduction.py --input-dir local_inputs --out regenerated
```

The fetcher retrieves pinned numeric files and their original license/README. Downloaded research code is not executed. The default reconstructs all records and saved directions, performs the planned independent programs and verifies the exact case. To regenerate the full primary study and all later checks, use a new output directory:

```sh
python run_reproduction.py --input-dir local_inputs --out regenerated_full --full
```

The full command is provided; the release validation reran the default independent workflow, not a second full primary study. Source-convention five-row blocks lack verified respondent IDs. The file's recruitment and vaccine-choice exclusions retain their source limitations.

## Contribution assessment

The [parent-result reading](prior_equivalence.md) and [actual search scope](literature_scope.json) narrow generic novelty claims. Fixed-dimensional consistency, finite-N covariance, residual augmentation, Gaussian geometry and classical support LPs remain prior. The new case strengthens a finite-design illustration while its low frequency limits application claims. The current 62-page reviewed manuscript is unchanged. Specific-result priority, sufficient choice-analysis value and the formal PPI++ comparison remain open; this package is not a submission or acceptance certificate.

## Endpoint accounting correction

The [complete endpoint flow](verification/transition_accounting.json) counts both exterior and zero-power losses. Earlier narrative used “loss” for the finite-to-exterior subset. The primary 36000 records and the existing paired-transition CSV are unchanged. There are 4/500 total losses for AI-aided tuning and 3/500 for attributes-only tuning at 20 labels, involving six allocations; none occurs at 50 labels. The exact all-positive-power obstruction remains allocation 435. Run `python analysis/summarize_auxiliary_transition_accounting.py --base . --out regenerated_flow.json` to reconstruct the flow without public inputs or new fits.
