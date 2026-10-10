# Finite support audits for discrete choice experiments

This repository contains the reproducible materials for the Journal of Choice Modelling manuscript **Testing Utility Basis Sufficiency in Discrete Choice Experiments**. The paper studies a finite support question: does a declared candidate menu summary leave any raw attribute variation available for checking? The main output is a pre-outcome certificate of within-fibre exposure and a support repair rule.

The scope is explicit. The certificate is finite-support and design-based. It is not a universal MNL misspecification test, a replacement for classical lack-of-fit design, or evidence of a human causal mechanism from observational data.

## Reproduce the release

Install the pinned analysis dependencies in `requirements-analysis.txt`, then run the following entry points from this directory:

```text
analysis/support_complete_fibre_design.py
analysis/fibre_baseline_comparison.py
analysis/support_complete_power_curve.py
analysis/swissmetro_specification_benchmark.py
analysis/electricity_external_validation.py
analysis/train_vehicle_external_validation.py
analysis/verify_candidate_preserving_questionnaire.py
analysis/common_dce_design_exposure_audit.py
```

The locked outputs used in the manuscript are:

```text
results/support_complete_fibre_design.csv
results/support_complete_fibre_benchmark.csv
results/support_complete_vs_saturated_benchmark_round32_n800.csv
results/support_complete_vs_saturated_benchmark_round32_n400_r1000.csv
results/fibre_baseline_comparison.csv
results/support_complete_power_curve.csv
results/swissmetro_specification_benchmark.csv
results/electricity_external_validation.csv
results/train_vehicle_external_validation.csv
results/questionnaire_candidate_preservation_audit.csv
results/common_dce_design_exposure_audit.csv
results/common_dce_design_menus.csv
```

`analysis/parse_electricity_rda.py` converts the public `mlogit` Electricity RDA to a local CSV. `analysis/electricity_external_validation.py` and `analysis/train_vehicle_external_validation.py` accept local public-data files; raw data are not redistributed. The exact source URLs, checksums, sample counts, preprocessing, and licence notes are in:

```text
data/provenance_swissmetro_2026-10-07.md
data/provenance_electricity_2026-10-08.md
data/provenance_train_vehicle_2026-10-08.md
data/README.md
```

## Main manuscript and survey instrument

The manuscript source is `manuscript/JOCM_fibre_sufficiency_revision.md`. The candidate-preserving pilot instrument for human respondents is `manuscript/candidate_preserving_dce_questionnaire.md`; the researcher-only fielding protocol is `manuscript/candidate_preserving_dce_fielding_protocol.md`. Editable Chinese and English Word versions are delivered under the task outputs. The rendered DOCX audit is stored in `results/questionnaire_candidate_preservation_audit_all_languages.csv`.

The clean release surface is `reproducibility_round33/` with the round-34 additions recorded in `REPRODUCIBILITY_MANIFEST_round34.md`; the matching zip is `outputs/JOCM_reproducibility_package_round34.zip`. The full code/data audit is `manuscript/full_code_data_audit_round34.md`. Earlier exploratory scripts and round notes remain in the repository for provenance, but they are not manuscript entry points and are excluded from the release manifest.

## Evidence hierarchy

- The finite rank/eigenvalue certificate is the primary design result.
- The complete-fibre LR, CMH, and cross-fitted ML residual scores are inferential comparators.
- The power curve varies departure amplitude and respondent count; the 1.000 rejection rates in the strong-signal benchmark are not presented as general power.
- Swissmetro, Electricity, and Train vehicle files are external implementation checks. They do not contain candidate-preserving randomization.
- A human-data sufficiency claim requires the randomized pilot instrument in the manuscript questionnaire.

## Relevant design literature

The common-design audit includes the NIST/SEMATECH L9 table and a local D-exchange search. The positioning now cites Atkinson and Fedorov's T-optimal model-discrimination designs, Goos et al.'s model-robust/model-sensitive designs, Gilmour and Trinca's compound precision and lack-of-fit criteria, and Wiens's lack-of-fit designs for binary responses. The manuscript states the increment directly: fibre conditioning on the complete candidate menu, full finite contrast coverage, and a computable support repair.
