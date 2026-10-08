# Reproducibility manifest round 33

This manifest defines the files needed to regenerate the paper's current claims. It is the clean submission surface; development history is retained outside this list.

## Environment

- `requirements-analysis.txt`

## Manuscript

- `manuscript/JOCM_fibre_sufficiency_revision.md`
- `manuscript/candidate_preserving_dce_questionnaire.md`
- `manuscript/literature_lack_of_fit_round33.md`
- `manuscript/baseline_comparison_round33.md`
- `manuscript/power_curve_round33.md`

## Analysis entry points

- `analysis/support_complete_fibre_design.py`
- `analysis/support_complete_fibre_benchmark.py`
- `analysis/support_complete_vs_saturated_benchmark.py`
- `analysis/fibre_baseline_comparison.py`
- `analysis/support_complete_power_curve.py`
- `analysis/swissmetro_specification_benchmark.py`
- `analysis/parse_electricity_rda.py`
- `analysis/electricity_external_validation.py`
- `analysis/train_vehicle_external_validation.py`
- `analysis/verify_candidate_preserving_questionnaire.py`

## Data records

- `data/README.md`
- `data/provenance_swissmetro_2026-10-07.md`
- `data/provenance_electricity_2026-10-08.md`
- `data/provenance_train_vehicle_2026-10-08.md`

## Locked outputs

- `results/support_complete_fibre_design.csv`
- `results/support_complete_fibre_benchmark.csv`
- `results/support_complete_vs_saturated_benchmark_round32_n800.csv`
- `results/support_complete_vs_saturated_benchmark_round32_n400_r1000.csv`
- `results/fibre_baseline_comparison.csv`
- `results/support_complete_power_curve.csv`
- `results/swissmetro_specification_benchmark.csv`
- `results/electricity_external_validation.csv`
- `results/train_vehicle_external_validation.csv`
- `results/questionnaire_candidate_preservation_audit.csv`

Raw third-party files are local inputs and are not included in the manifest.
