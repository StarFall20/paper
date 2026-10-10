# Full code and data audit — round 34

This report separates executable integrity from scientific readiness. A passing software check does not supply randomized human evidence.

- Audit date: 2026-10-10
- Tracked files: 332
- Tracked Python files: 81
- Tracked DOCX files: 5
- Clean release files: 44

## Findings

| Status | Check | Evidence | Required action |
| --- | --- | --- | --- |
| PASS | python_syntax | 81 tracked Python files; parse failures=0 | Fix every parser failure before release. |
| PASS | release_import_closure | 11 reachable release files; missing local imports=0 | Keep every runtime local dependency in the clean package. |
| PASS | release_hash_manifest | 44 release files; manifest failures=0 | Regenerate MANIFEST.sha256 after every release edit. |
| PASS | locked_csv_integrity | 14 release CSVs; malformed=0 | Repair malformed or empty locked outputs. |
| PASS | rendered_questionnaire_audit | 32 Chinese/English DOCX cards checked from rendered tables | Keep the DOCX audit in the release check. |
| OPEN | human_data | No collected candidate-preserving respondent data are in the repository | Do not claim empirical rejection or population validity; field and preregister a pilot first. |
| OPEN | independent_review | No independent mathematical or journal-style peer review is recorded | Obtain an external review before calling the paper submission-ready. |
| OPEN | design_scope | The common-design result is an illustrative L9/D-exchange audit under one toy coarsening | Add a real DCE design family or narrow the practical claim to the illustration. |
| OPEN | measurement_validation | The questionnaire composite candidate index has not passed cognitive validation | Run comprehension, difficulty, opt-out, and version-balance checks before confirmatory use. |

## Release CSV dimensions

| File | Data rows | Columns |
| --- | ---: | ---: |
| `results/common_dce_design_exposure_audit.csv` | 180 | 13 |
| `results/common_dce_design_menus.csv` | 931 | 9 |
| `results/electricity_external_validation.csv` | 4 | 17 |
| `results/fibre_baseline_comparison.csv` | 10000 | 8 |
| `results/questionnaire_candidate_preservation_audit.csv` | 16 | 16 |
| `results/questionnaire_candidate_preservation_audit_all_languages.csv` | 32 | 16 |
| `results/questionnaire_cluster_power.csv` | 18 | 10 |
| `results/support_complete_fibre_benchmark.csv` | 5000 | 7 |
| `results/support_complete_fibre_design.csv` | 2 | 7 |
| `results/support_complete_power_curve.csv` | 21 | 11 |
| `results/support_complete_vs_saturated_benchmark_round32_n400_r1000.csv` | 5000 | 6 |
| `results/support_complete_vs_saturated_benchmark_round32_n800.csv` | 2500 | 6 |
| `results/swissmetro_specification_benchmark.csv` | 4 | 15 |
| `results/train_vehicle_external_validation.csv` | 4 | 17 |

The release is executable after the repaired dependency closure and rendered-DOCX audit. The manuscript remains a method-and-design study with planning simulations. Human-data sufficiency claims, a validated measurement instrument, and independent review remain open gates.
