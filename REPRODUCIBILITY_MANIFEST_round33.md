# Reproducibility manifest round 33

The clean release surface is the `reproducibility_round33/` directory and the matching zip in `outputs/`. It contains the dependency closure for the manuscript's current primary claims. Exploratory scripts, internal notes, stale Word binaries, and raw third-party data are excluded.

## Environment

- `reproducibility_round33/requirements-analysis.txt`
- SHA-256 file manifest: `reproducibility_round33/MANIFEST.sha256`

## Primary entry points

- `analysis/support_complete_fibre_design.py`
- `analysis/fibre_baseline_comparison.py`
- `analysis/support_complete_power_curve.py`
- `analysis/verify_candidate_preserving_questionnaire.py`
- `analysis/common_dce_design_exposure_audit.py`
- `analysis/electricity_external_validation.py`
- `analysis/train_vehicle_external_validation.py`
- `analysis/parse_electricity_rda.py`

## Locked primary outputs

- `results/support_complete_fibre_design.csv`
- `results/fibre_baseline_comparison.csv`
- `results/support_complete_power_curve.csv`
- `results/questionnaire_candidate_preservation_audit.csv`
- `results/common_dce_design_exposure_audit.csv`
- `results/common_dce_design_menus.csv`
- `results/electricity_external_validation.csv`
- `results/train_vehicle_external_validation.csv`

Historical oracle-baseline diagnostics and the earlier dictionary benchmark are retained in the repository and clearly labelled; they are not primary round-33 evidence. Raw public datasets remain local inputs with provenance and checksums in `data/`.
