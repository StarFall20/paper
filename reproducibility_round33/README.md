# Finite support audits for discrete choice experiments

This directory is the clean reproducibility surface for the Journal of Choice Modelling manuscript **Testing Utility Basis Sufficiency in Discrete Choice Experiments**. The paper studies whether a declared candidate menu summary leaves raw attribute variation available for checking.

## Reproduce the primary release

Install `requirements-analysis.txt`, then run the entry points from this directory:

```text
analysis/fibre_baseline_comparison.py
analysis/support_complete_fibre_design.py
analysis/support_complete_power_curve.py
analysis/questionnaire_cluster_power.py
analysis/full_code_data_audit.py
analysis/common_dce_design_exposure_audit.py
analysis/verify_conditional_baselines.py
analysis/verify_candidate_preserving_questionnaire.py
analysis/observational_equivalence_test.py
analysis/electricity_external_validation.py
analysis/train_vehicle_external_validation.py
```

The primary locked outputs are:

```text
results/support_complete_fibre_design.csv
results/fibre_baseline_comparison.csv
results/support_complete_power_curve.csv
results/questionnaire_cluster_power.csv
results/common_dce_design_exposure_audit.csv
results/common_dce_design_menus.csv
results/questionnaire_candidate_preservation_audit.csv
results/electricity_external_validation.csv
results/train_vehicle_external_validation.csv
```

The common-design audit checks the NIST/SEMATECH L9 array under all 24 column mappings and a 50-start × 3-prior local D-exchange sensitivity search. The finite rank certificate is a design audit; it does not establish a human mechanism from public observational data.

The single-response power grid and the respondent-cluster power grid answer different planning questions. `questionnaire_cluster_power.py` keeps eight tasks per respondent and uses respondent-level version permutations; its output must not be read as evidence from a collected sample.

The questionnaire files are respondent-facing pilot drafts. The full static audit is summarized in `manuscript/full_code_data_audit_round34.md`. The verifier reads the rendered DOCX tables under `survey/` and checks the displayed levels, shared attribute, and prices; it no longer relies on a hard-coded reconstruction. Pass the English pair with `--english-v1 survey/candidate_preserving_dce_questionnaire_en_v1.docx --english-v2 survey/candidate_preserving_dce_questionnaire_en_v2.docx` for the bilingual audit. The researcher-only deployment and analysis instructions are in `manuscript/candidate_preserving_dce_fielding_protocol.md`. A confirmatory field study still requires cognitive pretesting, ethics/privacy completion, and preregistration.

Raw third-party data are not redistributed. Source URLs, checksums, preprocessing, and licence notes are in `data/`.

Historical oracle-baseline files remain in the full repository for provenance. They are not primary round-33 evidence.
