# Exploratory within-person assumption audit — 9 November 2026

Run from the repository root with numpy, pandas, scipy, openpyxl and threadpoolctl.
The frozen DH36 exports remain private. No new responses or AI generations are required.

```sh
python analysis/within_person_choice_audit.py --input-dir "$PRIVATE_DIR" --instrument survey/public_dh36/instrument.json --classification data/dh36_source_classification.json --out reports/within_person_audit_20261109
python analysis/write_within_person_audit_report.py --out reports/within_person_audit_20261109
```

The optional external preflight uses separately downloaded files from the two
CC-BY4.0 deposits listed in `external_resource_preflight.json`, their publisher
metadata and download receipt. It executes no downloaded research code:

```sh
python analysis/fetch_external_choice_resources.py --manifest reports/within_person_audit_20261109/external_resource_preflight.json --out "$EXTERNAL_DIR"
python analysis/external_choice_resource_preflight.py --input-dir "$EXTERNAL_DIR" --out reports/within_person_audit_20261109
```

`analysis_plan.json` fixes the two conditional models and all56 association tests
before these new fits; prior outcomes had already been examined. The method is
classical conditional likelihood, with shared slopes and stable alternative
intercepts. Conditional scores do not evaluate the unconditional population target.
Person-specific error-scale and slope heterogeneity are not eliminated by this
conditioning argument. All56 association tests are retained after Holm correction.
No human/AI origin detector, new theory or submission completion is claimed.
