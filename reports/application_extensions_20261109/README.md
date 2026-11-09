# Application extension and readiness reassessment — 2026-11-09

The fixed exploratory plan, all ten modifier tests, grouped validation diagnostics, linear-fee ratios, aggregate fitted parameters/covariances and a failed external-data preflight are retained. No independent innovation or submission readiness is claimed. Main analyses retain all 360 records; original source labels and the researcher's reported human-recruitment statement remain unchanged.

From the repository root, with numpy, pandas, scipy, matplotlib, openpyxl and threadpoolctl:

```sh
python analysis/application_background_extensions.py --input-dir "$PRIVATE_DIR" --instrument survey/public_dh36/instrument.json --folds "$PRIVATE_DIR/ideal_cv_folds_private.csv" --out reports/application_extensions_20261109
python analysis/public_healthcare_price_support_audit.py --input "$CHORUS_CSV" --out reports/application_extensions_20261109
python analysis/write_application_extension_report.py --out reports/application_extensions_20261109
```

`PRIVATE_DIR` contains the original two exports and private fold file. Its historical “ideal” filename does not establish random assignment. `CHORUS_CSV` is the unmodified CSV released at https://zenodo.org/records/19478777, MD5 0804176ec8f3114fd220da90966d8f26. The public audit records a preflight requiring clarification and deliberately fits no outcome model. The source file is not redistributed; the license and intended design must be clarified before any further release or inference. No emails or requests to the authors have been sent.

Whole-record folds are reused internally and do not constitute independent external validation. Holm adjustment is limited to the ten stated extension contrasts, not the entire exploratory project. Numerical summaries are checked against saved coefficient and covariance arrays. Raw answers, personal metadata, private record identifiers, individual predictions and external respondent rows are excluded from the public backup.
