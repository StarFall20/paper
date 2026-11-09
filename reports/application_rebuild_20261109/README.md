> 来源说明已于2026年11月9日澄清：两版各30条为AI预测试作答。下文保留澄清前的混合来源分析／核验记录；其中未明确来源的文字属于当时信息状态。当前真人主分析为300条，见[来源纠正与真人主结果](../human_primary_20261109/真人主分析与来源纠正.md)及[AI预测试计算审计](../synthetic_pairing_audit_20261109/AI预测试配对审计.md)。

# Exploratory application rebuild — 2026-11-09

This package preserves the questionnaire and evaluates an applied question. It makes no new-method, verified-randomization or submission-readiness claim. Recruitment is recorded as the researcher's report of a platform human-recruitment description. Authenticated inspection confirms two 30-record AI-sample task notices and 60 matching point-log entries; the task-specific respondent provenance remains unresolved. See the [source-verification update](../platform_source_verification_20261109/README.md). Original source labels remain in the private exports. Public files contain aggregate results and code only.

Run from repository root with scientific Python dependencies numpy, pandas, scipy, matplotlib, openpyxl, threadpoolctl. Raw exports and fold assignments remain private. `PRIVATE_DIR` must contain V1_raw.xlsx, V2_raw.xlsx and ideal_cv_folds_private.csv. The historic fold filename does not certify ideal randomization. A private access holder can reproduce the estimates; an independent public download cannot recover private records or certify respondent provenance.

```sh
python analysis/application_support_models.py --input-dir "$PRIVATE_DIR" --instrument survey/public_dh36/instrument.json --out reports/application_rebuild_20261109
python analysis/application_panel_comparison.py --input-dir "$PRIVATE_DIR" --instrument survey/public_dh36/instrument.json --folds "$PRIVATE_DIR/ideal_cv_folds_private.csv" --out reports/application_rebuild_20261109
python analysis/application_source_diagnostic.py --input-dir "$PRIVATE_DIR" --instrument survey/public_dh36/instrument.json --folds "$PRIVATE_DIR/ideal_cv_folds_private.csv" --out reports/application_rebuild_20261109
python analysis/write_application_rebuild_report.py --input-dir "$PRIVATE_DIR" --instrument survey/public_dh36/instrument.json --out reports/application_rebuild_20261109
```

The second command implements the fixed eight-model family; the third implements the documented boundary/source follow-up. Five folds keep each record's eight choices together. Joint and marginal prediction targets are separate. Mixture prediction uses population weights for held-out records. Hessian/boundary diagnostics are retained. Classical full-outside mixtures are not claimed as independent innovation.

All p-values/intervals are exploratory and model conditional. Pooled results are accompanied by a 300-record source sensitivity, not a silent exclusion. The latter sample is defined by original source labels and is not independently verified as representative.

The figure's “AI samples” text is a literal translation of the original export label, not a finding of machine generation. The researcher reports platform recruitment of humans.
