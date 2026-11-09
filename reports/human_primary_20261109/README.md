# Current human primary estimation — 9 November 2026

Source roles follow `data/dh36_source_classification.json`:300 human-recruitment
records for human preference estimation;60 AI-pretest records for separate
diagnostics. The360-row export is never described as360 human respondents.

Run from repository root with numpy, pandas, scipy, openpyxl, matplotlib and
threadpoolctl. Raw exports and existing folds remain private.

```sh
python analysis/dh36_human_and_pairing_audit.py --input-dir "$PRIVATE_DIR" --instrument survey/public_dh36/instrument.json --folds "$PRIVATE_DIR/ideal_cv_folds_private.csv" --classification data/dh36_source_classification.json --primary-out reports/human_primary_20261109 --pairing-out reports/synthetic_pairing_audit_20261109
python analysis/write_human_ai_revision_report.py --primary-out reports/human_primary_20261109 --pairing-out reports/synthetic_pairing_audit_20261109 --classification data/dh36_source_classification.json
```

The historical fold filename does not establish random assignment. Validation
uses reused internal whole-record folds. Re-pairings are labelled controls,
not observations or a proposal to replace humans. Their ranges are procedural
sensitivity, not confidence intervals or formal permutation tests. No new
AI-generation task, sample purchase or human fielding was performed.
