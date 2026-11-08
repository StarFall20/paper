# DH36 analysis snapshot, 8 November 2026

This folder contains aggregate results for 360 complete questionnaire records. Sixty records retain the platform source label AI样本. Separate outputs use all records, 300 without that label, and 299 without that label submitted on the bulk collection date. These labels do not certify human identity or response validity.

Different recruitment channels each received one questionnaire version. The analysis describes cohort associations and predictive fit. It does not supply a randomized sufficiency test. The full-source uncertainty calculations also assume independence across records, which is unverified for the AI-labelled source.

## Reproduction

Raw exports remain private. Save the two numeric-coded Wenjuanxing Excel exports as V1_raw.xlsx and V2_raw.xlsx in a private input folder. The first six columns are platform metadata, followed by 8 choice questions and 3 background questions. The generic pipeline locates questions from their header labels and validates their codes.

Run from the repository root:

    python analysis/dh36_observational_analysis.py --input-dir /path/to/private-input --spec survey/public_dh36/instrument.json --output-dir /path/to/results --bootstrap-reps 4999

The public code writes participant-level preparation and fold assignments only inside the private input folder. Do not commit that folder. The output folder contains aggregate tables, figures and a report. The questionnaire menus and coding are stored in the public instrument specification.

Python 3.12 was used. The runtime contains numpy, pandas, scipy, openpyxl and matplotlib. Optimization uses analytic gradients and Newton refinement; final gradient, rank and participant-split checks are recorded. The common MNL, cohort intercept and cohort interaction models contain 5, 7 and 10 parameters. Cross-validation retains all 8 responses in one fold for each record.

The earlier 28-second record is retained in the full-source and non-AI-labelled analyses. Its trial status remains unconfirmed; date-based removal is explicitly a post hoc sensitivity analysis. No answer was deleted from Wenjuanxing. The source workbooks were preserved unchanged. No raw IP address, timestamp or individual questionnaire record is included here.
