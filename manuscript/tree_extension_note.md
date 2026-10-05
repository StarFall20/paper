# Tree model extension note

Random Forest and HistGradientBoosting were evaluated on the same respondent-level holdout used by the behavioural models. Each task was represented by the attributes of all three alternatives plus the two observed respondent covariates. The extension uses fixed model settings and five replications per condition.

The XGBoost package was installed but could not load its macOS native library because `libomp.dylib` is unavailable in the execution environment. HistGradientBoosting is therefore reported as a boosted-tree proxy. It is not labelled as XGBoost evidence.

The tree extension is an implementation check. Its results are not part of the locked primary benchmark until tuning budgets, probability calibration, and the exact XGBoost environment are fixed.
