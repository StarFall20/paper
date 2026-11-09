# Source-conditioned validation of panel choices

Nine fixed exploratory benchmarks use the existing five whole-record folds.
The original source labels remain unchanged. All 60 AI-labelled vectors are
constant; all other 300 vectors are nonconstant. This fact does not certify
respondent origin or determine why a pattern occurs.

From the repository root, using the same scientific Python dependencies as
the application rebuild:

```sh
python analysis/source_conditioned_validation.py --input-dir "$PRIVATE_DIR" --instrument survey/public_dh36/instrument.json --folds "$PRIVATE_DIR/ideal_cv_folds_private.csv" --out reports/source_conditioned_validation_20261109 --private-out "$PRIVATE_DIR/source_conditioned_validation_20261109"
python analysis/write_source_validation_report.py --input-dir "$PRIVATE_DIR" --instrument survey/public_dh36/instrument.json --out reports/source_conditioned_validation_20261109
```

The historical fold filename does not certify ideal randomization. The
four-category hurdle is a classical diagnostic with an MNL component
conditioned on nonconstant responses. Training gate counts use fixed symmetric
Dirichlet smoothing of 0.5; 1.0 is a fixed sensitivity. Prediction inputs
exclude held-out answers. Do not interpret its category probabilities or
conditional utility coefficients as causal mechanisms or population WTP.

Source-specific and aggregate scores evaluate different questions. Source
weights in the standardization table are hypothetical. Paired intervals are
descriptive; reused folds do not provide selection-adjusted or independent
validation. Public outputs contain only aggregate results and fitted model
diagnostics. Individual predictions, assignments and raw records remain private.

This package is not a new-method claim or a declaration of submission readiness.
See the Chinese research assessment and the focused manuscript note.
