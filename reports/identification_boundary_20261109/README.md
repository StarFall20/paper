# Direct menu information and identification boundary — 2026-11-09

This resource completes a finite audit of all 28 pairs of the existing eight
menus. It reports direct paired statistics, two classical conditional path
restrictions, effective discordant counts, illustrative fixed-alternative
power, Fieller net-share ratios and explicit context-sensitivity bounds.
No independently new method, pure-control mechanism, verified random order,
model sufficiency or submission readiness is claimed.

From repository root, with numpy, pandas, scipy, openpyxl and threadpoolctl:

```sh
python analysis/questionnaire_identification_boundary.py --input-dir "$PRIVATE_DIR" --instrument survey/public_dh36/instrument.json --out reports/identification_boundary_20261109
```

`PRIVATE_DIR` contains the original two XLSX exports. No answer vectors,
record identifiers, individual predictions, IP addresses or timestamps are
exported. All 360 records remain in the main analysis. Source and version
results are sensitivity descriptions. The special export source label is
retained; the researcher reports platform recruitment of humans.

Ratio intervals are not clipped to [0,1]; an unbounded Fieller set is marked
as unbounded instead of replaced with a convenient bounded interval. The
net-share ratio is not a count of individuals causally switching choices.
Context bounds are hypothetical assumptions, not measured task effects.
The exact binomial restrictions concern conditional path orientations under
the stated model, not a randomization test of questionnaire versions.

`boundary_numerical_audit.json` records design ranks, contrast weights,
source hashes, linear-program checks and a numerical panel-identity check.
These validate the stated calculations and do not establish novelty.
