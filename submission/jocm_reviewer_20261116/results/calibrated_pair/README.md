# Calibrated predictors with matched conditional moments

[Derivation and complete finite-size table](derivation.md) · [Primary-source scope](prior_scope.md) · [Analysis plan](plan.json) · [Independent checks](verification/checks.json) · [Clean-run receipt](clean_validation.json)

Two conditionally calibrated prediction signals share the same target logit model, human design, conditional Brier risk and full fixed-dimensional score covariance function. Their finite-attainment boundaries can differ. This sharpens the existing Appendix B4 example within the same compensated objective.

The analytic balanced construction has a strict unit-power boundary inequality. A nonzero-signal example has computed boundaries 1.1886304119 and 1.2192432437 at the common variance reference power 0.8. All 36 exploratory pairs and 216 quadrature records remain. Independent adaptive integration, conditional posterior enumeration and 60 original support LPs check the calculation.

All 2400 finite-size classifications remain, including both adverse 1.35-ratio comparisons. At 80 labelled rows and the central 1.2 ratio the feasible counts are 106/200 and 107/200. The small-sample results do not establish a large or uniform practical advantage. Numerical agreement does not establish priority, and no publication-readiness score is assigned.

The [current main manuscript](https://github.com/StarFall20/paper/blob/backup/dh36-20261024/submission/finite_choice_20261110/manuscript/paper.md) remains the reviewed 58-page version. This separate follow-up awaits a concise integration decision. No participant or questionnaire records enter this package.

## Reproduce

Use Python 3.12 with NumPy, SciPy, Pandas, scikit-learn, Matplotlib and threadpoolctl. Versions used are recorded in [environment.json](environment.json). Run from this directory:

```sh
python run_reproduction.py
```

The command regenerates both stages into `regenerated/` and checks all eight scientific CSV/JSON files byte-for-byte. The 9 bundled analysis sources form the complete local import closure; supporting sources are existing public analyses. The two entry scripts generate synthetic data without loading public or private survey files. No historical scientific result is silently regenerated or replaced.

Verify the distribution inventory with:

```sh
python -c "import hashlib,pathlib; r=pathlib.Path('.'); a=[x.split('  ',1) for x in (r/'MANIFEST.sha256').read_text().splitlines()]; assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==h for h,p in a); print(len(a),'files verified')"
```

The retained [exploration](exploration/README.md) describes selection and the corrected numerical failure. The [Chinese progress note](进展与贡献判断.md) distinguishes the strengthened theoretical claim from remaining practical and submission limits.
