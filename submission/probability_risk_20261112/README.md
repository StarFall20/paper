# Reproduce finite solutions and probability loss

This standalone supplement evaluates the manuscript's existing compensated binary logit objective. It contains all 1920 planned path records, retained coefficients, raw and corrected metrics, independent support checks, plans and derivation. The 46-page main manuscript is preserved separately. No new estimator, new human responses or new AI responses are introduced.

Read [the results and limitations](evidence/README.md), [the derivation](evidence/theory.md), and [the closest-prior record](evidence/closest_prior_scope.json). Use the verified summary tables, not the initial Gaussian integration values. The mathematical consequence is about strictly positive powers tending to zero in the stated Gaussian model. The released categorical data supply a conditional finite-design diagnostic.

## Environment and integrity

Use Python 3.12 and the pinned versions in requirements.txt. Install dependencies in an isolated environment if needed. All local scientific imports are included. No external repository code is executed. The numeric public inputs are downloaded separately from the fixed commit in inputs/public_source_manifest.json, with size and Git blob verification before loading. The loader restricts pickle classes to the numeric types used by this release.

```sh
python verify_package.py
python analysis/fetch_public_conjoint_audit.py --manifest inputs/public_source_manifest.json --out local_inputs
```

## Recheck the retained results

This route independently replays the retained coefficient records and support endpoints without optimizing every path again. It verifies the finite-design bounds, gradient residuals, public losses and Gaussian integrals. It preserves the initial evaluation discrepancies and six support resolutions. Output goes to a new directory.

```sh
python analysis/verify_vanishing_power_risk.py --root evidence --input-dir local_inputs --out regenerated/verification
python analysis/summarize_vanishing_power_risk.py --root evidence --out regenerated/summary
```

The first command uses independent probability-coordinate interior feasibility and direction-coordinate exterior programs at the path endpoints. Coercivity nesting for the same frozen prediction covers intermediate powers. It replays 1401 numerical fits and 389 endpoint programs, with no unresolved final state. Numerical stationarity is distinguished from a uniform parameter-distance certificate. The second command derives summaries from the included verification_final files; it does not overwrite the raw results.

## Fit every planned path again

Use a fresh output folder; each run records its source closure and plan hash. The current driver corrects the Gaussian quadrature while using the same optimization and support code as the retained initial run. Its closure therefore differs from the raw v1 driver. Do not use --resume with an old-closure result directory.

```sh
mkdir -p regenerated/fresh_paths
cp evidence/public_plan.json evidence/gaussian_plan.json regenerated/fresh_paths/
python analysis/vanishing_power_risk_audit.py --mode public --plan evidence/public_plan.json --input-dir local_inputs --ordinary inputs/ordinary_support_checks.json --out regenerated/fresh_paths/results/public
python analysis/vanishing_power_risk_audit.py --mode gaussian --plan evidence/gaussian_plan.json --out regenerated/fresh_paths/results/gaussian
python analysis/verify_vanishing_power_risk.py --root regenerated/fresh_paths --input-dir local_inputs --out regenerated/fresh_paths/verification_final
python analysis/summarize_vanishing_power_risk.py --root regenerated/fresh_paths --out regenerated/fresh_paths/summary
```

Fresh raw results may retain numerical support proximity flags; the independent endpoint verifier resolves them with actual recession rays. Do not discard these rows or turn an optimizer success flag into proof of finite existence.

The public follow-up uses the first 60 previously fixed block allocations, 20 labels per allocation, and the same 3405 evaluation rows. Source blocks are not verified participant IDs. Public allocations overlap and receive no population confidence intervals. The Gaussian experiment has 200 independent simulated model draws. Every power and adverse result is retained. Private questionnaire data and downloaded numeric inputs are excluded from this package.

## Attribution and claim scope

The objective, tuning, ridge reference, convex support and Gaussian margin mechanisms have established sources. The supplement derives a gradient-aware coefficient bound and a proper-loss consequence for that same objective. The prior-reading record identifies the external Gaussian margin theorem explicitly. The released vaccine data come from Lu and coauthors, ChengLu1999/generative-augmented-inference at commit d18717206867c7a12ff470879a0dd5d488b300c2; its pinned README states code MIT and data CC-BY-4.0. The input manifest retains the original README and license hashes and links. No originality or journal-readiness certification is implied.

The [clean-directory validation](evidence/clean_validation.json) replays all 1401 retained numerical fits and 389 endpoint programs from a fresh extraction. Every verification and derived summary file is byte-identical. The scientific analysis sources are the same 12 files used for that check. Full path optimization is available through the separate commands above; it was not repeated as a packaging test.
