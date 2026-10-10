# Journal of Choice Modelling submission gate — round 34

## Current decision

The package now passes the static code, data-shape, dependency-closure, manifest, and rendered-questionnaire checks. The manuscript is still a method-and-design draft, not a submission-ready empirical paper. It has no collected candidate-preserving respondent data, no cognitive validation of the composite index, and no independent review of the continuous proposition.

## New audit evidence

- All 81 tracked Python files parse. The clean release has no missing local runtime imports across its nine primary entry points.
- The 14 locked release CSVs are non-empty and rectangular. The release manifest covers 43 non-manifest files and verifies their SHA-256 hashes.
- The bilingual verifier reads all four rendered DOCX files. It checks 32 displayed alternative cards, the two candidate vectors, the sharing row, the price row, and the expected profile layout.
- The Electricity validator now uses the standard library CSV reader, so its declared requirements no longer hide a pandas dependency. The clean release includes `observational_equivalence_test.py`, which the Electricity and Train validators import.
- The questionnaire power grid uses eight tasks per respondent, respondent-level random effects, fixed 1:1 version allocation, and respondent-level permutation. Under the planning null, rejection rates are 0.030, 0.045, and 0.050 for 200, 400, and 800 respondents. At η = 0.20 they are 0.100, 0.245, and 0.420.

## Open gates

1. Obtain an independent mathematical review of Proposition 1, including support-boundary regularity and the relationship between fibre conditional laws and the field assignment mechanism.
2. Run a comprehension pilot for both questionnaire versions. Record difficulty, response time, opt-out use, and version balance before a confirmatory design.
3. Either field and preregister the randomized pilot or submit the paper explicitly as a design-method study with simulations. Public observational DCEs cannot supply the candidate-preserving causal contrast.
4. Replace the questionnaire placeholders for team, contact, privacy, and ethics before fielding.
5. Expand the common-design audit to a real DCE design family or narrow the practical claim to the current illustrative L9/D-exchange mapping.

The earlier round-32 “ready” assessment remains superseded. The full machine-readable audit is `manuscript/full_code_data_audit_round34.md`.
