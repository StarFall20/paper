# Progress status round 30

## Current state

The work is at the corrected audit stage, not the final submission stage. The earlier claim that model disagreement was a recoverability certificate has been withdrawn. The defensible contribution is a predeclared policy-path sensitivity audit with a direct simulation oracle and an exact finite-support common-mode failure witness.

## Completed

- Direct oracle audit: 24 replications for each of eight mechanisms at n=400, with corrected opt-out-by-income specification and converged L-BFGS MNL fits.
- n=800 replication: 24 replications for each mechanism; the ordering of model gaps and oracle errors is stable.
- Common-mode witness: two models agree exactly on q in {-1,+1}, yet the oracle differs off support.
- Population oracle correction: random-coefficient conditions use quadrature integration instead of treating a respondent draw as the population truth.
- Manuscript correction: old alert and observed-task regret claims are removed from the primary evidence; titles, authorship metadata, formulas, DGP description, and section numbering are corrected.
- Render QA: the six-page DOCX was rendered and visually checked after the final content changes.
- GitHub: corrected audit script, CSVs, report, and manuscript are on `main` at commit `94087ebd1c4fe4adcbd3ee1abea2200e13fd7732`.

## Open gates

- Re-run a corrected finite-sample alert calibration if the manuscript is to report a thresholded decision rule.
- Add the direct oracle and common-mode negative control to the final simulation section and update the submission package.
- Complete the final near-literature positioning paragraph and a clean repository manifest.
- Decide whether the exploratory score gate remains in the paper; it currently has no corrected operating-characteristic table.

## Time estimate

With the current scope and no questionnaire work, the remaining manuscript and repository work is approximately 4–8 focused hours: 2–4 hours for corrected alert calibration and robustness checks, 1–2 hours for final prose, references, and submission materials, and 1–2 hours for the final reproducibility and GitHub verification pass. A full external validation with a second public DCE would add roughly one working day.
