# Journal of Choice Modelling submission readiness

## Current decision

The manuscript fits the journal's methodological scope once it is presented as an assisted utility specification study. The submission should wait until the main evidence package and the reproducibility archive are complete.

## Ordered work plan

### 1. Lock the paper claim

Use one central claim: ML diagnostics can identify utility terms that improve an interpretable choice model under specific preference mechanisms and validation conditions. Keep prediction, behavioural recovery, and decision value as separate evidence outcomes.

### 2. Finish the core experiment

- The assisted-specification experiment now has 30 replications per condition. Preserve this result set as the locked simulation benchmark and add the remaining behavioural model families.
- Add Mixed Logit and Latent Class MNL.
- Add a flexible learner with documented tuning and an identical respondent-level validation budget.
- Report coefficient recovery, WTP recovery, calibration, choice-share error, and decision regret.
- Add sample-size and task-count sensitivity analyses.

### 3. Add external evidence

Run the locked workflow on LPMC and Swissmetro. Use temporal holdout for LPMC and respondent-grouped panel validation for Swissmetro. State the data licence and the exact preprocessing decisions.

### 4. Rewrite the manuscript around the evidence

Rewrite the title, abstract, introduction, results, and conclusion after the final tables exist. Each experiment should have one declared argumentative duty. Put implementation settings, seeds, and extended tables in the supplement.

### 5. Prepare the submission package

Follow the journal's current Guide for Authors and Elsevier's Editorial Manager checklist. Prepare the manuscript, figures, supplementary files, cover letter, author information, ORCID identifiers, funding statement, competing-interest statement, ethics statement if applicable, data-availability statement, and AI-use disclosure. The journal scope expects methodological or innovative applied contributions in choice modelling.

### 6. Reproducibility release

Add a clean README, environment lockfile, data provenance, download instructions, one-command smoke test, and a frozen results manifest. Push the complete repository to GitHub and verify that the commit referenced in the manuscript is publicly accessible.

## Submission gate

Submit after the main claim, model set, external validation, declarations, and public code release all point to the same version of the results.
