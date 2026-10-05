# Journal of Choice Modelling submission readiness

## Current decision

The manuscript fits the journal's methodological scope as a specification
study, but the original generic "ML-assisted model search" claim overlaps with
recent JOCM work on assisted specification, extensive mixed-Logit hypothesis
search, reinforcement learning, and LLM-supported specification. An
independent pivot has now been prototyped: a model-equivalent choice-pair test
holds the candidate utility differences fixed while changing the attribute
decomposition. It is not ready for the main manuscript until a valid paired-task
instrument is verified.
The public Swissmetro file currently has too few clean repeated profiles for
that test, so a paired-task supplement or a different dataset is required
before the pivot can carry the empirical paper.

## Novelty and overlap gate

- Keep the contribution centred on mechanism-specific operating boundaries: observable functional-form misspecification, discrete segmentation, and continuous latent heterogeneity.
- Report term recovery, behavioural refitting, calibration, and decision regret alongside predictive fit. Prediction alone is insufficient evidence of a new choice-modelling contribution.
- Cite and distinguish the recent JOCM assisted-specification, mixed-Logit search, reinforcement-learning, and LLM papers in the introduction and discussion. The detailed positioning matrix is in `manuscript/novelty_positioning.md`.
- The current RF selector recovers 4.0 of 5 represented terms on average in the combined condition (precision 0.91; recall 0.80) and recovers no threshold hinge terms in the threshold-only condition. Treat this boundary as a result, not as a hidden weakness.
- Before submission, run a candidate-library omission stress test. The present benchmark gives the selector access to all represented terms, so external validity of term recovery remains an open threat.
- For the paired-task pivot, report bootstrap size, power, localization, and the loss of rejection after the omitted term is added. The public Swissmetro file does not contain enough clean randomized matched pairs; do not label its ordinary nine-task panel as an empirical equivalence test.

## Ordered work plan

### 1. Lock the paper claim

Use one central claim: ML diagnostics can identify utility terms that improve an interpretable choice model under specific preference mechanisms and validation conditions. Keep prediction, behavioural recovery, and decision value as separate evidence outcomes.

### 2. Finish the core experiment

- The assisted-specification experiment now has 30 replications per condition. Preserve this result set as the locked simulation benchmark.
- A pure-NumPy two-class Latent Class MNL extension is available as a model-family check. Its results should remain separate from the locked benchmark until the full estimation settings are fixed.
- A targeted random-price Mixed Logit extension is available as a heterogeneity check. It should remain separate from the locked benchmark until the full random-coefficient specification and WTP recovery are documented.
- Random Forest and HistGradientBoosting are available in a grouped-holdout extension. HistGradientBoosting is labelled as a boosted-tree proxy while the exact XGBoost runtime remains unavailable on the current macOS environment.
- Upgrade the targeted Mixed Logit check to the full random-coefficient specification and run exact XGBoost with documented tuning and an identical respondent-level validation budget.
- Report coefficient recovery, WTP recovery, calibration, choice-share error, and decision regret.
- Add sample-size and task-count sensitivity analyses.

### 3. Add external evidence

Run the locked workflow on LPMC and Swissmetro. Use temporal holdout for LPMC and respondent-grouped panel validation for Swissmetro. State the data licence and the exact preprocessing decisions.

### 4. Rewrite the manuscript around the evidence

Rewrite the title, abstract, introduction, results, and conclusion after the final tables exist. Each experiment should have one declared argumentative duty. Put implementation settings, seeds, and extended tables in the supplement.

### 5. Prepare the submission package

Follow the journal's current Guide for Authors and Elsevier's Editorial Manager checklist. Prepare the manuscript, figures, supplementary files, cover letter, author information, ORCID identifiers, funding statement, competing-interest statement, ethics statement if applicable, data-availability statement, and AI-use disclosure. The journal scope expects methodological or innovative applied contributions in choice modelling.

### 6. Reproducibility release

The repository has a clean README, an analysis requirements file, grouped-holdout scripts, and frozen simulation outputs. Add the data provenance records, download instructions, one-command smoke test, and a frozen results manifest before citing the release in the manuscript. Push the complete repository to GitHub and verify that the commit referenced in the manuscript is publicly accessible.

## Submission gate

Submit after the main claim, model set, external validation, declarations, and public code release all point to the same version of the results.
