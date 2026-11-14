# Independently trained predictions and the correction label budget

This supplement supplies a worked training limit for the existing compensated-choice objective. It retains a negative control using only fitted attributes and a second case with additional observed proxy information. The empirical signed-label feature mean and its Gaussian overlap law are established ingredients. No new learning algorithm or general conic theory is claimed.

The [worked argument](theory.md) states the assumptions, training proof, total-budget existence condition and an elementary necessary budget bound. The proxy information is available on all profiles; true auxiliary choices remain hidden from the learner and the corrected objective. Both total human labels and auxiliary profiles are held fixed across allocation fractions.

## Evidence

Each case has 8 conditions with 500 independent Gaussian model draws per condition: 4000 draws and 16000 criterion classifications. The same draw supplies its direct-human and three fixed-power comparisons. Allocation conditions use separate draws. These are simulated statistical outcomes, not respondents or LLM-generated answers.

The proxy case then reuses all 4000 existing cohorts for an all-positive-power audit. It finds 1080 explicit finite-power witnesses and 2920 obstructions. Strict human margins and negative unit-recession costs make the latter robust to numerical boundary proximity. This secondary audit adds no independent cohorts. [Its timing and scope](evidence/proxy/secondary_all_power_plan.json) distinguish it from the primary plan.

All original-loss directions, source hashes, 96 primary population conditions, independent integrals and secondary envelope calculations are retained. Primary direction records use deterministic compressed shards; the verifier also accepts the original JSON output format.

At a total slope-to-label ratio of 0.3 and auxiliary-to-total-label ratio one, increasing the training fraction from 25% to 50% reduces the limiting trained-predictor classification error from 0.334517 to 0.292939. The computed all-power envelopes increase from 0.477758 to 0.510186, while the required correction ratios increase from 0.4 to 0.6. The second allocation exceeds the envelope. At 160 total labels and 48 slopes, some positive power is finite in 419/500 and 90/500 cases, respectively. Direct human logit using all 160 labels is finite in 485/500 and 480/500 cases. All other conditions remain in the archive. These events do not evaluate the final corrected estimator's forecast loss or interval coverage.

The [figure](figures/learned_training_budget.png) displays the budget slack, trained-predictor classification error and the complete all-power event frequencies. Positive slack is the strict lower side of the computed envelope. Error bars in the event panels are 95% Wilson binomial intervals from 500 draws. Connecting segments join measured points; no continuum of allocation fractions was estimated. A separate [vector PDF](figures/learned_training_budget.pdf) is included.

## Reproduce

Use Python 3.12 and the recorded packages in requirements.txt. Commands run from this directory. The default workflow verifies the compressed evidence in a fresh output directory, reconstructs all datasets and saved original-loss directions, solves the preplanned 960 independent interval LPs, and reevaluates the normal integrals:

~~~sh
python verify_package.py
python run_reproduction.py --out regenerated
~~~

The default also verifies all 1080 finite-power witnesses and all 2920 strict obstruction directions. It does not rerun all 32000 primary support programs.

To regenerate the complete primary and secondary workflows:

~~~sh
python run_reproduction.py --out regenerated_full --full
~~~

Use a new output directory. No downloads, questionnaire data or external scripts are required. To rebuild only the figure:

~~~sh
python analysis/write_learned_training_assets.py --base evidence/proxy --out regenerated_figures
~~~

## Attribution and remaining scope

The primary reading [record](evidence/proxy/literature_scope.json) distinguishes the inspected source units from unread full proofs. Saglietti and Zdeborova already give a Hebbian mean and its signal/noise representation for a different Gaussian-mixture model. General training alignment, Gaussian rotation, classical support LPs and the manuscript's existing all-power envelope retain their attribution. The explicit training limit closes a condition of the current manuscript for this particular rule.

The attributes-only case has zero regular variance-optimal correction power under the correctly specified human model. Its positive-power comparisons are a negative control. The proxy case supplies additional score information; the examined finite powers remain diagnostics.

An initial standalone attempt lacked one transitive import; it stopped before generation. An initial secondary nesting assertion incorrectly included the direct-human baseline that uses all labels. Both records are preserved; the corrected audit retains every primary cohort. Earlier numerical weak directions were refined to strict-margin directions without changing classifications.

The current reviewed 61-page manuscript and its Word files are unchanged by this supplement. This package does not prove priority, sufficient journal value, general cross-fitting validity, optimal allocation or journal acceptance.
