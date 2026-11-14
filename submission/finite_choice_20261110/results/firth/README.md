# Human Firth comparison on the original choice simulations

The completed replay adds an established human-only Firth comparator to the two displayed simulation conditions. Each uses the same 500 original samples, with no auxiliary predictions in the Firth fit. No questionnaire response, new respondent or LLM answer was generated. [The plan](plan.json) was fixed after the literature check and before these fits; this is an additional comparator, not a new estimator or a preregistered original study.

| Condition | Finite fits | Parameter MSE | MSE MCSE | Expected excess log loss | Marginal Wald coverage | Mean interval width |
| --- | --- | --- | --- | --- | --- | --- |
| 150_labels_overconfident | 500/500 | 0.108848 | 0.002946 | 0.037617 | 0.963091 | 1.336455 |
| 20_labels_oracle | 500/500 | 1.949426 | 0.147245 | 0.507106 | 0.999455 | 5.939401 |

Firth returns finite adjusted-score fits in both conditions. Its 20-label coverage of 0.99945 accompanies a mean interval width of 5.9394. At 150 labels its MSE exceeds dimension-rule DY by 0.007746, MCSE 0.000581, on the same 500 samples. Among the 412 shared 20-label fits, its MSE is 1.979159 versus unit PPI's 4.836151; their paired difference is −2.856992, MCSE 0.715238. The [paired table](results/paired.csv) retains every comparator and both error metrics. Different availability strata and covariance conventions remain explicit; no overall ranking is inferred from one comparison.

All 3000 predefined optimizer starts are retained. One 20-label design, repetition 157, has two distinct positive-Hessian local minima: zero and human-ridge starts yield penalized negative loss 6.260859; the negative-ridge start yields 5.827455 and is selected by the fixed best-of-three rule. Its coefficient spread is 5.07425. Optimizer success flags and verified residuals are reported separately. No global optimum certificate is claimed.

Independent determinant finite differences verify every retained root. Information inversion and reported metrics are rechecked for all 1000 fits. Fifty stratified direct-objective fits plus all three starts of the multiple-mode design give 53 independent optimizer checks on 51 designs. The initial 50-design check and the subsequent single-start comparison failure are retained. The corrected verification compares the same prescribed multistart operation; it does not assume a unique optimum. Final checked maximum coefficient discrepancy is below 6.9e−7 and objective discrepancy below 8.6e−14. [Complete verification](verification_complete/checks.json) identifies the scope.

Firth intervals are conventional inverse expected-information Wald intervals. They are not profile-likelihood intervals and do not use the other rows' estimating-equation sandwich. Coverage averages eleven coordinates within each repetition; the Monte Carlo unit is one repetition. The original human existence decisions match all 1000 replayed designs and their available MSEs agree within the recorded tolerance.

The [source comparison](prior_scope.md), [37-entry citation screen](current_citation_audit.json), and [prepared manuscript changes](prepared_manuscript_changes.md) identify what must change in the next integrated release. The currently reviewed 58-page paper and 28-page reading version are unchanged in this follow-up. The Firth results do not establish a novel estimator, general superiority or a journal submission threshold. The research goal remains active.

The inputs contain public attribute contrasts and a frozen simulation coefficient, plus selected already-generated baseline summaries at the repetition level. Public-data attribution and transformations are in [provenance](inputs/provenance.json). Raw human responses, original pickle files, participant identifiers and private questionnaire records are excluded.

The [isolated complete rerun](clean_validation.json) regenerates all 1000 fits and 3000 starts: six primary CSVs are byte-identical, the run JSON is semantically identical, and all three final verification outputs match. [Source reading and database access](primary_reading_and_access.json) records the primary-source scope and failed Scopus/Scholar access.
