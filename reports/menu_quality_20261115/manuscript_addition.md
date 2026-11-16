> Historical preparation draft. The reviewed 16 November manuscript contains a condensed revision in Section 5.7 and Appendix F5; consult the current manuscript for authoritative text.

# Verified material prepared for the next manuscript integration

The current reviewed Word has not yet incorporated this addition. Counts and values below come from the completed whole-menu follow-up.

## Proposed Section 5.7 Probability quality on the same choice menus

We refit all 1000 Gaussian cohorts and all three nested dimensions from Section 3.9. The original human choices, predictions and fixed powers remain unchanged. Human ridge and equal-menu-weight pooling add explicit comparators. All 15000 records are retained. Ridge penalizes every fitted coefficient by its squared norm divided by twice the label budget. Pooling uses observed human categories and hard auxiliary predictions. Its target differs from the compensated loss. Neither comparator is tuned against the evaluation risk.

At 80 labelled menus and 52 orthogonal attributes, the variance-reference correction yields 269 finite estimates and ordinary logit yields 32. On the same 269 draws, median new-menu human log loss is 1.8524 for correction, 1.6837 for ridge and 1.0207 for pooling. With ridge substituted in the 231 unavailable correction cases, all-allocation mean loss is 2.264907, compared with 1.696402 for direct ridge; the paired excess is 0.568505 with Monte Carlo standard error 0.128705. Appendix F5 retains every condition and separates conditional fits from this explicit fallback rule. Greater availability in this experiment does not establish a prediction advantage.

## Proposed Appendix F5 Parameter and probability quality

The follow-up plan fixes every cohort and dimension of the existing availability experiment before risk fitting. Its original population settings were already selected and examined. Ordinary MLE, the existing oracle variance power, unit power, human ridge and hard-label pooling give 30 cells with 500 draws each. Every one of the 9000 original support states reproduces. All 11018 numerical finite fits pass an original-objective score check. Complete coefficients, squared coefficient errors, loss means, medians, 95th quantiles and maxima accompany the results. Interval coverage is not evaluated here.

Expected human menu loss is computed conditionally on each fitted coefficient. Gaussian rotation replaces the fitted noise component by its norm times independent alternative normals; constants and the common signal coefficient remain. Eight independent scrambled normal integrals use 16384 total nodes per fit, without generating evaluation choices. A separate antithetic integral checks 67 fits, including the 20 largest norms. Scramble errors assess integration and do not replace allocation Monte Carlo uncertainty.

Table F3 reports median human log loss on the same available draws for correction, ridge and pooling. The complete allocation and fallback tables retain all five methods. Conditional availability and heavy tails make both the denominator and the metric consequential. The true-model expected loss is approximately 0.72334, and the uniform forecast has loss log(3), approximately 1.09861. These are reference values under the stated generating law, not additional fitted methods.

Table F3  Median human menu log loss on shared numerical finite fits

|Labels|Slopes|Shared|Corrected|Ridge|Pooling|
|---|---|---|---|---|---|
|40|6|499/500|0.8518|0.8262|0.8073|
|40|16|447/500|1.1547|1.0409|0.9000|
|40|26|260/500|1.7641|1.3454|1.0256|
|80|12|500/500|0.8274|0.8278|0.7939|
|80|32|490/500|1.1356|1.1361|0.8841|
|80|52|269/500|1.8524|1.6837|1.0207|

At 80 labels and 52 noise slopes, 237 draws have a finite corrected estimate and no finite ordinary estimate. Their mean corrected loss is 2.893638, compared with ridge loss 1.713726 on those same draws. This availability-stratum comparison is a secondary description after risk fitting; all 24 condition-by-stratum rows are retained. Fixed ridge tuning, the changed pooling target and the known population power limit method comparisons to these stated conventions.
