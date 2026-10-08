# Baseline comparison round 33

The comparator benchmark uses 50 replications, 400 observations per replication, eta = 0.45, 99 conditional-randomization draws for the support-complete, complete-fibre LR, and CMH procedures, and 49 held-out permutations for the ML residual score. The ML score is cross-fitted across five folds; each forest is trained without the held-out outcomes and evaluated by its held-out log-score gain over the fibre-only probability.

| Condition | Support-complete | Complete-fibre LR | CMH | ML residual |
| --- | ---: | ---: | ---: | ---: |
| Null | 0.02 | 0.02 | 0.04 | 0.04 |
| Aligned | 1.00 | 1.00 | 1.00 | 1.00 |
| Interaction | 1.00 | 1.00 | 0.40 | 1.00 |
| Hidden quadratic | 1.00 | 1.00 | 1.00 | 1.00 |
| Hidden orthogonal | 1.00 | 1.00 | 0.70 | 1.00 |

The table is a comparator pilot rather than a final size study. The power curve is the calibrated evidence for amplitude and sample-size dependence. The CMH score uses a predeclared `a > b` raw contrast, so its lower power for the interaction conditions is a property of the chosen coarse contrast. The complete-fibre LR and support-complete score target the same saturated finite alternative; their role is to establish the design certificate and its inferential boundary. The ML score shows that a flexible learner can recover the strong signals in this small support while relying on a data split and a randomization reference.
