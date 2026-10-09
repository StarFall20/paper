# Baseline comparison round 33

The primary comparator uses 500 replications, 400 observations, η = 0.45, and 499 hypergeometric within-fibre draws for the support-complete score, generalized CMH, and complete-fibre LR. The ML residual score uses the same respondent-level 70/30 split and 499 held-out draws. The support score and generalized CMH use all profile contrasts; the LR compares unrestricted profile probabilities with unrestricted fibre probabilities.

| Condition | Support-complete | Generalized CMH | Complete-fibre LR | ML residual |
| --- | ---: | ---: | ---: | ---: |
| Null | 0.046 | 0.046 | 0.046 | 0.042 |
| Aligned | 1.000 | 1.000 | 1.000 | 0.998 |
| Interaction | 1.000 | 1.000 | 1.000 | 0.994 |
| Hidden quadratic | 1.000 | 1.000 | 1.000 | 0.992 |
| Hidden orthogonal | 1.000 | 1.000 | 1.000 | 0.998 |

The score and generalized CMH p-values are identical because both span the complete finite contrast space and use the same fixed-margin conditional law. This is a validation of the implementation, not evidence of an additional inferential family. The ML score is exploratory. The hidden direction is the only condition that is orthogonal to the old hand-written dictionary; that dictionary remains a negative control in the historical files and is not the primary comparator.
