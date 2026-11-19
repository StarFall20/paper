# Local diagnostics on fixed-label auxiliary replacements

This secondary check uses the 96 fixed label groups in the earlier RADIANT and Swissmetro experiment. Each group supplies eight auxiliary samples of 64 people for each of two powers. The labels, prediction rule, utility specification, auxiliary size and zero reference coefficient stay fixed within each comparison. The support classifications were already in the archive. The follow-up uses its own [measurement plan](plan.json).

| Study | Correction weight | Label groups observed in both support states | Median distance within one support state | Median distance across support states | Closest opposite-state distance |
| --- | ---: | ---: | ---: | ---: | ---: |
| RADIANT | 1 | 6 of 48 | 0.0619 | 0.0770 | 0.0240 |
| RADIANT | 0.5 | 1 of 48 | 0.0326 | 0.0331 | 0.0188 |
| Swissmetro | 1 | 6 of 48 | 0.1786 | 0.2134 | 0.0473 |
| Swissmetro | 0.5 | 1 of 48 | 0.1114 | 0.0855 | 0.0599 |

The closest reported opposite-state pair uses the same five labelled food respondents at power 0.5. The normalized gradient and Hessian distances are 0.0136 and 0.0188. Its finite and unbounded cases have Newton decrements of 0.96079 and 0.96052. Across the 171 opposite-state pairs, 70 give the unbounded problem the smaller decrement. The nearest cases illustrate numerical proximity. The exact derivative-matching result in the paper is a mathematical construction.

The gradient and Hessian distances use the group mean Hessian as a common whitening matrix. Each norm difference is divided by the sum of the two norms. The reported distance is their maximum. An invertible recoding of the utility features changes the numerical calculation by at most (1.7\times10^{-14}). For one nearest opposite-state pair per mixed group, the verification uses the original menu probabilities and the earlier pair of independent support formulations. Complex-step gradients and central-difference Hessians independently check the derivatives in these 28 designs.

All 1,536 menu sets have a positive definite zero-point Hessian. The 5,376 pairs overlap in designs and respondents. The table summarizes the recorded comparisons without a population interpretation.

## Reproduction

Download the two original public sources using the notes in Reproduction Materials. Run its `verify_release.py` with both `--food-source` and `--swissmetro-source`. The verifier reconstructs every group from the upstream seeds and checks the saved summary. [RADIANT workbook](https://zenodo.org/records/15420092), [Swissmetro source](https://github.com/michelbierlaire/biogeme/blob/master/src/biogeme/data/data/swissmetro.dat)
