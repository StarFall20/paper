### Frequency changes in released vaccine choices

The public vaccine choice archive of Lu et al. (2026) supplies a second design library. The 1,200 auxiliary rows contain 1,199 distinct ordered menus. All menus retain positive integer frequencies in each of the seven designs, and the auxiliary total is always 4,800. The original design repeats each row four times. Two designs allocate additional rows by multinomial draws. Four others upweight the top or bottom fifth of rows by forecast probability at total variation distances 0.20 and 0.60 from the original row frequencies. The fits use 11 utility contrasts. Both ridge logit predictors use a separate 1,200-row training block, one with human choices and the other with archived AI choices as the training outcome. The same fitted probability function is used for both sample roles and is held fixed across the seven designs.

The following counts use 96 nested allocations from a disjoint label pool. The denominator retains allocations with full-rank finite ordinary Logit. An allocation crosses if some pair of its seven designs includes a finite estimate and an objective unbounded below. The label counts refer to choice records. The archive has no confirmed respondent identifiers.

Table J1  Allocations that cross the finite-estimation boundary by correction weight

| Predictor training outcome | Label records | Eligible | 0.75 | 0.90 | 0.95 | 1.00 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Human choices | 25 | 32 | 2 | 2 | 2 | 3 |
| Human choices | 50 | 94 | 1 | 2 | 4 | 4 |
| Human choices | 150 | 96 | 0 | 0 | 0 | 0 |
| Archived AI choices | 25 | 32 | 13 | 10 | 10 | 12 |
| Archived AI choices | 50 | 94 | 32 | 36 | 36 | 42 |
| Archived AI choices | 150 | 96 | 0 | 4 | 6 | 8 |

### Finite estimation along a frequency change

Fix the labelled records, forecasts and correction weight. Let the auxiliary frequencies move from $f_0$ to $f_1$ along $f_\rho=(1-\rho)f_0+\rho f_1$, with $0\leq\rho\leq1$. If both frequency vectors are positive on the same menus, every intermediate design has those menus. The menu masses and the corrected target are affine in $\rho$. Maximize $\rho$ in the probability-mass program of Section 2, replacing its fixed menu weights and moment with their affine expressions. The optimum $\rho_*$ is the last feasible transfer along this path.

Suppose the initial target is interior, the final target is exterior and the contrasts of the active menus span the coefficient space throughout the path. Mixing any feasible mass representation at $\rho_*$ with a strictly positive representation at zero yields interior targets for $0\leq\rho<\rho_*$. The moment is on the boundary at $\rho_*$. Interior feasibility there would extend to a larger transfer by continuity of the support inequalities on the unit sphere. After $\rho_*$ the support inequalities fail. The objective has a finite unattained infimum at $\rho_*$ and is unbounded below for $\rho>\rho_*$.

An example uses 50 labelled records, the AI-trained predictor and a correction weight of 0.75. Moving from the original frequencies toward the strongly upweighted lowest quintile gives a last feasible transfer of 0.743, at total variation distance 0.446 from the finite design. Expanded-row calculations and two probability formulations verify all 19 selected endpoint pairs. The frequency program also verifies an interior and an exterior point on either side of each computed boundary.

A separate exploratory calculation tests whether local derivatives can be held fixed on these same released menus. Starting from each selected finite design, choose positive fractional frequencies that match the objective value, gradient and Hessian at zero. The constraints preserve the score and the second moment of the utility contrasts. The search minimizes the recession cost in the direction recovered from the paired exterior design. It tests two total variation budgets, 0.20 and 0.60. Of 38 candidates, 32 are certified exterior, including 13 at budget 0.20. The latter cases include both predictors at 50 label records and correction weight 0.75. The largest numerical discrepancy among the matched derivatives is $2.85\times10^{-13}$. These frequencies are designed from the sample and describe what can happen within this menu library. The crossing counts describe the stated allocation and frequency rules for this archive.
