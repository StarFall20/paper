# Prediction loss when a correction limit meets several faces

At a positive correction-weight limit, several faces of the attainable moment set may bind. The new calculation recovers a unique leading coefficient direction from the original choice menus. A sequence of linear programs first minimizes the cost of approaching the corrected moment, then maximizes the smallest remaining utility gaps. For an evaluated choice, its utility shortfall in the selected direction gives its log-loss growth rate. A choice with zero leading shortfall has bounded loss. [Proof and attribution](proof.md)

## Checks on the choice objective

| Design | Normal dimensions | First minimizing set | Selected normal direction | Extra minimum-gap levels |
|---|---:|---|---|---|
| Single facet | 1 | Unique | 1 | None |
| Two dimensions | 2 | Unique | (1, 1) | None |
| Two dimensions with a free slope | 2 | Second component in [−1, 2] | (1, 1/2) | 5/2 |
| Three dimensions with two free slopes | 3 | Second and third components vary | (1, 1/2, 1) | 5/2, 5 |

All examples have three alternatives per menu, a finite tangent coefficient, strictly positive predictions and a corrected boundary at weight 1/2. The two-dimensional example with a free slope shows why the first program alone cannot identify a choice forecast. At each tested weight, one original score equation forces the second normal coefficient to half the first. The model predicts a holdout loss growth rate of 1/2 where the evaluated choice exposes that coordinate. Changing the utility basis with an invertible map and translating the menus yields the same predictions.

The program verifies 72 fitted points on the original three-term score at distances down to 10⁻²⁸ from the upper limit. In the free-slope example the final observed rates are 0.98284 and 0.49142. Dividing by the log distance leaves a bounded additive contribution, which accounts for the remaining difference. Independent formulas recover the limiting intercept corrections for all eight design versions. [Fit records](results.json) · [Independent checks](verification.json)

The existing RADIANT and Swissmetro audits contain 107 and 93 verified positive uncapped endpoints. All 200 use a single supporting facet. The multiaxis evidence uses constructed choices. The paper’s existing empirical comparisons continue to cover the single-facet case.

## Scope and provenance

The result applies to affine changes in the corrected moment and menu weights, with positive menu weights at the limit and full contrast rank. It describes a finite design. It extends the original single-facet expansion and ties an established successive-slack selection principle to grouped categorical logit and prediction loss. The exponential-penalty linear-programming center predates this choice-model calculation. [Cominetti and San Martín](https://doi.org/10.1007/BF01582224)

Run the commands below from the extracted archive with Python, NumPy and SciPy. `archive/` contains two pinned audit outputs needed for the empirical scope check. The example program uses constructed choices and does not require respondent data.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python check.py --out replay.json
python verify.py --results replay.json --archive archive --out replay_verification.json
```
