# Finite choice estimation at unit correction

The unit endpoint removes the labelled log partitions. The [proof](proof.md) treats both possible feasible endpoints. When the unit fit has an affine set of minimizers, the continuation selects the ordinary labelled-loss minimizer on that set. When the unit target is on a proper auxiliary face, a linear program minimizes ordinary labelled recession subject to keeping every outside auxiliary gap at least one. The subsequent max-min programs use labelled utility gaps and auxiliary gaps reduced by one. These gaps select the logarithmic coefficient direction and give the prediction-loss rate on any fixed evaluation menu.

The contrast between the two families matters. With the same labelled negative-utility directions and a three-alternative auxiliary menu, the first program permits every second-coordinate direction from −1 to 1. The joint gap rule selects zero. Using the human gaps alone selects 1/2. Using only the varying auxiliary gap selects −1. In the original corrected logit, the second coefficient converges to −log(3)/2. A held-out choice on the second attribute has limiting loss log(1 + 1/√3). The full calculation also covers binary menus, loss of auxiliary rank, a positive-dimensional auxiliary face and a design with no finite continuation.

The mathematical checks fit 30 points in five constructed paths down to a distance of 10⁻¹² from unit correction. All scores are re-evaluated using the original human, predicted-human and auxiliary losses. At both ends of the grid, the solution is also recovered starting from zero. An invertible recoding preserves the direction in original utilities. [Results](results.json) · [Pre-analysis plan](plan.json)

The [saved unit states](archive_states.json) contain 3,824 fits across the Swissmetro, RADIANT food and released-prediction audits. All were recorded as interior or exterior. The unit-face paths here are constructed tests of a distinct mathematical case. The saved counts describe the existing allocation designs.

The overlap and max-min principles are established. The contribution here is the rule for combining the two choice-gap scales when labelled weights vanish and the resulting loss rate. The public comparison checked the original exponential-penalty treatment and the publisher's description of weighted supervised separation. [Cominetti and San Martín](https://www.ceremade.dauphine.fr/~carlier/Cominetti-SanMartin.pdf) · [Weighted logistic existence](https://doi.org/10.1080/03610926.2016.1260742)

## Reproduction

Use Python 3.12, NumPy 2.5.3 and SciPy 1.18.1.

    python verify.py --out replay.json
    python audit_archives.py --repo /path/to/paper --out replay_archive_states.json

The first program takes no external data. The second reads only the three existing derived archives. Pass the extracted Reproduction_Materials directory as --repo when using the submission archive. The files have the same source digests in both layouts.
