# Existence of corrected Logit with a Jeffreys penalty

For a corrected mean choice loss with full-rank active contrasts, the growth of the negative log information determinant is governed by a minimum-cost basis of within-menu pair contrasts. This gives the exact global penalty threshold for strict coercivity on a given design with a negative recession direction, and a simple way to certify that a proposed penalty is insufficient. The general claim permits binary and multinomial menus. The selected public-data audit uses binary menus.

In Table 1 design C, scaling the corrected objective by the 32 labelled rows and applying the conventional one-half log-information penalty gives a normalized penalty of 1/64. The penalized objective still decreases along one ray with asymptotic slope −1/64. The strict coercivity threshold is 1/32.

We reproduce all 19 originally reported exterior witness designs from the publicly released vaccine choices, with their fixed forecasts, labels and auxiliary frequency scenarios. All 19 separating directions still give negative upper bounds on penalized slopes at the human-sample convention. The audit also compares the multinomial pair formula to a direct information calculation and tests 17 exact common-support designs that match derivative orders from 1 through 33.

Run from the extracted reproduction archive using its Python environment.

    python results/jeffreys_penalty_recession/audit.py --archive . --data /path/to/public/gai --out /path/to/results.json

The GAI archive uses the manifest and download instructions included with the earlier frequency audit. Appendix K states the threshold and its conditions. The calculation at the critical penalty classifies boundedness and coercivity. Finite attainment at equality can depend on the particular loss. The selected public cases give witness-level evidence. [Public data paper](https://proceedings.mlr.press/v306/lu26am.html)
