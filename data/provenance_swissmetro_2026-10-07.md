# Swissmetro external-validation data record

- **Source:** [Biogeme public Swissmetro data](https://github.com/michelbierlaire/biogeme/blob/master/src/biogeme/data/data/swissmetro.dat)
- **Raw download:** `https://raw.githubusercontent.com/michelbierlaire/biogeme/master/src/biogeme/data/data/swissmetro.dat`
- **Access date:** 2026-10-07
- **Local filename:** `data/raw/swissmetro.dat` (ignored by Git)
- **SHA-256:** `27432693cf052985d79a950b4b888be3efca798fc89b0d3ffefe40608ede00f2`
- **Raw rows:** 10,728
- **Retained rows:** 10,719
- **Dropped rows:** 9 records with `CHOICE=0`, the missing-choice code
- **Respondents after filtering:** 1,191
- **Alternatives:** train, Swissmetro, car; availability flags are retained
- **Choice coding:** source values 1/2/3 are mapped to 0/1/2 in the analysis
- **Transformations:** travel time and cost enter the candidate utility basis after division by 100; train and car alternative-specific constants are included
- **Analysis script:** `analysis/observational_equivalence_test.py`
- **Output:** `results/swissmetro_external_validation.csv`

The public file does not document one-member-per-pair assignment for a
candidate-preserving transformation. The run is therefore an external,
observational check of model fitting and approximate pairing. It does not
identify the randomized fibre estimand proposed for the paired DCE supplement.
The raw third-party file is not redistributed in the repository.
