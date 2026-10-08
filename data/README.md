# Data provenance and acquisition

The repository does not redistribute third-party raw data. Download the source files from the original providers, record the download date and file checksum, then place local copies under `data/raw/`.

## London Passenger Mode Choice (LPMC)

Biogeme describes LPMC as 81,086 trips from the London Travel Demand Survey collected from April 2012 through March 2015. The official Biogeme data page is:

- https://transp-or.epfl.ch/biogeme-3.3.2/

The external validation protocol uses the final observed year as a temporal holdout. The preprocessing record must report the trip filters, alternative coding, missing-value handling, and scaling applied before estimation.

## Swissmetro

The Swissmetro stated-preference data and preparation code are documented in the official Biogeme documentation:

- https://biogeme.epfl.ch/sphinx/code/biogeme/data/swissmetro.html
- https://transp-or.epfl.ch/biogeme-3.1.0/examples_swissmetro.html

The validation protocol keeps respondent groups intact. The preprocessing record must report availability filtering, choice coding, panel structure, and any variable transformations.

## Local data record

Before an empirical run, add a local record containing:

- source URL and access date;
- filename and SHA-256 checksum;
- licence or redistribution terms;
- number of respondents, choice tasks, alternatives, and retained observations;
- preprocessing decisions and the script commit used for the run.

## Electricity (`mlogit`)

The public Electricity DCE is distributed as `Electricity.rda` in the CRAN `mlogit` package. The package documentation is https://search.r-project.org/CRAN/refmans/mlogit/html/Electricity.html and the raw package file is https://raw.githubusercontent.com/cran/mlogit/master/data/Electricity.rda. Convert it with `analysis/parse_electricity_rda.py`; run the external validation with `analysis/electricity_external_validation.py`. The exact RDA and converted-CSV checksums are recorded in `data/provenance_electricity_2026-10-08.md`.

## Train vehicle stated-choice sample

The Train vehicle archive is the public sample distributed with Kenneth Train's mixed-logit software. The official source page is https://eml.berkeley.edu/Software/abstracts/train1006mxlmsl.html and the archive URL is recorded in `data/provenance_train_vehicle_2026-10-08.md`. Raw data remain local. The validation entry point is `analysis/train_vehicle_external_validation.py`.

Neither public file contains the candidate-preserving random assignment required for a human-data sufficiency claim. They are external implementation and predictive specification checks.
