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
