# Public-data audit for the BMST empirical gate

## Decision

The public-data search did not find an existing dataset that supplies the
randomizable, candidate-preserving task pairs required by the primary BMST
estimand. The best new candidate is the open Computer Vision-enriched DCM
dataset from TUDelft. It contains repeated stated-choice observations for each
respondent, but its public schema identifies observations (`ID`) and
respondents (`RID`), not randomized duplicate task pairs. The associated
experiment assembled image pairs and attribute levels by random draws. This
creates repeated tasks, not a documented intervention in which one member of a
candidate-preserving pair is assigned independently to each respondent.

The dataset can support a descriptive external check of representation and
ordinary predictive transfer. It cannot support the primary BMST claim without
reconstructing the original task generator and proving the required assignment
exchangeability. The manuscript therefore keeps a purpose-built paired-task
supplement as the empirical gate.

## Audited sources

| source | what is available | BMST status | decision |
|---|---|---|---|
| [TUD-CityAI-Lab/Computer-vision-enriched-DCMs](https://github.com/TUD-CityAI-Lab/Computer-vision-enriched-DCMs) | Two-alternative residential location choices with travel time, housing cost, and images; repeated observations per respondent; `ID`, `RID`, and `N_TASKS` are documented in the README. | Repeated observations are available, but no public pair identifier or assignment variable establishes one-member-per-pair randomization. | Use for external descriptive or predictive analysis only. |
| [Van Cranenburgh & Garrido-Valenzuela (2025) experiment description](https://doi.org/10.1016/j.tra.2024.104300) | The experiment drew image pairs at random and pulled attribute tasks from preconstructed tables. | Random task generation does not establish a prespecified candidate-preserving transformation or an exchangeable assignment between two task members. | Do not treat the data as a BMST instrument. |
| Swissmetro public file | Existing repository audit found very few exact repeated profiles within respondent and orientation-sensitive approximate matches. | Exact randomized fibre is absent; the observational fallback has unstable orientation results. | Retain as a documented feasibility boundary, not as the main empirical test. |

## Why repeated tasks are insufficient

BMST needs a declared transformation (T) such that the candidate basis
difference vector is preserved, (d(Tx)=d(x)), before outcomes are observed.
The test then assigns one member of each pair to a respondent and uses a
respondent-cluster randomization reference. Repeated tasks without a pair
identifier leave the assignment mechanism, task ordering, and carryover
relationship unknown. Matching rows after observing choices would turn the
instrument into an observational comparison and would inherit the orientation
and selection problems already seen in the Swissmetro fallback.

## Required data collection to clear the gate

The supplement should preregister a finite grid of common-shift transformations,
generate both members of every focal pair, randomize one member per respondent,
counterbalance the display order, and record pair ID, transformation ID,
respondent ID, assignment coin, task order, and opt-out availability. The
development fold estimates the candidate basis. The held-out fold supplies the
randomization test. A carryover placebo and a scale-only negative control should
be included in the same instrument.

Until those fields are available, the paper should report BMST as a formally
defined and simulation-validated method proposal with a transparent empirical
data limitation. It should not present an existing repeated-task dataset as if
it were a randomized paired-task validation.

## Reproducibility record

The source README and the associated paper were inspected on 2026-10-06. The
repository is licensed CC BY-NC-SA 4.0. No raw copy was added to this project;
the audit records only public metadata and links.
