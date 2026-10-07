# Literature and collision audit for NO-FQC

**Audit date:** 2026-10-07

## Search coverage

The audit used the current JOCM DOI corpus, open scholarly metadata, official
publisher records, and exact-concept web searches.

| route | search material | result |
|---|---|---|
| JOCM DOI corpus | Crossref feed for 2021--2026 | 222 records; 197 after removing editorials, errata, and corrigenda |
| OpenAlex | fibre, candidate-preserving, invariance, nuisance-orthogonal, model-discrimination, maximin DCE | no exact NO-FQC phrase; adjacent invariance, model-discrimination, and efficient-design records |
| Google Scholar-like web index | exact phrases and concept combinations | no exact phrase hit for “nuisance-orthogonal fibre” or “residualized fibre completeness” |
| direct Google Scholar | exact-phrase and title searches | public page timed out in the available session; no export or citation count is claimed |
| public Scopus | exact-phrase and title searches | public result pages required an institutional session; no Scopus count or citation count is claimed |

The final manuscript must rerun the recorded queries inside the authors'
institutional Scopus and Google Scholar accounts and archive the exports. The
accessible search supports a bounded novelty claim; it does not prove priority.

## What recent JOCM work covers

| paper | main object | boundary for NO-FQC |
|---|---|---|
| Ortelli et al. (2021), assisted specification | interpretable utility-term search using flexible learners | does not construct candidate-preserving task pairs or residual tangent rank |
| Pérez-Troncoso (2022), sequential DCE strategy | sequential Bayesian design updates for precision | targets estimator precision, not testability of a frozen candidate basis |
| Kazagli and de Lapparent (2023), heterogeneous decision rules | latent classes, inertia, serial correlation, and rule membership | models process heterogeneity after data; it does not supply a pre-outcome fibre certificate |
| Fok and Paap (2024), MNL misspecification tests | outcome-side composite likelihood/GMM tests | uses observed alternatives and moments; it does not hold the full candidate menu vector fixed across task versions |
| Quainoo et al. (2024), model choice and framing | how post-data model choices affect loss-aversion estimates | studies model-choice sensitivity, not a design-based nuisance projection |
| Biswas et al. (2024), stochastic variables and random coefficients | joint stochastic attributes and taste heterogeneity | changes the estimation model; it does not create a candidate quotient |
| Akinc et al. (2024), varying choice-set sizes | efficiency and respondent burden under different menu sizes | design varies menu size; it does not preserve a candidate menu prediction |
| Mao et al. (2025), simulated annealing | algorithmic improvement of Bayesian design efficiency | no rank/null-space certificate for structurally blind departures |
| Nova et al. (2025), modeller serious game | telemetry of model-development workflows | studies analyst behavior, not utility testability |
| Cubero Dudinskaya et al. (2026), engagement and complexity | latent engagement and task ease/complexity | motivates nuisance controls; it does not identify a utility signal orthogonal to them |

The common unresolved issue is design support: a model can look adequate on
the observed tasks while a particular departure is never exposed. NO-FQC
addresses that issue before outcome collection and states the support condition
under which the test has content.

## Collision matrix

### Fok and Paap: MNL misspecification

Their tests are the closest JOCM econometric comparator. They test observable
moment restrictions and IIA-related implications. NO-FQC imposes a different
restriction: a complete candidate menu-difference vector is preserved across
two observed attribute decompositions, then the odd response is projected away
from a frozen nuisance tangent library. The estimands differ: outcome-side
misspecification moments versus residual fibre testability.

### Errore, Nachtsheim, and Li: maximin robust DCE

Their model-robust maximin criterion is a direct precedent for maximin design.
NO-FQC drops any generic maximin novelty claim. Maximin is used only after
residual rank completeness to improve the weakest direction.

### Invariance and transformation work

Breitmoser and latent-utility permutation papers establish that invariance can
support choice-model identification. NO-FQC uses an observed attribute-space
fibre as a randomized DCE instrument. It does not claim a general invariance
theory or a latent revealed-preference theorem.

### Nuisance-orthogonal inference

Semiparametric orthogonal scores and nuisance tangent spaces are the method
source for the projection step. The independent DCE increment is the
candidate-preserving fibre construction, the respondent-level assignment, and
the residual rank certificate. The manuscript must cite the method transfer
and state that the projection itself is not being claimed as a new general
inference principle.

### Complexity and framing

Shubatt and Yang, paired-valuation studies, and recent JOCM engagement work
show that raw comparison structure can affect choices. NO-FQC treats these
mechanisms as declared nuisance columns or deliberately unlisted negative
controls. They are not folded into the utility-departure claim.

### Adaptive model discrimination

Adaptive model-discrimination designs select informative stimuli among rival
models. NO-FQC begins with one frozen candidate basis and asks whether its
within-fibre relation is testable after nuisance projection. A future adaptive
extension is outside the primary claim.

## Novelty sentence that survives the audit

> We introduce a nuisance-orthogonal fibre quotient certificate for discrete
> choice experiments: reflected tasks hold the candidate's full menu
> differences fixed, and a weighted nuisance projection identifies the rank
> and weakest direction of utility departures that remain locally testable on
> the declared attribute support.

The manuscript must avoid “first transformation test,” “first maximin robust
DCE,” “general invariance test,” and “universal MNL misspecification test.”

## Decision

NO-FQC passes the conceptual independence gate with a bounded claim. It does
not pass the final priority or empirical gate until the full Scopus/Google
Scholar exports and a paired DCE or structurally matched external dataset are
available. The repository records this limitation explicitly.
