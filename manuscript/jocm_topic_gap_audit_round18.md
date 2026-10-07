# JOCM topic and gap audit: 2021–2026

**Audit date:** 2026-10-07  
**Metadata source:** Crossref journal query for ISSN 1755-5345, 2021-01-01 to 2026-10-07  
**Purpose:** identify a problem that is visible in the recent JOCM agenda and remains distinct from the existing fibre-test claim.

## Corpus construction

The Crossref query returned 222 records. Removing editorial-board items,
errata, and corrigenda left 197 records. This is an abstract/title corpus, not
a full-text systematic review and not a Scopus count. The query is recorded as

`https://api.crossref.org/journals/1755-5345/works?filter=from-pub-date:2021-01-01,until-pub-date:2026-10-07&rows=1000&select=DOI,title,published,abstract,URL,type`

After removing common stop words, the leading terms were:

| term | count | interpretation |
|---|---:|---|
| experiments | 25 | DCE design and behavioural validation remain central |
| logit | 22 | MNL remains the principal reference model |
| preferences / preference | 33 | substantive preference interpretation remains the core use |
| analysis / modelling / modeling | 48 | workflow and specification decisions are frequent topics |
| estimation | 11 | parameter recovery and model fitting dominate the statistical output |
| learning | 11 | ML, experience, and sequential updating are growing themes |
| latent / heterogeneity | 21 | unobserved preference and process variation are persistent challenges |
| Bayesian | 9 | prior-sensitive design and estimation are active areas |
| utility | 7 | utility structure is discussed less often than generic fit or modelling |
| testing / specification | 7 | formal model adequacy remains a smaller stream |

The counts are descriptive. They identify the agenda and do not establish a
literature gap by themselves.

## Recent agenda streams

Recent JOCM work covers four visible streams.

1. **Assisted specification and ML.** Ortelli et al. formulate utility
   specification as a multi-objective search; the JOCM ML agenda and later
   latent-class/deep-learning papers extend data-driven estimation; recent
   work studies the workflow of choice modellers and reference-dependent
   neural choice models. A new algorithm-versus-MNL comparison would enter a
   crowded stream. [Assisted specification](https://doi.org/10.1016/j.jocm.2021.100285),
   [ML agenda](https://doi.org/10.1016/j.jocm.2021.100340),
   [modeller workflow](https://doi.org/10.1016/j.jocm.2025.100562)

2. **Efficient DCE design.** Bayesian designs, simulated annealing, sample
   size rules, ordering effects, and choice-set-size studies improve
   estimation efficiency or reduce design artefacts. Their objective is
   usually information for a chosen model, not a certificate that an
   incumbent utility basis can or cannot be falsified. [Bayesian design by
   simulated annealing](https://doi.org/10.1016/j.jocm.2025.100551),
   [ordering-effects review](https://doi.org/10.1016/j.jocm.2024.100489)

3. **Behavioural extensions.** Experience, reference dependence, latent
   attitudes, hypothetical bias, and task complexity enrich the data-generating
   process. These papers show why a choice-share difference can have several
   explanations; they do not provide a pre-outcome coverage measure for a
   candidate utility basis.

4. **Specification tests and invariance.** Fok and Paap propose pairwise
   composite-likelihood and GMM tests for general MNL misspecification, while
   Breitmoser derives observable invariance foundations for conditional logit.
   These results establish important neighbours and prevent a general
   “first invariance test” claim. [Fok and Paap](https://doi.org/10.1016/j.jocm.2024.100531),
   [Breitmoser](https://doi.org/10.1007/s00199-020-01281-1)

## The unresolved design problem

The streams leave a specific practical question open:

> Given a coarsened candidate basis, which candidate-preserving DCE tasks make
> the weakest local departures from that basis detectable, and which departures
> are structurally invisible on the available attribute support?

This question is different from selecting the best-fitting model, maximizing
the information matrix for its coefficients, or choosing between two fully
specified rival models. It concerns the **coverage of a falsification design**.
The answer should be available before outcomes are observed and should remain
valid when the candidate coefficients are estimated rather than known.

## Independent contribution selected from the gap

The new direction is a **structural-fibre maximin falsification design**. It
has one object:

- construct candidate-preserving pairs by equality of the candidate sufficient
  statistic vector, so equality holds for every candidate coefficient;
- screen the pairs for balanced raw comparison signatures;
- define a frozen library of local departures that can vary inside the fibre;
- select the task block that maximizes the smallest standardized local signal;
- report the smallest eigenvalue as a pre-outcome coverage certificate.

The certificate has a direct interpretation. A positive value says that every
departure direction in the declared local library has some support in the
selected block. A zero value identifies a blind direction and blocks a broad
power claim. This makes the method useful even when it rejects a proposed
design: the output tells the researcher which candidate relation cannot be
tested without changing the coarsening or the task space.

## Novelty boundary

Optimal design and model-discrimination literatures already contain D-, A-,
E-, and T-type criteria. The contribution is not a new generic optimal-design
criterion. The domain-specific increment is the restriction to **candidate-
preserving utility fibres**, the coefficient-robust sufficient-statistic
construction, the nuisance-balanced pair screen, and the falsification
coverage certificate. The criterion is transferred into a choice-modelling
problem where standard efficient designs can have zero support for a relevant
omitted direction. [Optimal experimental design for model discrimination](https://pmc.ncbi.nlm.nih.gov/articles/PMC2743521/)

The method must not claim global model adequacy, unique omitted-term
identification, or universal power. Its guarantee is conditional on the
declared candidate basis, departure library, nuisance screen, and feasible
attribute support.

## Decision

The structural-fibre maximin design is stronger than the previous
coefficient-calibrated parity grid and should become the main methodological
contribution. The smart-device application is a testbed for the certificate;
the final paper still needs a preregistered paired DCE or a documented dataset
with the structural assignment. The previous ML comparison remains a
benchmark and no longer carries the novelty claim.
