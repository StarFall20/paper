# Literature overlap audit: structural-fibre contribution

**Audit date:** 2026-10-07

## Search protocol

The overlap check used four routes. First, the JOCM 2021–2026 Crossref
corpus was collected from the journal DOI feed (222 records; 197 after
removing editorials, errata, and corrigenda). Second, exact-title and concept
queries were run through the open web search index for Google Scholar-like
coverage. Third, OpenAlex was queried for the terms below; its raw response is
saved in `work/literature/openalex_overlap_queries.json`. Fourth, direct public
Scopus result pages were attempted. Scopus returned no accessible result page
without an institutional session, so this audit does not claim a complete
Scopus count or citation count. A future revision should rerun the exact query
inside the authors' Scopus subscription and archive the export.

Queries:

* `structural fibre discrete choice`
* `falsification discrete choice experimental design`
* `utility preserving discrete choice`
* `invariance discrete choice experiment utility`
* `maximin model discrimination choice experiment`
* `model robust discrete choice experiments maximin interactions`

The search found no exact phrase matching the proposed “structural-fibre
maximin falsification design” or “fibre quotient completeness.” It did find a
materially adjacent 2013 conference abstract that must be cited and delimited.

## Closest collision

Errore, Nachtsheim, and Li, **“Maximin Model Robust Discrete Choice
Experiments”** (Spring Research Conference 2013), constructs designs that
maximize the minimum efficiency across main-effects, interaction, and
second-order model classes. The institutional record and abstract are
available at the [University of Palermo repository](https://iris.unipa.it/handle/10447/117074).
This is a genuine overlap in the words *maximin*, *model robustness*,
*interactions*, and *discrete-choice design*. The present paper must not claim
that maximin robust DCE design is new.

The proposed object differs in four testable ways:

1. **Constraint:** each reflected profile is required to have the same
   candidate sufficient-statistic vector within its fibre. This makes the
   candidate utility equality hold for every coefficient vector. The 2013
   abstract describes model-class efficiency and does not impose candidate-
   preserving reflected pairs.
2. **Estimand:** the target is the rank and smallest singular/eigen direction
   of an odd contrast operator for declared local departures. It is a
   falsification coverage question, not minimum estimation efficiency across
   competing utility models.
3. **Failure output:** a zero rank/eigenvalue identifies a departure direction
   that the available attribute support cannot test. This blind-direction
   certificate is part of the result, including the deliberate exclusion of
   interactions that are constant on a fibre.
4. **Nuisance control:** reflected tasks are screened for equal raw comparison
   signatures and the assignment unit is frozen before outcomes. These
   restrictions are part of the causal interpretation of the odd contrast.

The contribution is therefore a domain-specific testability construction, not
a new generic maximin criterion. The manuscript should cite the 2013 abstract
in the method section and state this boundary explicitly.

## Other adjacent work

* **Fok and Paap (JOCM, 2024):** pairwise composite-likelihood and GMM tests
  for general MNL misspecification. This is a neighbouring outcome-side
  specification test; it does not construct candidate-preserving task fibres
  or a pre-outcome coverage certificate. [JOCM record](https://doi.org/10.1016/j.jocm.2024.100531)
* **Breitmoser:** observable invariance foundations for conditional logit and
  presentation effects. These results delimit the interpretation of an
  invariance rejection; they do not select a fibre-complete DCE block.
  [Axiomatic foundation](https://doi.org/10.1007/s00199-020-01281-1)
* **Healy and Leo (JET, 2026):** a general framework asking which experiments
  test a model. It motivates design-aware falsification, while the proposed
  contribution supplies a concrete quotient-space operator for DCE utility
  bases. [JET article](https://doi.org/10.1016/j.jet.2026.106191)
* **McCausland, Marley, and Davis-Stober:** a population experiment exposing
  implications of stochastic choice axioms through broad menu coverage. It is
  model-free and menu-complete; the proposed design is candidate-basis-specific
  and uses within-fibre contrasts.
* **Shubatt and Yang; McGranaghan et al.:** comparison complexity and paired
  valuation designs explain why equal expected utility can still yield
  different choices. They are negative controls for interpretation, not the
  source of the structural-fibre construction.
* **JOCM efficient-design and Bayesian-design papers:** optimize precision or
  expected information for a chosen model. They are relevant comparators and
  do not supply a rank certificate for departures that are invisible to an
  incumbent basis.

## Anti-stitch decision

The original “BMST/UFIT + maximin” wording was too easy to read as a bundle of
borrowed components. The paper should make **fibre quotient completeness** the
primary idea:

\[
  D_S(h)=\frac12\{[h(A_+)-h(B_+)]-[h(A_-)-h(B_-)]\},
\]

and a task block is complete for a declared departure space \(H\) when

\[
  \operatorname{rank}(D_S)=\operatorname{rank}(D_{\mathcal X}).
\]

The null space is an explicit list of locally untestable departures. Maximin
selection is a secondary rule for choosing a well-conditioned complete block
under gap-bin and nuisance constraints. This hierarchy gives the paper one
central object and one falsifiable theorem instead of presenting a collage of
invariance, complexity, and optimal-design ideas.

## Current verdict

There is a **material partial overlap** with robust model-discrimination design,
so a claim such as “the first maximin robust DCE design” would be false. The
stronger bounded claim survives the audit: no located paper combines
coefficient-robust candidate fibres, an odd-contrast quotient operator, an
algebraic rank/NULL-space testability certificate, balanced nuisance signatures,
and a DCE task-selection algorithm. This claim remains provisional until the
authors rerun the exact searches in institutional Scopus and Google Scholar
and check the full text of any new hits.
