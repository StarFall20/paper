# Literature overlap audit, round 32

## Decision

The current paper should be presented as a **finite-support design certificate for
conditional sufficiency of a coarsened utility representation**. The evidence does
not support a claim that the inferential statistic is a new general lack-of-fit test.
On the declared finite support, the support-complete score spans the same fibre
contrast space as a saturated fibre alternative. The defensible increment is the
pre-outcome design object: a complete-menu-preserving fibre construction, an
explicit rank/eigenvalue certificate, and a support-repair rule.

This wording is deliberate. It keeps the contribution distinct from generic
misspecification testing and prevents a reviewer from identifying the score as a
relabelled saturated likelihood-ratio test.

## Search coverage and limits

The audit combined four sources: (i) a Crossref title/metadata sweep of 222
Journal of Choice Modelling records from 2021--2026; (ii) publisher and open
repository pages for the closest JOCM papers; (iii) broad web searches using
`conditional sufficiency`, `exposure mapping`, `level set`, `conditional
randomization`, `fibre/fiber`, `DCE`, `utility basis`, `support-complete`, and
`model discrimination`; and (iv) citation chasing from the closest methods.

Direct Scopus and Google Scholar result pages were not consistently accessible in
this environment. The audit therefore does **not** claim a database-complete
systematic review or a priority guarantee. The search log records the accessible
primary records and the boundaries that remain open to a reviewer check.

## Overlap matrix

| Work | Domain and null | What is shared | What remains different | Severity |
|---|---|---|---|---|
| Hoshino & Yanagi (2026), *Journal of Applied Econometrics*, DOI [10.1002/jae.70076](https://doi.org/10.1002/jae.70076) | Network interference; whether an exposure mapping correctly indexes potential outcomes | Coarsening map, level sets, conditional randomization, nested coarseness | Their assignment is a network treatment and the outcome is potential-outcome interference. This paper constructs complete candidate-menu fibres in a multinomial DCE and certifies all within-fibre response contrasts before outcomes | **Closest partial overlap; must be cited and bounded** |
| Athey, Eckles & Imbens (2018), exposure mappings and level sets | Network interference and randomization inference | Level-set logic and design-based testing of a coarse mapping | No utility-menu representation, DCE support repair, or fibre contrast basis | Partial |
| Fok & Paap (2025), JOCM, DOI [10.1016/j.jocm.2024.100531](https://doi.org/10.1016/j.jocm.2024.100531) | MNL/IIA misspecification; pairwise composite likelihood and GMM overidentification | Specification testing for choice probabilities; Monte Carlo power comparisons | Observational fit/moment restrictions; no candidate-preserving task construction, no finite-fibre rank certificate, no pre-outcome support repair | Adjacent, not same core |
| Bierlaire et al. (2009), nonparametric DCM specification test, DOI [10.1016/S1755-5345(13)70021-6](https://doi.org/10.1016/S1755-5345(13)70021-6) | General residual-based misspecification | Broad concern that a fitted DCM may be wrong | Smooth residual test of a chosen fitted model; it does not condition on a complete candidate prediction or enumerate finite fibre contrasts | Adjacent |
| Ortelli et al. (2021), JOCM, DOI [10.1016/j.jocm.2021.100285](https://doi.org/10.1016/j.jocm.2021.100285) | Assisted specification search | Utility specification selection and design implications | Searches candidate models; does not make candidate-equivalent menus and test discarded coordinates under a saturated response surface | Adjacent |
| Atkinson (2008), DT-optimal model discrimination, DOI [10.1016/j.jspi.2007.05.024](https://doi.org/10.1016/j.jspi.2007.05.024) | Design for distinguishing specified rival models | Experimental design can expose a model difference | T/DT criteria are pairwise or class-specific discrimination criteria. A candidate fibre can have zero conditional exposure even when a selected rival is separated; the present certificate covers every declared nonconstant finite-fibre response direction | Conceptual neighbour; explicit boundary needed |
| Standard saturated multinomial/fibre LR | Arbitrary profile effects within each fibre | Same finite contrast dimension and, in the current benchmark, the same rejection boundary | The LR is an outcome-stage inferential comparison. The proposed object adds a pre-outcome design certificate and a constructive support-repair algorithm | Inferential collision; acknowledged |
| Sufficient dimension reduction (Cook and successors) | (Y\perp X\mid R(X)) in regression | The conditional-sufficiency statement and reduction language | Classical SDR estimates a reduction from random covariates; this paper takes a candidate reduction as given and designs finite DCE support to make its violation observable | Conceptual ancestor, not a DCE method |
| Shubatt & Yang (2024/2026), trade-off complexity | Behavioural choice errors under difficult trade-offs | Complexity can alter observed choice even at similar utility differences | Complexity is a nuisance explanation to control or stratify; it does not provide a utility-basis sufficiency certificate | Nuisance mechanism |
| Extreme values, invariance and choice probabilities (2014), DOI [10.1016/j.trb.2013.10.014](https://doi.org/10.1016/j.trb.2013.10.014) | Invariance properties of random-utility choice probabilities | Choice probabilities can satisfy invariances under structural shock assumptions | The paper studies shock distributions and closed-form probabilities, not design-based testing of a user-specified coarsening map | Distant |

## What the audit rules out

Three stronger claims are not supported and have been removed from the manuscript:

1. A claim to be the first conditional-randomization specification test. Hoshino
   and Yanagi make that claim in a neighbouring interference setting.
2. A claim to define a new inferential family beyond a saturated fibre alternative.
   The current score and saturated fibre LR have matching rejection rates in the
   locked benchmark, which is a useful boundary result.
3. A claim of global nonparametric identification on a continuous attribute
   domain. The certificate is complete only relative to the declared finite support
   and the complete candidate menu vector.

## What remains independently useful

The paper's independent object is operational and design-based:

* preserve the entire candidate probability vector, not only one utility index;
* enumerate each non-singleton feasible fibre in a finite DCE support;
* span every nonconstant response contrast in each fibre;
* report rank and the smallest exposure eigenvalue before outcomes are observed;
* repair the task pool when a fibre block has zero rank or poor conditioning;
* separate structural visibility from power, and report effect-size/sample-size
  calculations for the latter.

The current 3x3 design gives a direct certificate: an endpoint candidate-precision
design has fibre exposure rank 0, whereas the support-complete design has rank 4
and minimum eigenvalue 0.1222. This is a design failure that a parameter-precision
criterion can miss. The larger simulations are being added separately; they are
validation of this boundary, not evidence that the statistic is a new LR.

## JOCM threshold assessment

The journal's stated scope welcomes theoretical and applied choice-modelling papers
with a methodological contribution or an innovative application, including discrete
choice models and survey design. The topic fits that scope. The threshold risk is
novelty framing: JOCM already publishes general misspecification tests and design
methods, so the manuscript must lead with the finite-support **design certificate**
and show why a conventional parameter-precision or specified-rival design can have
zero exposure to a plausible discarded-attribute direction.

The paper is at a credible submission threshold for a methods-oriented JOCM
submission after the expanded simulations and the explicit overlap paragraph are
included. It is not evidence for a guaranteed acceptance. A reviewer can still ask
for a real randomized candidate-preserving DCE; the public Swissmetro analysis is
properly labelled an implementation check and cannot answer that request.

