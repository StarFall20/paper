# Journal of Choice Modelling submission readiness

## Current decision

The paper now has one defensible candidate contribution:
**anchored behavioural metamorphic specification testing (BMST) for a
coarsened utility basis**. The test evaluates whether a frozen candidate treats
two randomized task versions identically when their full menu candidate
difference vector is unchanged. The domain excludes singleton fibres. The
instrument includes a zero-candidate-difference anchor and a geometry-preserving
common translation, then reports a geometry-changing decomposition arm as a
complexity control.

The original generic ML-assisted search claim is dropped from the lead. It
overlaps recent JOCM specification and flexible-model work. The manuscript is
not ready to submit until the randomized paired-task instrument is collected
or an equivalent documented dataset is found. The current simulation and
design evidence supports the method; it does not substitute for human-data
validation.

## Novelty and overlap gate

- Define the estimand as a violation of candidate-implied choice-probability
  equivalence on a predeclared, nontrivial fibre.
- Preserve the **full** menu vector \(\phi(x)\) in multinomial tasks. Matching
  one scalar or one chosen-versus-rejected gap is insufficient.
- Screen the candidate basis before design. Full raw-attribute linear bases with
  singleton fibres are outside the primary claim.
- Use the zero-gap arm to anchor the binary probability at \(1/2\) under
  symmetric errors for any positive scale. A rejection is evidence against the
  relation, not proof of a particular omitted mechanism.
- Use a common translation to preserve pairwise raw geometry. Record a
  geometry-changing arm and response-time/confidence measures to quantify
  comparison complexity.
- Compare against observational LR and an out-of-fold residual proxy on the
  same respondent budget. The contribution is a designed probe of an omitted
  direction that is weakly supported observationally.
- Cite Pedersen et al.'s randomized cost-attribute split and Shubatt and Yang's
  comparison-complexity model. State clearly that their interventions and
  estimands differ from the candidate-preserving fibre relation.
- Do not claim the first specification test, a universal MNL test, unique term
  identification, or proof of global model correctness.

## Locked planning evidence

The anchored benchmark uses 100 replications, 400 respondents, and 199
respondent-level sign flips. Under an omitted decomposition, the observational
LR and residual proxy reject at .04 and .04; the zero-gap geometry-preserving
arm rejects at .83; the geometry-changing nonzero-gap arm rejects at 1.00.
Under complexity-only data, the zero-gap arms remain near size (.02 and .07)
while the geometry-changing nonzero-gap arm responds at .41.

The sample-size curve uses 60 replications and 49 sign flips. For one focal
pair, nonlinear power is .20, .42, and .70 at 150, 300, and 600 respondents.
For three focal pairs it is .90, 1.00, and 1.00. Additive rejection remains
below .08. These are planning results and must be labelled as such.

## Ordered work plan

### 1. Freeze the method and preregistration

Specify the coarsened basis, the full menu \(\phi\), feasible shift grid,
nontrivial-fibre screen, geometry index, zero-gap anchor, translation and
decomposition arms, one-member assignment coin, respondent split, score
statistic, sign-flip draws, multiplicity rule, and mixed-signature action.

### 2. Build the empirical DCE

Use a forced-choice block for the first test. If an opt-out is included, give it
a visible profile and apply the same transformation to every alternative.
Randomize one pair member per respondent and balance the orientation across
respondents. Add additive, random-taste, order/carryover, and complexity
placebos. Record response time and confidence before analysis is unlocked.

### 3. Run the comparison

Fit the candidate on a development fold and freeze it. Evaluate BMST, the
observational LR, and the residual proxy on the same held-out respondents.
Report null size, power, usable-pair count, assignment balance, and the
direction of any relation violation.

### 4. Rewrite the paper

Lead the introduction with the omitted-direction problem created by a
coarsened basis. Give the formal relation before technical implementation.
Make the empirical DCE the main validation and keep the old ML/process work as
comparison and failure-boundary material. State that non-rejection is
insufficient evidence.

### 5. Reproducibility and submission

Keep the public repository pinned to the manuscript commit. Include the
preprocessing and data-provenance records, one-command smoke test, results
manifest, codebook, and simulation outputs. Prepare the manuscript, figures,
supplement, cover letter, author information, ORCID identifiers, funding,
competing interests, ethics statement, data-availability statement, and AI-use
disclosure required by the journal.

## Submission gate

Submit when one manuscript, one preregistered instrument, one results manifest,
and one public GitHub commit report the same claim. Until the empirical
paired-task gate is closed, label the work as a method-and-design study and keep
the human-data conclusion open.

See also \`manuscript/submission_gap_audit.md\`,
\`manuscript/index_search_log_round7.md\`, and
\`manuscript/ufit_proposition.md\`.
