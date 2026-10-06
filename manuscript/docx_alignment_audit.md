# Word manuscript alignment audit

## Reviewed file

`/Users/Admin/Documents/BDAIEM Revision.docx` renders to eight readable pages.
The title, tables, figures, equations, references, and yellow-highlighted
revision blocks are visually intact. The file is a conference-style working
draft, not the current JOCM manuscript version.

## Claim mismatch

The Word draft is organised around the question “when does XGBoost improve
choice prediction?” Its abstract, introduction, methods, results, discussion,
and conclusion all treat the five-level synthetic complexity gradient as the
main contribution. The current repository has moved the main innovation audit
to a candidate-preserving paired-task randomization test. The two claims cannot
share one introduction without a clear hierarchy.

The paired-task method is currently validated only by simulation. The public
Swissmetro file does not contain the randomized matched tasks needed for an
empirical test, and the original A/B/opt-out instrument does not preserve
candidate utility differences when a shift is applied only to product
attributes. The Word draft must not describe the paired-task test as an
empirical result until a supplement or a different valid dataset is available.

## Text that must be replaced before submission

1. **Title and abstract.** Remove the generic XGBoost-versus-MNL title if the
   paired-task method becomes the main paper. Report the new null, the
   cluster randomization reference, and the instrument condition. If no
   supplement is collected, keep the paper as a narrower simulation benchmark
   and remove any claim of empirical diagnostic validity.
2. **Introduction.** Replace the model-race framing with the specification
   question: can a choice design create an equality that a candidate utility
   basis must satisfy? State the boundary against Breitmoser’s observable
   invariances and Fok and Paap’s pair-based MNL tests.
3. **Methods.** Add the common-shift construction, coefficient-wise equality,
   respondent-fold freezing, cluster sign flips, counterbalanced task order,
   and the opt-out rule. Move implementation settings to the appendix.
4. **Results.** Replace five-replication XGBoost endpoint tables with the
   50-replication randomization benchmark and its all-pair and shift-specific
   rows. Keep the factorial contrast as a documented failure boundary.
5. **Discussion and conclusion.** State that the test detects a violation of
   candidate-preserving task invariance. It does not identify a unique omitted
   term, prove the candidate correct after non-rejection, or generalize to
   arbitrary taste heterogeneity.

## Evidence that can remain as a comparison

The existing XGBoost, Random Forest, SHAP, latent-class, and Mixed Logit
analyses remain useful as comparison and boundary evidence. They should be
reported after the main method question, with their synthetic nature and
five-replication precision limits stated next to the results. The current
Word-draft numbers are not interchangeable with the locked repository outputs.

## Release gate

The Word manuscript is ready for a full rewrite only after one of these paths
is selected:

- **Method path:** collect or obtain a purpose-built paired-task supplement,
  preregister the transformation and order rule, and add the empirical test;
  or
- **Simulation path:** present the candidate-preserving randomization test as
  a simulation method paper, state that the empirical instrument is missing,
  and reduce the claim to finite-sample task-design and randomization evidence.

