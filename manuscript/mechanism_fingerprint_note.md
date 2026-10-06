# Mechanism-Fingerprint Choice Audit

## Candidate contribution

The candidate-preserving fibre test establishes whether a utility basis is
compatible with a common attribute transformation. The mechanism-fingerprint
extension asks a different question: which class of invariance fails? It uses
three held-out relations under one frozen additive candidate:

- a common shift of every alternative's attributes;
- a counterbalanced permutation of displayed alternatives, with outcomes
  mapped back to alternative identity; and
- a repeated task after a neutral filler that changes its sequence position.

The candidate predicts equal identity probabilities for all three relations.
The first relation is sensitive to functional form, the second to presentation
or position effects, and the third to sequence dependence. The method estimates
one respondent-cluster contrast per axis and uses a shared-sign maxT reference
to control the familywise rejection rate across axes. The resulting signature
is a diagnostic class, not a structural identification theorem.

## Simulation gate

The benchmark uses 50 replications, 300 respondents, eight nuisance tasks, and
199 respondent-cluster randomizations. The fibre shift is `(35,45)`; the
presentation permutation swaps alternatives 0 and 1; the sequence intervention
places the repeated task after a neutral filler and adds a 1.20 inertia bonus
to alternative 0 only under the inertia conditions. The candidate is fitted on
nuisance tasks from development respondents and frozen for held-out contrasts.

| condition | fibre reject | presentation reject | sequence reject | dominant signature |
|---|---:|---:|---:|---|
| additive | 0.02 | 0.00 | 0.00 | none (49/50) |
| random price sensitivity | 0.02 | 0.02 | 0.02 | none (47/50) |
| nonlinear utility | 0.82 | 0.00 | 0.04 | fibre only (39/50) |
| position bias | 0.02 | 1.00 | 0.00 | presentation only (49/50) |
| inertia | 0.02 | 0.00 | 1.00 | sequence only (49/50) |
| nonlinear + position | 0.78 | 0.92 | 0.04 | fibre and presentation (33/50) |
| position + inertia | 0.02 | 1.00 | 1.00 | presentation and sequence (49/50) |
| nonlinear + inertia | 0.82 | 0.00 | 0.60 | mixed or fibre only |

The null size is controlled after maxT adjustment. A common random price
coefficient does not activate the fibre relation, which is the required linear
heterogeneity boundary. Single-mechanism conditions produce the intended
dominant signatures. Combined conditions are reported as mixed signatures or
remain unresolved; the diagnostic never assigns a unique behavioural process
from a mixed pattern.

## Novelty boundary

Ordering effects, non-trading, inertia, and MNL misspecification tests each have
substantial prior literatures. The proposed object is the cross-axis design and
its estimand: a predeclared vector of candidate-implied invariance violations
with a calibrated ambiguity outcome. It does not add a new inertia model or a
new position-bias parameterization. It does not claim that a signature proves a
single causal mechanism. This boundary is essential for a credible JOCM claim.

The empirical supplement must use a forced-choice block or define opt-out
attributes for the common shift, counterbalance alternative order within
respondent, separate the repeated task with a filler, and preregister the
maxT rule. Without that instrument, the result remains a simulation-level
method contribution.

The implementation is `analysis/mechanism_fingerprint_benchmark.py`; the
frozen output is `results/mechanism_fingerprint_benchmark.csv`.

