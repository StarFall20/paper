# Four-cell factorial repair audit

## Question

The exact paired-task benchmark localised omitted terms imperfectly. A natural
repair is to randomise four tasks for the same base profiles: no shift, a time
shift, a cost shift, and both shifts. The proposed mixed contrast is

\[
  r_{TC}-r_T-r_C+r_0,
\]

where each \(r\) is the one-hot choice vector minus the candidate probability.
This contrast cancels a common additive shift under the candidate and appears
to isolate a time-by-cost interaction.

## Audit run

The benchmark uses 50 replications, 300 respondents, four nuisance tasks, and
199 respondent-level multiplier draws. The focal shifts are 40 time units and
100 cost units. The probability-scale mixed contrast produces:

| DGP | candidate | rejection rate |
|---|---|---:|
| additive | additive | 0.12 |
| nonlinear time main effect | additive | 0.02 |
| cost threshold main effect | additive | 0.10 |
| interaction | additive | 0.28 |
| interaction | interaction repair | 0.02 |
| interaction + nonlinear time main effect | additive | 0.86 |
| random cost sensitivity | additive | 0.04 |

The repair reduces rejection for a pure interaction. It does not provide
clean interaction localisation when a nonlinear main effect is present. The
choice-probability map is nonlinear, so a mixed difference of probabilities can
remain nonzero even when the underlying utility component is additive in the
two attributes. The result is a design boundary, not evidence for a stronger
interaction test.

## Decision

The four-cell contrast is excluded from the paper's primary innovation claim.
It remains an audit artifact that prevents the manuscript from claiming
term-level localisation without a separate probability-scale identification
argument. The primary candidate remains the first-order model-equivalent
task-pair test. Its claim is detection of a violation of candidate invariance;
it does not identify a unique omitted functional form.

The implementation is `analysis/factorial_equivalence_contrast.py`, and the
full output is `results/factorial_equivalence_contrast.csv`.
