# Round 30: direct oracle audit of the policy-path idea

## Why this audit was required

The earlier policy-path statistic compared an additive MNL with a structured
MNL. That comparison is informative about model sensitivity, but it is not a
certificate of prediction error. Two specifications can share an omitted term
and agree with each other while both miss the same policy response. This audit
keeps model disagreement as a diagnostic and measures its limit directly.

## Design

The simulation uses the same three-alternative smart-device data-generating
process and the nine predeclared quality/price interventions
`{-0.6, 0, 0.6}^2`. The additive and structured libraries were corrected to
include the opt-out-by-income term that is present in the generator. Each
model is estimated on 60% of respondents with converged L-BFGS MNL fits; the
validation fold contains 20% of respondents. The exact latent price
coefficient is recovered from the simulated utility for the heterogeneity
condition, so the oracle counterfactual probabilities are known. The reported
gap is the mean absolute probability gap across alternatives, tasks, people,
and path points; it is not an unnormalised L1 norm.

## Results

The table reports means over 24 independent replications. `D` is model
disagreement, `E0` and `ES` are the additive and structured model errors
against the oracle, and `R0` and `RS` are anchored response errors. Anchoring
subtracts the model and oracle probability at the zero intervention before
comparing path changes.

| condition | D | E0 | ES | R0 | RS |
|---|---:|---:|---:|---:|---:|
| additive | 0.01036 | 0.01156 | 0.01587 | 0.00424 | 0.00847 |
| nonlinear | 0.09821 | 0.09874 | 0.01387 | 0.06321 | 0.00823 |
| threshold | 0.01857 | 0.02047 | 0.01520 | 0.01767 | 0.00833 |
| interaction | 0.09020 | 0.09072 | 0.01506 | 0.03865 | 0.00786 |
| heterogeneity | 0.00978 | 0.01419 | 0.01710 | 0.00614 | 0.00840 |
| nonlinear + threshold | 0.09542 | 0.09596 | 0.01359 | 0.06496 | 0.00885 |
| nonlinear + interaction | 0.12782 | 0.12835 | 0.01352 | 0.07239 | 0.00850 |
| combined | 0.12188 | 0.12219 | 0.01439 | 0.07165 | 0.00939 |

The direct benchmark supports a narrower claim. A large model gap tracks
observable curvature and interactions in this generator, while heterogeneity
produces a small model gap and a modest population oracle gap after integrating the random coefficient. The model gap therefore measures sensitivity between the declared specifications; it cannot certify that either specification is correct. In the pooled mechanism benchmark its correlation with additive oracle error is 0.996, while its correlation with the smaller of the two oracle errors is 0.069. The exact common-mode witness remains the stronger limitation because it gives D=0 with nonzero off-support oracle error.

## Exact common-mode witness

For observed support `q in {-1, +1}`, let the omitted policy term be
`h(q)=q^2-1`. It is exactly zero at every observed support point, so two
models can agree on all observed utilities and have `D=0`. At the declared
interventions `q=-0.5, 0, 0.5`, the oracle term is `-0.75, -1, -0.75`.
The witness is saved in `results/common_mode_witness.csv`. It proves that a
model-to-model gap is not an upper bound on counterfactual error.

## Consequence for the innovation claim

The manuscript should present the policy-path object as a calibrated
counterfactual sensitivity audit with an explicit common-mode failure test.
It should not call the model gap a recoverability certificate without a
declared basis, support condition, and oracle-validating design. A strong
contribution remains possible if the paper makes the audit operational:
predeclare the policy path, report the two model errors when simulation is
available, include the exact common-mode witness, and state the structural
support limitation. The current evidence is not sufficient for a global
recoverability theorem.

## Reproducibility

Code: `analysis/policy_oracle_audit.py`.

Outputs: `results/policy_oracle_audit.csv` and
`results/common_mode_witness.csv`.

The 24-replication output is a corrected audit, not a replacement for an
external causal validation. The Swissmetro result remains descriptive because
its public data do not contain randomized intervention outcomes.

## n=800 replication

A second 24-replication run with 800 respondents reduces the additive-null
model gap from 0.01036 to 0.00778. The nonlinear and interaction gaps remain
0.09672 and 0.08954, while the combined condition remains 0.12182. The
threshold gap is 0.01995. The same ordering appears in the oracle errors, so
the direct sensitivity result is not a small-sample artifact. The n=800 raw
output is saved in `results/policy_oracle_audit_n800.csv`.
