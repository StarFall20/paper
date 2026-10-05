# Model-equivalent choice-pair test: prototype note

## Purpose

This experiment tests the selected independent innovation from
`manuscript/independent_idea_search.md`. It asks whether two binary choice
tasks with the same candidate-model utility difference generate the same
choice share. The attribute levels differ across the tasks, so an omitted
nonlinear response can change the true probability while the candidate model
continues to predict equality.

## Design

The candidate utility difference is

\[
\Delta V=0.8(q_A-q_B)-0.6(p_A-p_B).
\]

The two task pairs are `(q_A,p_A,q_B,p_B)=(2,1,1,1)` and
`(4,1,3,1)`. Both have candidate \(\Delta V=0.8\). The nonlinear stress
condition adds \(0.35q^2\) to the true utility. The null uses the additive
candidate model. Each replication draws 300 choices for each pair and uses a
parametric bootstrap under the equality restriction.

## Prototype result

The 500-replication run gives:

| DGP | mean equivalence gap | rejection rate at 5% |
|---|---:|---:|
| additive null | -0.0016 | 0.06 |
| omitted quadratic term | 0.0990 | 0.986 |

The raw replication file is `results/equivalent_pair_test.csv`; the entry
point is `analysis/equivalent_pair_test.py`.

## Swissmetro feasibility audit

The public Biogeme Swissmetro file contains 10,728 rows for 1,192 respondent
IDs, with nine choice situations per ID. An exact scan of the full
alternative-attribute profile finds only eight repeated profiles within a
respondent, concentrated in two IDs. Those rows look like duplicated survey
records rather than a balanced matched-task block. The dataset therefore does
not supply enough clean paired tasks for an empirical equivalence test. The
main-paper pivot requires a new paired-task supplement or a different dataset;
the ordinary Swissmetro application can validate transfer of the baseline
specification workflow only.

An observational fallback is possible: freeze a candidate model on a separate
development sample, match within-respondent tasks by the candidate's two
utility-difference coordinates, and test choice-share equality on held-out
tasks. In the public file, the audit finds 26, 85, 535, and 1,924 within-ID
pairs at coordinate tolerances 0.01, 0.02, 0.05, and 0.10, respectively. The
small tolerances give little power; the larger tolerances weaken the equality
restriction. `analysis/audit_swissmetro_pairs.py` records this trade-off. The
fallback is a separate, calibrated observational test and cannot be treated as
equivalent to randomized paired-task data.

The null rejection rate is close to the declared size and the nonlinear
condition is detected with high power in this deliberately simple design. The
result supports feasibility, not a submission claim. The next experiments
must estimate the candidate basis from a separate development sample, test
threshold and interaction omissions, vary pair separation and sample size,
and include latent heterogeneity and process violations.

## Required boundary checks

1. A correctly specified flexible basis should remove the violation.
2. A random-coefficient DGP should be reported separately because a violation
   may reflect heterogeneity rather than a missing functional-form term.
3. Pairs must be randomized across respondents or tasks to avoid order effects.
4. The bootstrap must preserve respondent clustering when each respondent sees
   both members of a pair.
5. An empirical application requires matched tasks in the instrument or a
   paired-task supplement. LPMC and the public Swissmetro file cannot establish
   this property without such a design.
