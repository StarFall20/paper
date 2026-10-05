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
   paired-task supplement. LPMC cannot establish this property without such a
   design.

