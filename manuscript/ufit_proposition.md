# UFIT proposition and inference conditions

## Proposition: candidate-preserving fibres

Let \(b_j(x)\in\mathbb{R}^p\) be the candidate basis for alternative \(j\),
and let

\[
 p_j(x;\beta)=\frac{\exp\{b_j(x)'\beta\}}
 {\sum_{k=1}^J\exp\{b_k(x)'\beta\}}.
\]

For two tasks \(x\) and \(Tx\), the following statements are equivalent:

1. \(p_j(Tx;\beta)=p_j(x;\beta)\) for every alternative \(j\) and every
   coefficient vector \(\beta\in\mathbb{R}^p\);
2. \(b_j(Tx)-b_k(Tx)=b_j(x)-b_k(x)\) for every pair \(j,k\).

### Proof

The second statement makes every utility vector differ by the same additive
constant, so the softmax probabilities are equal. For the reverse direction,
choose any pair \(j,k\). Equality of the two softmax vectors implies equality
of their odds, hence

\[
 [b_j(Tx)-b_k(Tx)]'\beta
 = [b_j(x)-b_k(x)]'\beta
\]

for every \(\beta\). Equality for all coefficient vectors implies equality of
the two basis-difference vectors.

The proposition is a characterization of the candidate restriction. It does
not assert that a non-rejection proves the candidate globally correct. The
empirical test samples a declared set of transformations \(T\), a declared
support of attributes, and a declared choice process.

## Test estimand

For a transformation \(T\), define

\[
 \Delta_j(T)=P^*(Y=j\mid x)-P^*(Y=j\mid Tx),
\]

where \(P^*\) is the data-generating choice process. The candidate null on the
tested fibre is \(\Delta(T)=0\). The test respondent must be assigned one
member of each pair by a predeclared Bernoulli randomization, with the
candidate frozen on development respondents. A respondent-cluster assignment
permutation then gives the reference distribution for the share-difference
statistic. If respondents answer both members, carryover and order must be
modelled or separately randomized; the simple reference distribution no longer
follows from exchangeability alone.

## Interpretation boundary

The alternative \(\Delta(T)\ne0\) establishes that the candidate basis is
insufficient for the tested intervention. It does not identify whether the
source is a nonlinear term, an interaction, scale variation, random taste,
presentation, or sequence dependence. Term labels require separate,
predeclared transformations and remain non-identifying when mechanisms
co-occur. The proposition therefore supports a basis-sufficiency audit, not a
universal misspecification theorem.
