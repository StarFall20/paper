# Nearest general existence theorem: applicability audit

10 November 2026. Primary source inspected: [Koriyama and Bellec2025, Assumptions2.4–2.5, Theorem2.6 and AppendixA](https://arxiv.org/html/2501.03163v3), [published record](https://proceedings.mlr.press/v267/koriyama25a.html).

Their theorem supplies a general Gaussian single-index existence threshold. Its assumptions require every row loss to be strictly convex and the absolute rowwise infimum to have finite expectation. The generic claim of an existence phase transition for convex M-estimation is prior work. The present argument uses the same established random-subspace machinery.

The corrected labelled row in m Q_λ is

l_L(u;Y,g)=(1−λ)log(1+exp(u))−(Y−λg)u.

For every strictly interior g and positive λ,

Y=0: l_L(u)→−∞ as u→−∞,
Y=1: l_L(u)→−∞ as u→+∞.

Its rowwise infimum is −∞. At λ=1 it is affine. Thus the inspected assumptions do not apply directly; merely encoding labelled/auxiliary row types in the response does not repair the failure. The auxiliary row is λ/τ times soft logistic loss and is coercive. Aggregate finite estimability depends on compensation across rows with unbounded labelled losses.

This algebra establishes an applicability distinction from the cited theorem. It does not establish priority over all general convex Gaussian optimization results. The candidate increment remains the explicit corrected-objective boundary and consequences such as the [matched-risk/covariance proposition](matched_moment_existence_proposition.md). The paper must cite the general theorem and state this limited distinction directly.

Two broader CGMT sources were also inspected. [Loureiro et al.2021, AppendixB.1 assumptionA3](https://arxiv.org/html/2102.08127v3) assumes a coercive total cost for its estimator characterization; that assumption does not itself supply the existence boundary under study. [Thrampoulidis, Abbasi and Hassibi2016, §2.2 and Theorem3.1](https://arxiv.org/html/1601.06233) uses a residual-loss model with a finite attained loss minimum, independent additive noise and a scalar saddle minimizer. Its CGMT reduction is prior machinery, while a direct specialization to the present unbounded labelled rows has not been shown.

This is a bounded applicability comparison with the inspected versions. It excludes neither all extensions of their techniques nor an equivalent theorem elsewhere. No claim to a new general conic theorem, a complete error law for the estimator or a priority certificate follows from this audit.
