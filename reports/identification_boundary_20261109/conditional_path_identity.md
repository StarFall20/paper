# Classical conditional path identity and its limits

For person i, let stable utilities of unchanged alternatives j and k be a_i
and b_i at tasks t and s. The third alternative can change arbitrarily.
Write S_it and S_is for the respective conditional logit denominators.
If task errors are independent conditional on stable individual parameters,

\[
P_i(Y_t=j,Y_s=k)=\frac{e^{a_i+b_i}}{S_{it}S_{is}}
=P_i(Y_t=k,Y_s=j).
\]

Integration over any distribution of stable individual parameters preserves
the equality. Conditional on a record producing either discordant path, its
orientation has probability one half. With independent records the number of
forward paths, conditional on the discordant records, is exactly binomial.
The identity can be checked for every instrument-eligible pair; Bonferroni
controls the family without assuming the two tests are independent.

This is standard conditioning on a panel-logit likelihood. It is not a new
theorem, an assertion of unconditional IIA for mixed logit, or exchangeability
of all alternatives' errors. The panel likelihood assumptions are explained
in [Train (2009), Chapter 6, Section 6.7](https://eml.berkeley.edu/books/choice2nd/Ch06_p134-150.pdf).
The computational exact binomial test and Clopper–Pearson interval use the
[SciPy implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).

Serial dependence beyond the stable individual parameters, task-specific
utility shifts, changing noise scales, learning or framing can break the
identity. A rejection would not isolate any one cause. Non-rejection does not
verify the model. Independent records and the null's conditional error law
are substantive conditions; no field-randomization claim is made.

For the separate index sensitivity calculation, D is the full-rank 8-by-8
within-version conditional utility design and c selects G2 at CNY89 relative
to G1 at CNY89. Solving D' w = c yields w=(-2,1,1,0,1,-2,1,0).
Thus delta=w'e. If e=D theta+h and |h_t|<=epsilon, then
delta_structural lies in [w'e-8 epsilon,w'e+8 epsilon]. This is a sharp
classical box-constrained linear-contrast bound. It is conditional on the
index representation and does not convert mixture-level log odds into
individual utility or recover a causal rule effect from fixed task order.
