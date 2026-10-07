# Conditional sufficiency, finite fibres, and exposure

## 1. Null and design measure

X is a complete raw menu, including its fixed labels and number of alternatives.
Z = phi(X) is a preregistered candidate menu summary. A known feasible design
measure pi supplies the distribution of X. Comparisons condition on fixed
respondent covariates and presentation controls when these enter the estimand.
Under the primary null,

    P(Y=j | X=x) = q_j(phi(x)),     j=1,...,J.

The vector q(z) is unrestricted except for positivity and summing to one.
This is predictive sufficiency, not a Fisher--Neyman sufficient statistic for
an unknown parameter. It places no linear-index restriction on q.

For finite DCE support the nuisance space has (J-1) free probabilities per
supported summary value. With continuously varying summaries it is a function
space. Discrete attributes do not prevent local probability submodels:
locality below refers to the scalar departure amplitude eta.

## 2. Binary local information

Consider the local submodel

    logit P_eta(Y=1 | X=x) = logit q(phi(x)) + eta h(x).

At eta=0, the departure score is (Y-q(Z)) h(X). Scores for the unrestricted
null response have the form (Y-q(Z)) a(Z), for arbitrary a. Since q(Z)(1-q(Z))
is constant within each fibre, the score projection onto this nuisance space is

    (Y-q(Z)) E_pi[h(X) | Z].

The efficient score is therefore

    s_eff = (Y-q(Z)) {h(X)-E_pi[h(X) | Z]}.

Its variance is

    I_pi(h) = E_pi[q(Z)(1-q(Z)) Var_pi(h(X) | Z)].

For eta=c/sqrt(N), independent observations, and the regular efficient score
test, the limiting noncentrality parameter is c^2 I_pi(h). For a small fixed
eta, N eta^2 I_pi(h) is its local planning approximation. Clustered task
responses require cluster information rather than multiplying independent
task information.

If all q lie strictly between zero and one, I_pi(h)=0 exactly when h is
constant on each fibre given positive design mass. A direction with I_pi(h)>0
has local information; an effect size and sampling plan are needed for power.

## 3. Multinomial probability tangent

Consider p_eta(x)=q(z)+eta a_z h(x)+o(eta), where sum_j a_zj=0 and every
q_j(z)>0. The score for eta is h(X) a_(Z,Y)/q_Y(Z).
Projecting on the unrestricted multinomial q(Z) scores centres h within Z.
Hence

    s_eff = {h(X)-E_pi[h(X)|Z]} a_(Z,Y)/q_Y(Z),
    kappa_z = sum_j a_zj^2 / q_j(z),
    I_pi(h) = E_pi[kappa_Z Var_pi(h(X)|Z)].

For any full independent contrast matrix C, the same kappa equals

    (C a_z)' [C {diag(q)-q q'} C']^(-1) (C a_z).

This is an efficient information identity, not the covariance of an
uncentred signed one-hot estimator. The prototype multinomial script uses
Z_i=2 R_i C Y_i. Its null covariance is 4 C diag(q) C'; this is generally
NOT a scalar multiple of C{diag(q)-q q'}C'. Each estimator must use its own
covariance. Both positive metrics give the same zero-signal condition.

## 4. Multinomial utility dictionary

Let H(x) be a J by r matrix of declared menu utility departures and consider

    p_eta(x) = softmax(log q(z) + H(x) eta).

At the null, S_z=diag(q(z))-q(z)q(z)' is the choice-probability Jacobian.
Set H_c(x)=H(x)-E_pi[H(X)|Z=z]. The efficient information matrix is

    K_pi = E_pi[H_c(X)' S_Z H_c(X)].

For any coefficient direction u,

    u' K_pi u = 0

exactly when H_c(x)u is a common shift of all J alternatives on every
positive-mass profile. Common shifts do not alter choice probabilities.
After a fixed reference-alternative normalization this is equivalently
zero within-fibre variation in all menu utility differences.

This formulation preserves the full menu. A scalar A--B gap is sufficient
only for a binary task; in a larger menu all J-1 independent gaps are needed.

## 5. Finite combinatorial certificate

Fix a finite feasible menu universe X_all and a declared dictionary H with
fixed scaling. Within each fibre choose a reference menu x_g0. Define
QH(x) by subtracting the utility of reference alternative J from each of
the other alternatives. Stack the rows of

    QH(x_gk) - QH(x_g0),       k=1,...,K_g-1,

over all feasible fibres. Call the resulting matrix L_all. Its kernel contains
exactly the declared directions that preserve all within-fibre menu gaps on
the full feasible support. Compute its rank, singular values, and a basis of
the kernel. These are a checkable structural certificate.

For a selected support S, compute L_S using only its menus. Because S is a
subset of the declared feasible universe,

    ker(L_all) is contained in ker(L_S).

The quotient-dimension loss rank(L_all)-rank(L_S) measures directions lost
through design selection. If ranks agree, the kernels agree and the selected
support exposes every direction that the full universe can expose. This
statement assumes the same summary, labels, and departure dictionary in both.

The weighted binary version for a residual feature matrix R_g, fibre mass
alpha_g, conditional weights w_g, W_g=diag(w_g), and
T_g=I-1 w_g' is

    K_pi = sum_g alpha_g q_g(1-q_g) R_g' T_g' W_g T_g R_g.

Positive weights make its kernel agree with the corresponding contrast matrix.
Its eigenvalues depend on design weights, baseline probabilities, dictionary
scaling, and response covariance. Rank alone does not provide sample size.

## 6. Three distinct zero-signal cases

1. Intrinsic null direction: a departure is a function of phi on the full
   feasible support (or is a common utility shift). It does not violate
   sufficiency, and is removed before counting target residual directions.
2. Feasible-support restriction: a hypothesized mechanism varies on a larger
   ambient space but is constant on every feasible fibre. The audit has no
   content for that mechanism within the declared universe.
3. Design-induced loss: L_all u is nonzero but L_S u is zero. This is a
   substantive violation that the selected design cannot reveal.

Only the third case establishes an avoidable structural blind spot for the
specified feasible universe. Increasing respondent count cannot repair it.
Adding feasible menus repairs u when the enlarged L_S maps u to a nonzero
contrast. Exact rank calculations on finite rational features require no
attribute derivatives.

## 7. Randomization and process interpretation

A frozen within-fibre assignment law supplies a conditional-randomization
test of the primary null without fitting q. Under iid respondents, or a joint
block null that conditions on task history and respondent covariates,
resampling the recorded within-fibre assignment leaves the null distribution
invariant. A sharp causal no-effect null is a different claim; the paper uses
the stated distributional sufficiency null.

Framing or scale variation beyond phi is itself a sufficiency violation.
Projecting it away changes the target to utility departures modulo a declared
process span. The nuisance-orthogonal extension therefore requires a separate
local submodel and calibration argument. It cannot be presented as the same
unrestricted sufficiency null with exact finite-sample process immunity.
