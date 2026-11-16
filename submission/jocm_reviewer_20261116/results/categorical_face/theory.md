# The corrected path on a partially separated categorical design

13 November 2026. This is a finite-design consequence of the existing compensated choice objective. Classical separation faces and vanishing-regularization arguments supply the foundation. Gaussian attributes, independent rows, and a prediction trained independently of this realized design are unnecessary for this conditional deterministic statement.

## The human separation face

Write M_i=(2Y_i−1)X_i for the labelled signed rows, H for their mean logistic loss, and C={v:Mv≥0}. A row belongs to I+ when some v in C gives M_i v>0. The remaining rows form I0. Adding finitely many exposing directions gives a single v with M_+v≥1 and M_0v=0. Every vector in C has zero margin on I0.

These sets can be checked by the classical linear program maximizing sum(s_i), subject to Mv≥s and 0≤s≤1. Its optimum is |I+|. The retained primal has s_i=1 on I+ and zero on I0. A dual nonnegative vector u with M' u=0, u_+=0, and u_0≥1 proves that every neutral row has zero margin on all of C. This implements established facial separation, not a new certificate principle.

The neutral logistic loss H0 has a finite minimizer b0 modulo its design kernel. To see this, suppose M0w≥0 with a strict margin on one neutral row. Adding a sufficiently large simultaneous exposing direction on I+ would yield a member of C exposing that neutral row, a contradiction. Thus the neutral design has overlap on its identified row space. Its minimum h0 is also inf H: H≥H0≥h0, and H(b0+t v) tends to h0 along a simultaneous exposing direction.

## The same corrected recession selects the large coefficients

Let A=Q1 and Qλ=(1−λ)H+λA for frozen predictions. Its positively homogeneous recession is

$$a(v)=N^{-1}\sum_{U}(X_i'v)_+ + \{m^{-1}\sum_L(g_i-Y_i)X_i-N^{-1}\sum_Ug_iX_i\}'v.$$

The auxiliary softplus identity gives 0≤A(β)−a(β)≤log(2). The function a can be negative away from C. Assume a(v)>0 on C\{0}, and that the positive-curvature combined design has full column rank. Equivalently, one sufficiently small positive-power loss is coercive: on C its recession is λa, and compactness gives coercivity for all sufficiently small λ. The empirical support calculation can check the required positive-power coercivity directly.

Define the compact nonempty set

$$V_* = \operatorname{argmin}_v a(v)\quad\text{subject to}\quad M_+v\geq1,\quad M_0v=0.$$

For every sequence λ tending to zero through positive values, its finite unique minimizers satisfy

$$\operatorname{dist}\{\widehat\beta_\lambda/\log(1/\lambda),V_*\}\longrightarrow0.$$

This includes complete separation, quasi-complete separation, and overlap. With overlap and full labelled rank, C={0} and V*={0}. On a partially separated design, margins of at least one apply to I+ alone. Imposing them on every labelled row would incorrectly declare the selector infeasible.

Proof. Put L=log(1/λ). For feasible v, use β=Lv+b0. Its neutral loss equals h0, its separable loss is O(λ), and A(Lv+b0)=La(v)+O(1), by Lipschitz continuity and the bounded softplus remainder. Consequently [Qλ(Lv+b0)−(1−λ)h0]/(λL) tends to a(v).

For localization, let h(v)=m^{-1}sum(−M_i v)_+. Continuity, homogeneity and positivity of a on the compact unit slice of C give constants c>0 and K<∞ with a(v)≥c||v||−Kh(v) for all v. Indeed, a≥c near C on the unit sphere; h is bounded away from zero on its compact complement. As H≥h and H≥h0, for small λ,

$$Q_\lambda(\beta)-(1-\lambda)h_0\geq\lambda c\|\beta\|-\lambda K h_0.$$

Comparison with the preceding recovery point proves ||βhatλ||=O(L). Every scaled subsequential limit lies in C. A direction outside C has a negative human margin and a human loss growing at least linearly in L, contradicting the O(λL) excess. Thus it has M0v=0. If M_i v<1 for any i in I+, its positive labelled loss divided by λL diverges; the remaining neutral loss exceeds h0, while a(βhatλ)/L remains bounded. This contradicts the same recovery bound. Every limit is feasible for V*. Dropping the nonnegative excess human loss gives the lower bound a(v) for the rescaled criterion. The recovery bound for each minimizer of V* gives the reverse inequality. Positivity of a on C makes V* compact; all scaled coefficient limits therefore lie in V*.

The conclusion is set-valued. A solver returning one LP vertex does not establish a unique asymptotic coefficient ratio. The numerical audit measures distance to the whole near-optimal LP set, recording the objective tolerance explicitly.

## Consequences for probability and attribute interpretation

For any fixed released evaluation rows with signed margins T_i=(2Y_i^test−1)X_i^test, the exact softplus bounds give

$$0\leq R_{test}(\beta)/L-\operatorname{mean}_i(-T_i\beta/L)_+\leq\log(2)/L.$$

If V* is a singleton v*, the rescaled test log loss tends to mean(−T_i v*)_+. With a nonunique V*, each subsequential limit is the value at an attained limiting selector; uniqueness or a constant test functional on V* is needed for a single predicted limit. This evaluates the fixed released rows and makes no population validation claim.

Likewise, any coefficient ratio β_j/β_k tends to v*_j/v*_k when the selector is unique and v*_k≠0. At a nonunique selector, a stable finite coefficient ratio requires additional conditions. The formula is a conditional implication; anonymous numeric columns have no verified monetary unit in this audit. No willingness-to-pay estimate is reported.

## Attribution and role in the paper

Albert and Anderson's classical separation classification, Rosset, Zhu and Hastie's norm-regularization margin limit, and Ji and Telgarsky's maximal separable subset already establish the relevant general mechanisms. The calculation here uses the same corrected, potentially signed auxiliary recession and extends its previously strict-separation consequence to the realized categorical face. Its role is to connect the paper's global-attainment question with the recorded public coefficient paths. It is not an independent framework, estimator, or proof of journal-level originality.
