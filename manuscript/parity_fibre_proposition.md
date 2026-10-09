# Local parity proposition for candidate-preserving fibres

Let \(T_t\) be a smooth path of task versions that preserves the complete
candidate menu vector \(\phi\). Let \(p(t)\) denote the multinomial choice
probability vector under a local data-generating process

\[
V_j(t)=V_{0j}+\eta q_j(t),
\qquad
P_j(t)=\operatorname{softmax}_j\{V(t)/\sigma(t)\}.
\]

Assume the declared processing mechanism is reflection-symmetric,
\(\sigma(t)=\sigma(-t)\), and the candidate-preserving transformation leaves
\(V_{0j}\) unchanged. More generally, let \(g(t)\) be a predeclared nuisance
signature containing the scale, distance, and display-complexity features used
by the processing model, and require \(g(t)=g(-t)\). Suppose the omitted
direction has an odd component, \(q_j(-t)=-q_j(t)\).

At \(\eta=0\), \(P(t)=P(-t)\). The first-order response satisfies

\[
\left.\frac{\partial P(t)}{\partial\eta}\right|_{\eta=0}
 =J\!\left(V_0/\sigma(t)\right)q(t)/\sigma(t),
\]

where \(J\) is the softmax Jacobian. Because both the Jacobian and the scale
factor are even in \(t\), the derivative is odd. Consequently,

\[
\frac{P(t)-P(-t)}{2}
 =\eta\,J\!\left(V_0/\sigma(t)\right)q(t)/\sigma(t)
   +o(\eta).
\]

The central parity contrast removes every declared nuisance mechanism whose
signature is held fixed to first order and retains the odd omitted direction.
The even contrast
\([P(t)+P(-t)]/2\) is reported as a diagnostic for scale or comparison
complexity.

The result is a local statement. An odd order, framing, or attention mechanism
can survive the contrast. A candidate-omitted term with only an even response
can be invisible to the primary parity statistic. The design must report both
parities and predeclare the nuisance library.

For a multinomial menu, preservation of one pairwise gap is insufficient. The
path must keep every component of \(\phi\) fixed, and inference must use the
covariance of the full probability vector or respondent-cluster robust
inference. The binary zero-gap construction is a special case of the same
logic, not a substitute for the multinomial condition.

## Moderate-utility extension

The parity implication does not require a logit link. Suppose the binary
choice probability belongs to a moderate-utility class,

\[
P(A\succ B\mid t)=F\!\left(\frac{\Delta V(t)}{D(t)}\right),
\]

where (F) is a smooth increasing link and (D(t)) is a product-difference
index. On a candidate-preserving reflection, let
\(\Delta V(t)=\Delta V_0+\eta q(t)\), (q(-t)=-q(t)), and
\(D(t)=D(-t)). Then the candidate restriction gives
\(P(t)=P(-t)), while a local omitted direction gives

\[
\frac{P(t)-P(-t)}{2}
 = \eta\,\frac{F'\!\left(\Delta V_0/D(t)\right)}{D(t)}q(t)+o(\eta).
\]

Thus the central contrast is a link-robust directional test within the
moderate-utility class. It tests the odd component of the utility gap after an
even distance or scale mechanism has been held fixed. The result does not
claim that all complexity is even: an order, wording, or attention mechanism
with an odd response remains a failure boundary.
