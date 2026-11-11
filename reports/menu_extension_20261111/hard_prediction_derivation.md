# Hard predictions and the number of alternatives

11 November 2026. This addendum investigates a specific difference between binary and larger choice menus within the same compensated objective. It extends the candidate derivation to predictions on simplex faces. The main manuscript has not yet incorporated this result. A completed numerical illustration will not establish priority or sufficient practical importance.

## The scope that must change

The previous menu derivation assumed that every predicted category probability was positive. Published prediction-assisted categorical procedures also use hard category predictions. Fan et al. (2024, Section 3.2 and Appendix A.3) give multiPPI++ for categorical regression. Ye and Yoganarasimhan (2026, Appendix J.2, Example J.4, and J.3) explicitly include common-coefficient choice logit. The prediction correction and its application to MNL are established. The candidate increment concerns finite-solution boundaries with a finite auxiliary ratio.

Sources inspected: [Fan et al., published paper](https://soc.washington.edu/sites/soc/files/documents/research/ppi_colm.pdf); [Ye and Yoganarasimhan, Appendix J](https://arxiv.org/html/2604.17267v1). The former model uses class-specific coefficient blocks; the latter includes the same choice-model coefficient structure used here. Their inspected constructions must be attributed even though their displayed inference and allocation results do not themselves verify the candidate proportional-dimensional existence formula.

Retain all Gaussian profile, finite-signal, independent-sample and frozen-predictor assumptions of `menu_boundary_derivation.md`. Replace strict interiority by

$$g\in\Delta_J\quad\hbox{almost surely},\qquad a_*:=E\|g-e_Y\|^2>0.$$

The artificial Gaussian projection normal $H$ remains independent of $(S,Y,g)$. The remaining attributes also remain independent conditional on the stated low-dimensional variables. A hard predictor $g=e_Z$ with a positive error probability satisfies $a_*=2\Pr(Z\ne Y)>0$. Perfect hard labels have $a_*=0$ and are handled by the exact weighted-likelihood reduction below.

## Why simplex faces preserve the boundary calculation

For each fixed positive power, retain the same residual sets

$$D_L=(1-\lambda)\Delta_J-e_Y+\lambda g,\qquad D_U=(\lambda/\tau)(\Delta_J-g).$$

Their labelled minimum norm is still $\lambda\|g-e_Y\|$. Every unchosen component is nonnegative and at least $\lambda g_j$; the chosen coordinate is minus their sum. Increasing any unchosen component increases the squared norm. Thus the minimum occurs at the translated vertex.

With probability tending to one, at least one labelled row has $g_Y<1$. Its chosen residual coordinate is bounded above by the strictly negative number $-\lambda(1-g_Y)$ before common scaling. This coordinate bounds the scale on any bounded nonzero residual sequence and makes the finite cone closed and pointed. This qualification replaces the previous claim that every labelled row supplies a strictly negative anchor. The probability that no such row occurs tends to zero, as $a_*>0$.

The low-dimensional support identity is unchanged:

$$\rho(\eta)=E[\max_j(-h_{\eta,j})+h_{\eta,Y}].$$

It uses cancellation of the two prediction expectations and boundedness of the simplex, not strict positivity of its coordinates. Finite true utility signal keeps all true choice probabilities positive. Uniform positivity on the nuisance unit sphere still gives the relative-interior qualification and bounded minimizing multipliers. On any compact multiplier set the scale bound now uses $a_*>0$. The Gaussian and signal envelopes and the unit-power contrast-rank bound are unchanged. These observations extend the original candidate finite-power and positive-power-limit arguments to this prediction class. They do not extend the Gaussian design assumptions.

## A strict-gap condition that depends on menu size

Let $C_Y=\operatorname{cone}\{e_j-e_Y:j\ne Y\}$, $w=H-h_\eta$ and $p_Y=\Pi_{C_Y}w$. The original derivative identity remains valid for boundary-simplex predictions:

$$\tfrac12\partial_v\Psi_\infty(\eta,0+)=E[\max_jp_{Y,j}-g^Tp_Y]. \tag{H1}$$

The integrand is nonnegative for every simplex vector. Its strict positivity requires separate consideration.

**Binary choice.** Write $p_Y=c(e_{1-Y}-e_Y)$ with $c\geq0$. Then

$$\max_jp_{Y,j}-g^Tp_Y=2c\,g_Y. \tag{H2}$$

Conditional on any finite $(S,Y,g)$, the Gaussian normal gives $c>0$ with positive probability. Thus the derivative in (H1) is positive for every finite multiplier exactly when $\Pr(g_Y>0)>0$. If $g=e_{1-Y}$ almost surely, the derivative is zero for every multiplier. Concavity in $v\geq0$ puts the maximum at $v=0$, giving $\kappa_\infty=\kappa_0$. The previously proved strict gap for interior predictions cannot be extended to every binary hard predictor.

**At least three alternatives.** The derivative is positive for every finite multiplier for any simplex-valued predictor satisfying the stated independence conditions. To see this, fix $(S,Y,g)$. If $g_Y>0$, any nonzero tangent projection makes the integrand strictly positive. If $g_Y=0$, choose an unchosen category $a$ with $g_a>0$ and another unchosen category $b\ne a$. There is an open Gaussian projection region in which

$$p_Y=c(e_b-e_Y),\qquad c>0.$$

For example, the normal component has equal chosen and $b$ coordinates, with every other coordinate strictly smaller. Strict inactive inequalities and a positive ray coefficient describe a full-dimensional projection cell in the contrast space. Gaussian full support gives it positive conditional probability. On that cell the integrand is $c(1-g_b)\geq c g_a>0$. This proves the claim without requiring any positive prediction accuracy.

The common compact multiplier and scale argument gives an attained limiting multiplier. Evaluate (H1) at that minimizer: its positive-scale supremum strictly exceeds its zero-scale value, which is at least $\kappa_0$. Consequently the extended candidate law gives $\kappa_\infty>\kappa_0$ for $J\geq3$. Under the same law, the binary strict gap holds precisely under the condition above. All statements concern strict sides of the boundary; the equality case in the dimension ratio is unspecified.

## Perfect predictions and the established variance reference

If $a_*=0$, then $g=e_Y$ almost surely. For $0<\lambda<1$, the objective reduces exactly to

$$Q_\lambda=(1-\lambda)L_m(Y)+\lambda L_N(Y). \tag{H3}$$

Every row has positive weight. Its existence event is ordinary pooled separation, independent of those positive weights. With $\kappa_0$ denoting the ordinary per-menu boundary, the boundary is $(1+\tau)\kappa_0$. At $\lambda=1$ only the auxiliary likelihood remains, giving $\tau\kappa_0$; at zero it is $\kappa_0$. This is an exact ordinary-likelihood reduction, not a new prediction-assisted estimator. It also supplies a separate treatment of the excluded zero-error endpoint. Every sequence $0<\lambda_m<1$ has the same pooled finite-sample event, regardless of its rate of approach to zero.

For an imperfect hard proxy with symmetric error $\epsilon$, the established PPI++ variance rule must use the proxy's actual score covariance. For one common Gaussian orthogonal slope in the fixed-dimensional model, let $\mu$ be the true menu probabilities and define

$$R=E(1-\|\mu\|^2),\qquad a=1-\epsilon J/(J-1),$$

$$K=aR,\qquad V_g=2\epsilon+(2a-1)R.$$

Conditional summation of $(Y,Z)$ gives the human score variance $R$, its covariance with the proxy score $K$ and the proxy score variance $V_g$. The ordinary fixed-dimensional sandwich coordinate is

$$\{R-2\lambda K+(1+1/\tau)\lambda^2V_g\}/R^2.$$

Its minimum over $[0,1]$ is the existing control-variate choice

$$\lambda_{\rm var}=\operatorname{clip}_{[0,1]}\{\tau K/[(1+\tau)V_g]\}. \tag{H4}$$

This scalar reference is for the stated orthogonal coefficient, not an assertion of optimal high-dimensional risk or optimal variance for every utility coefficient. The posterior-oracle value $\tau/(1+\tau)$ generally differs. At $\epsilon=1$, $K<0$ and (H4) selects exactly zero. This makes the hard-prediction strict-gap calculation relevant to a zero endpoint of an established criterion; it does not imply that a small positive power has lower coefficient error.

## Interpretation and validation target

This is a menu-geometry statement about an unpenalized corrected objective. Zero-accuracy predictions may contain information that a different calibration or relabeling rule could use. No claim is made that retaining a known wrong predictor is a good estimation strategy, that finite coefficients are accurate, or that the Gaussian limit validates the current fixed-menu questionnaire.

The recorded population plan tests hard one-hot proxies at error rates 0.25 and 1, two menu sizes, two finite signals, two auxiliary ratios, two integration resolutions and two scramble seeds. Always-wrong binary predictions must reproduce the ordinary boundary; larger menus must satisfy the strict derivative condition. The complete values, integration variability and any failures will be retained. The existing posterior-prediction grid remains unchanged and represents a different prediction rule.
