# Positive powers approaching zero

10 November2026. This develops the same corrected binary logit objective. The generic facts about separation,vanishing regularization and margin limits have substantial prior literature. The independent increment to assess is the explicit corrected-objective boundary and its zero-endpoint behavior.

## Fixed sample existence

Write H=L_m(Y), A=Q_1 and Q_λ=(1−λ)H+λA. Let h=H^∞ and a=A^∞ denote the recession functions. The human loss is nonnegative and h(d)≥0. The stacked design has full column rank; at unit power require full rank of the auxiliary curvature design when asserting uniqueness there. All predictions are fixed in this argument.

For0<α<λ≤1,

Q_α/α=Q_λ/λ+(1/α−1/λ)H.

If Q_λ is coercive,Q_α is coercive. Strict convexity under the stated rank condition gives a unique finite root. Thus finite existence is nested within positive powers. The endpoint0 can lose auxiliary curvature and needs its own separation check.

Let S={d:h(d)=0},the human weak-separation cone. A finite root exists for some positive power if and only if

a(d)>0 for every d∈S\{0}.

Necessity: on S,Q_λ^∞(d)=λa(d),which must be strictly positive for a coercive full-rank logit loss. Sufficiency: if S meets the unit sphere,compactness gives min_(S∩sphere)a=c>0. In a neighborhood of that compact set,a≥c/2. Off the neighborhood,h has a positive minimum and a is bounded below. A sufficiently small positive λ makes (1−λ)h+λa strictly positive on the whole unit sphere. If S={0},h is already strictly positive there. Proposition1 of the main manuscript then gives the finite root. This is a direct recession/support argument,not a new general separation theorem.

## Balanced oracle right limit

Use the balanced Gaussian oracle benchmark with error ε∈(0,1/2],auxiliary ratio τ>0 and p slopes with a fixed zero intercept. Put v=λt and f(a)=E(G−a)_+². The existing finite-power formula can be written

κ_c(λ)=1+τ−min_(v≥0){G(v)+R_λ(v)},

G(v)=(1−ε)f(−εv)+εf(−(1−ε)v)
     +τ{f((1−ε)v/τ)+f(εv/τ)},

R_λ(v)=(1−ε)f((λ^−1−1+ε)v)
       +εf((λ^−1−ε)v).

For every v>0,the positive remainder strictly increases with λ. The finite-power minimizer is positive because the derivative at v=0 is −2φ(0)/λ. Hence κ_c strictly decreases with λ on(0,1]. A numerical table can saturate within floating-point precision; it must not report exact equality from that saturation.

The function G is coercive,its derivative at zero is2φ(0){2ε(1−ε)−1}<0,and G(0)=1/2+τ. Its minimizer is positive and finite. Since R_λ≥0 and R_λ(v)→0 at each positive v,

κ_c(0+)=1+τ−min_(v≥0)G(v)>1/2=κ_c(0).

The limit follows by bounding the finite-power minimum below by min G and evaluating it at a minimizer of G for the reverse bound. This is an iterated limit: first the proportional-dimension limit at each fixed positive power,then power tending to zero. It does not prove a boundary for an arbitrary sequence λ_m→0 jointly with sample size.

At τ=1,ε=0.25,the right limit is approximately0.664488,compared with0.602855 at power0.5,0.355439 at unit power and0.5 at exactly zero. Error0.5 gives an uninformative prediction;its corresponding phenomenon concerns regularization/existence,not additional choice information.

## Fixed sample coefficient path

If H has no finite stationary point,and Q_λ has finite minimizers β_λ for every sufficiently small positive λ,then ||β_λ||→∞. A bounded subsequence would converge to β*,and continuity of ∇Q_λ=(1−λ)∇H+λ∇A would imply ∇H(β*)=0,a contradiction.

A more explicit path follows under strict human separation and the positive recession condition on S above. Define the compact nonempty minimizing set

V*=argmin a(v) subject to (2Y_i−1)X_i'v≥1 for all labelled i.

Then dist(β_λ/log(1/λ),V*)→0. If V* is a singleton,the normalized coefficient converges to that point. The cost a is the corrected auxiliary support cost,which can take negative values away from S;it is not assumed to be a norm.

Proof: let L=log(1/λ). For a feasible v,H(Lv)≤λ and A(Lv)=La(v)+r(Lv),where0≤r≤log2 follows termwise from the auxiliary softplus remainder. Thus Q_λ(Lv)/(λL)→a(v). For an infeasible limiting v,the human loss divided by λL diverges,using log(1+exp(−L margin)) and any margin strictly below1. For bounded v sequences,the auxiliary remainder divided by L vanishes uniformly.

It remains to localize minimizers. The positive recession condition and compactness imply inf_(||u||=1){(1−λ)h(u)+λa(u)}/λ is bounded below by a positive constant at small λ. Otherwise a minimizing subsequence tends to u∈S with a(u)>0,contradicting that lower value. Since H≥h and A≥a,the objective is at least a positive constant times λ||β||. Comparison with a feasible Lv gives ||β_λ||=O(L). Compact subsequences and the preceding lower/upper bounds place every limit in V*. Coercivity of a on S also makes V* compact.

This adapts the established vanishing-regularization/margin argument to the specified compensated loss. Rosset,Zhu and Hastie's norm-penalty results must be cited. Generic coefficient divergence or margin convergence is not claimed independently new.

## Comparison with proper pseudo-label augmentation

For ordinary augmentation H+cB,where B is a sum of logistic losses with targets in[0,1] and c>0,both h and b=B^∞ are nonnegative. Its zero recession directions are exactly {d:h(d)=0,b(d)=0},independently of c. The positively weighted stacked design also has the same rank for every c>0. Its finite-root existence event is thus invariant to the positive auxiliary weight. Soft interior targets with a full-rank auxiliary design make B coercive and give a finite augmented root at every positive weight. These are classical separation/regularization properties.

For the corrected objective A includes the subtracted labelled prediction loss. Its recession a can be negative. Finite existence depends on cancellation in(1−λ)h+λa,and changing positive λ can change the event. The explicit phase boundary evaluates this compensated event. Consequently,one cannot transfer a proper-augmentation existence theorem merely by renaming its scalar weight or encoding row types. Li and Huang's arxiv2407.04194v5 Section2 weighted likelihood and Section3.1 existence result belong in this comparison. Their broader high-dimensional error theory and inference results are established prior work;this paper does not claim to supply those results for the corrected estimator.

## Analytic examples and interpretation

With every feature equal to1,labelled choices all1,predictions g_L=0.75 and g_U=0.25,the corrected target is1−λ/2. Its exact root is log(2/λ−1),finite for every positive λ and diverging at zero. Its normalized coefficient tends to1. If g_L=g_U,the target remains1 and every power is boundary. Reversing those two prediction values makes the corrected target exterior at every positive power. All three use the same full-rank auxiliary design. Auxiliary curvature by itself does not guarantee a corrected root.

The theory supports explicit reporting of the fitted power and actual score accuracy. Very small powers can make an absolute gradient tolerance misleading. Root-path checks use analytic or independently scaled equations;they do not treat an optimizer's stopped iterate as a certified root. No confidence-interval validity or new automatic power-selection estimator follows from these propositions.
