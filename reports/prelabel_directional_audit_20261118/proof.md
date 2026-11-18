# Which designs and powers can guarantee finite estimation?

For each of the labelled and auxiliary menu samples, let $C_s$ be the set of achievable mean feature moments under arbitrary menu probabilities. Let $G_s$ be the mean predicted feature moment and $h_s(d)$ the support function of $C_s$. Let $b_Y$ be the mean of the observed labelled choices. Write
$$
M_s(d)=h_s(d)-d^TG_s,\quad
E_Y(d)=h_L(d)-d^Tb_Y,\quad
D(d)=M_U(d)-M_L(d).
$$
All moments use the original sample averaging weights. Availability restricts the alternatives in every menu.

## A design criterion before observing choices

At power $\lambda\in(0,1]$, the corrected moment is $b_Y+\lambda(G_U-G_L)$ and the support function of its active design is $(1-\lambda)h_L+\lambda h_U$. Subtraction gives the exact recession rate
$$
r_{\lambda,Y}(d)=E_Y(d)+\lambda D(d).
$$
For any $d$, choose a maximizing available alternative in every labelled menu. This sets $E_Y(d)=0$ simultaneously for every $\lambda$. The minimum of $r_{\lambda,Y}(d)$ over admissible labelled response vectors is therefore $\lambda D(d)$. The following conclusions hold.

1. Every possible labelled response vector gives a unique finite fit at a positive power exactly when $D(d)>0$ for every nonzero $d$. The condition is independent of positive power. It is equivalent to $C_L-G_L\subset\operatorname{int}(C_U-G_U)$.
2. If $D(d)<0$ for one direction, the same deterministic top-alternative response produces an unbounded loss at every positive power. The ordinary logit loss for this response is bounded below.
3. Suppose $D\ge0$ and equality holds for a nonzero direction, with full contrast rank for the active design. The corrected loss is bounded below for every response. A response maximizing the common direction yields a boundary target, with a finite unattained infimum.

For claim 1, positivity on the sphere gives a strictly positive minimum by continuity. Necessity follows by taking the maximizing response for any direction with nonpositive gap. Strict positive gaps in both opposite directions imply the auxiliary design has full contrast rank. For claim 3, positivity of all directional support inequalities characterizes the interior of the active support. A boundary moment has no finite score root under full rank. Lower boundedness follows from log-sum-exp exceeding the maximum of its input utilities. These implications use the classical exponential-family support criterion. [Geyer (2009)](https://www.stat.umn.edu/geyer/gdor/)

A simple obstruction follows from $D(d)+D(-d)=w_U(d)-w_L(d)$, where $w_s(d)=h_s(d)+h_s(-d)$ is directional width. Equal width in any nonzero direction rules out strict containment. A shift of the prediction-centered endpoints gives a negative gap in one sign whenever the endpoints differ. In choice models with fitted alternative constants and identical full availability across all rows, a contrast between two groups of alternatives has the same width in both samples. This obstruction is independent of curvature or the number of repeated profiles.

## A positive weight after observing choices

Assume the frozen probabilities are strictly positive on every available alternative. Define $\epsilon_*$ as the largest $\epsilon\in[0,1]$ with
$$
b_Y\in (1-\epsilon)C_L+\epsilon G_L.
$$
The analogous labelled moment with a shifted anchor has the same convex geometry as a mixture toward interior pseudo-responses, as in conjugate-penalty methods. Here the mixing fraction instead controls the existing unpenalized compensated objective. [Rigon and Aliverti (2023)](https://www.sciencedirect.com/science/article/pii/S0167715223001256)

Equivalently, $\epsilon_*=\min(1,\inf_{M_L(d)>0} E_Y(d)/M_L(d))$, with the infimum over an empty set defined as positive infinity. The equivalence follows from the supporting inequalities for $(1-\epsilon)C_L+\epsilon G_L$. If $M_L(d)=0$, all labelled alternatives tie in that direction because the predictor is strictly positive on each menu and $b_Y\in C_L$. The resulting inequality is automatic. For $0<\lambda\le\epsilon_*$, use the identity
$$
r_{\lambda,Y}(d)=\{E_Y(d)-\lambda M_L(d)\}+\lambda M_U(d).
$$
The first term is nonnegative and the second is strictly positive whenever $d\ne0$ and the auxiliary contrasts have full rank. The corrected logit then has a unique finite minimizer for every such auxiliary sample. This proof also applies at the exact endpoint $\lambda=\epsilon_*$. If the labelled design has full rank, then $\epsilon_*>0$ exactly when the ordinary fit exists.

The bound is sharp over unrestricted full-rank auxiliary designs that allow vanishing contrast widths. If $\lambda>\epsilon_*$, there is a direction with $E_Y(d)-\lambda M_L(d)<0$. Scale a full-rank auxiliary contrast design by a sufficiently small positive factor. Its rank stays full and $M_U(d)$ is bounded by its vanishing width, even if the fixed predictor changes on the scaled profiles. The corrected loss is unbounded in this direction. A fixed experiment with compulsory constants or minimum attribute spreads may admit a larger design-specific guaranteed power.

The audit calculates $\epsilon_*$ by a linear program over labelled menu probabilities with a lower floor $\epsilon g_{ij}$ within each menu. A separate program moves outward from $G_L$ through $b_Y$ until the first support boundary. For a target unequal to the anchor its ray extension $t_*$ gives $\epsilon_*=1-1/t_*$. The empirical check uses half the computed positive bound to leave room for numerical tolerance. A power of zero in the result means no positive guarantee from this rule. It does not specify a fitted fallback. The guarantee concerns finite estimation. Forecast loss and uncertainty are assessed independently.
