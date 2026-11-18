# Matching local derivatives of compensated choice logit

Consider an intercept \(\alpha\), one bounded attribute \(x\in[-1,1]\), and one slope \(\beta\). Write the two empirical attribute distributions as \(P_L\) and \(P_U\). Both are symmetric. Their moments agree through degree \(2K\), for an integer \(K\ge1\). Symmetry also gives agreement at degree \(2K+1\). Write \(M_L=E_L|x|\), \(M_U=E_U|x|\), and choose their order so that \(\Delta=M_L-M_U>0\). Their common second moment is then positive.

Fix a rational power limit \(c\in(0,1)\). On each labelled profile \(x\), choose a fraction \(1/2+(d/2)\operatorname{sgn}(x)\) of responses equal to one, where \(d=1-c\Delta/M_L\). Both outcomes have positive fractions on every labelled profile. Symmetry gives \(E_LY=1/2\) and \(b=E_L(xY)=(M_L-c\Delta)/2\). Rational masses, attributes and outcome fractions represent a finite equal-weight dataset by repetition. The labelled data and auxiliary attributes stay fixed when evaluating the power path.

Use the logit loss \(\ell(u,y)=\log(1+e^u)-yu\). At power \(\lambda\), use the same frozen predictor \(g_\lambda(x)=1/2+(1/16\lambda)x\) on both groups. Compare powers \(1/4,1/2,3/4\) and set \(c=1/2\) for the numerical witnesses below. All these predictors lie in \([1/4,3/4]\). The same construction works for other rational \(c\) by choosing a positive shrinkage constant small enough for the compared powers.

## Exact equality of local information

The centred predictor is linear in \(x\). The intercept and slope prediction moments agree across groups because the first two design moments agree. In particular the aggregate compensated loss, in the notation of equation (2), is
\[
Q_\lambda(\alpha,\beta)
=(1-\lambda)E_L A(\alpha+\beta x)
+\lambda E_U A(\alpha+\beta x)
-\alpha/2-b\beta,\qquad A(v)=\log(1+e^v).
\]
This expression also holds when the same linear predictor \(g(x)=1/2+x/4\) is fixed at every power.

The value at \((0,0)\) is \(\log 2\). The intercept and slope derivatives of total order \(j\ge2\), with \(v\) derivatives in the slope, are \(A^{(j)}(0)\{(1-\lambda)E_Lx^v+\lambda E_Ux^v\}\). Thus the entire Taylor polynomial through total degree \(2K+1\) is the same at all powers. The gradient is \((0,-b)\), and the Hessian is \(\operatorname{diag}(1,E_Lx^2)/4\), which is positive definite. This compares local summaries at a specified reference point with nonzero score.

With the scaled predictor, the individual corrected reference scores also coincide along the path. Write \(X=(1,x)^\mathsf T\). The labelled scores are \(X\{1/2-Y+x/16\}\), and the auxiliary scores are \(-Xx/16\). Any covariance formula using these same reference rows, sample counts and Hessian is identical across powers. The usual sandwich with unbiased within-group covariances is positive definite because both labelled outcomes occur at each profile, and the labelled design spans the intercept and slope. With the fixed linear predictor the aggregate Taylor polynomial still agrees and the individual score rows vary with power.

## Three different existence outcomes

For every fixed \(\beta\), symmetry makes \(\alpha=0\) the unique intercept minimizer. On that line the slope derivative is strictly increasing, is \(-b<0\) at zero, and has positive-infinity limit
\[
r(\lambda)=\tfrac12\{(1-\lambda)M_L+\lambda M_U\}-b=(c-\lambda)\Delta/2 .
\]
For \(\lambda<c\) it crosses zero at a finite slope. The full Hessian is positive definite at finite coefficients because the positive-weight design spans both variables. The zero of the two scores is the unique global minimum. At \(\lambda=c\) the slope derivative stays below zero, and \(Q_c(0,\beta)\) approaches the weighted mass of \(x=0\) times \(\log 2\) as \(\beta\to+\infty\). Its infimum is finite and unattained. For \(\lambda>c\), the positive slope ray has negative recession and the loss tends to minus infinity. The human-only model at \(\lambda=0\) has a unique finite estimate. At \(c=1/2\), the slope margins at \(1/4\) and \(3/4\) are \(\Delta/8\) and \(-\Delta/8\).

The absolute-moment gap can be constructed at arbitrary \(K\). Take \(K+2\) distinct positive rational magnitudes \(t_i\) and define weights proportional to
\[
\omega_i=\{\,\prod_{j\ne i}(t_i^2-t_j^2)\,\}^{-1}.
\]
Their weighted sums of \((t_i^2)^k\) vanish for \(0\le k\le K\), by the leading coefficient in the Lagrange interpolation polynomial. The positive and negative parts have the same mass. Normalize and symmetrize them to produce \(P_L,P_U\). The signed sum of the \(t_i\) is the divided difference of \(f(z)=\sqrt z\) at the squared nodes. Its order-\(K+1\) derivative keeps a strict sign on the positive interval, so the gap is nonzero. A finite rational construction is therefore available for every finite order.

## How large can the directional difference be?

Let \(E_{2K}\) be the best uniform error for approximating \(|x|\) on \([-1,1]\) by a polynomial of degree at most \(2K\). Matching the moments makes the polynomial expectation identical under \(P_L\) and \(P_U\), so \(\Delta\le2E_{2K}\). Cai and Low (2011, Lemma 1) give the sharp matching pair using symmetric measures. Their alternation characterization supplies \(K+2\) contact magnitudes via the equivalent approximation of \(\sqrt z\) on \([0,1]\). Rational magnitudes can approach those nodes. The signed interpolation weights above preserve exact moment matching and vary continuously at distinct nodes. Thus the same sharp supremum applies to the absolute-moment gap over finite empirical designs, allowing the sample sizes to vary. [Cai and Low, Lemma 1](https://arxiv.org/pdf/1105.3039)

In the compensated choice objective the difference between the positive slope recession rates at two powers is exactly \(|\lambda_2-\lambda_1|\Delta/2\). Its sharp supremum is therefore \(|\lambda_2-\lambda_1|E_{2K}\). The classical Chebyshev expansion of \(|x|\) has absolute coefficient \(4/[\pi(4j^2-1)]\) on \(T_{2j}\). Truncating at \(K\) gives
\[
 E_{2K}\le2/[\pi(2K+1)].
\]
This quantifies slope-margin ambiguity over the constructed design class. Empirical frequencies are addressed separately.

The numerical calculation uses a linear program on an equispaced magnitude grid to locate candidate contact nodes. The exact verifier turns those nodes into rational empirical weights and recomputes the lower bound from the absolute-moment gap. For the upper bound it treats all polynomial coefficients as decimal rationals and evaluates the maximum grid error exactly. Between adjacent grid nodes, the error is bounded by linear interpolation plus \(h^2\sup|p''|/8\), where \(h\) is the grid spacing and \(|x|=x\) on the magnitude interval. For Chebyshev polynomials \(\sup|T_j''|\le(j^4-j^2)/3\). This follows from \(T_j'=jU_{j-1}\), expressing \(U_{j-1}\) as a sum of cosine terms with frequencies \(j-1,j-3,\ldots,1-j\), and using \(\sup|T_s'|\le s^2\). Consequently the grid solution proposes the polynomial and contacts. The certified continuum bounds use rational checks.

## Contribution and scope

Moment matching and its sharp approximation duality are established in Cai and Low (2011) and Han, Jiao and Weissman (2018). Their statistics estimate nonsmooth functionals or distributions. Here the moment difference is converted into an exact change of the recession margin inside the same compensated choice loss, with a common labelled dataset whose ordinary MNL is finite. The local equalities and the existence change hold at every prescribed finite order. Their connection to the fitted objective is the increment of this fixed-design result. The proportional-dimensional theorem in the manuscript has its own assumptions and proof. [Han, Jiao and Weissman](https://proceedings.mlr.press/v75/han18b.html)
