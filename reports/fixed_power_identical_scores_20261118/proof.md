# Global attainment with fixed power and prediction

For every positive rational correction power $\lambda\leq1$ and prescribed finite derivative order, there are three rationally weighted fit problems with the same labelled sample, predictor, power, and auxiliary sample size. Their corrected score rows, objective derivatives at the true coefficient through the prescribed order, and ordinary empirical sandwich agree. The corrected losses have a unique finite minimum, a finite unattained infimum, and an unbounded infimum, respectively. Ordinary labelled logit has a unique finite minimum. The three designs can arise as finite samples under one population with a correct nonconstant conditional prediction and zero true utility coefficients.

## Construction

Take symmetric probability measures $P_+$ and $P_-$ on $[-1,1]$ with rational finite support, equal moments through $2K$ and absolute moments $M_+>M_-$. Symmetry gives equality through $2K+1$. Their existence follows by splitting the signed interpolation measure on $K+2$ distinct rational magnitudes, using the squared magnitudes as interpolation nodes. The difference $\Delta=M_+-M_-$ is nonzero because the divided difference of the square root has a fixed nonzero sign on positive squared magnitudes. The equality can be arranged through any finite order by increasing $K$. [Cai and Low (2011), Lemma 1](https://arxiv.org/pdf/1105.3039)

Write $X=(1,x)^T$ for the fitted features and use $g(x,z)=1/2+z/4$. One half of the records in each sample are informative profiles, split equally between $(x,z)=(0,-1)$ and $(0,1)$. On these profiles assign labelled outcomes in proportions $g(x,z)$. In the remaining labelled rows draw the feature frequencies from $P_+$ and set $z=0$. Assign conditional labelled fraction $1/2+(d/2)\operatorname{sign}(x)$ to outcome one, for $d=1-\lambda\Delta/(2M_+)\in(0,1)$. The remaining auxiliary features have $z=0$ and follow one of $P_+$, $(P_++P_-)/2$, or $P_-$. Rational profile frequencies and outcomes are exact integer counts when expanded at common multiples. The auxiliary sample size is shared across the three cases.

For a single population, take a symmetric finite covariate law that assigns positive mass to every profile in the union, with the same conditional $z$ symmetry at $x=0$. Generate $Y$ given $(x,z)$ as Bernoulli $g(x,z)$. The marginal $E(Y\mid x)=1/2$, so the true intercept and slope in the fitted model are zero. The required labelled sample and every auxiliary sample have positive probability under this law. The exact observed fractions on the other labelled profiles are a realized sample event. Population or sampling probabilities of the three outcomes are outside the claim. In the labelled sample, grouping by $g$ also gives exact observed calibration. Its Brier score is $7/32$ against $1/4$ for the constant half prediction.

## Identical local information

Each group has identical fractions at the two informative profiles. The auxiliary score at the origin is $-\lambda Xz/4$. Its nonzero rows agree across the samples and its remaining rows are zero. Arrange the rows in the same group order to match every score index. The labelled score rows agree because that sample is held fixed. Mean predicted feature vectors in both samples equal $(1/2,0)^T$. The corrected moment is $(1/2,b)^T$, with $b=dM_+/4$. Thus the corrected objective at $(\alpha,\beta)$ is
$$
(1-\lambda)E_L A(\alpha+\beta x)+\lambda E_U A(\alpha+\beta x)-\alpha/2-b\beta ,
\qquad A(u)=\log(1+e^u).
$$
It has the same value $\log 2$ and gradient $(0,-b)^T$ at zero in each case. A derivative of total order $j\ge2$ with $v$ slope differentiations equals $A^{(j)}(0)\{(1-\lambda)E_L x^v+\lambda E_U x^v\}$. These agree for $j\le 2K+1$. The Hessian is positive definite because active profiles at zero and nonzero symmetric values span two features. The sandwich and Newton step are identical by the same row scores and Hessians. Both outcomes have positive counts on every labelled profile. This also gives full-rank positive definite labelled score covariance.

## Different global outcomes

Symmetry and strict convexity in the intercept give $\alpha=0$ as the unique intercept minimum at each slope. The slope derivative on that line is strictly increasing, starts below zero, and tends to
$$
r_+=\frac{(1-\lambda)M_+ +\lambda M_U-d M_+}{4}
=\frac{\lambda}{4}\left(M_U-\frac{M_++M_-}{2}\right) .
$$
The three limits are $+\lambda\Delta/8$, zero and $-\lambda\Delta/8$, as determined by the auxiliary absolute moment. The negative-slope recession is strictly positive. For the positive margin the slope score crosses zero once. At the zero margin the infimum equals the active mass of zero features times $\log 2$. It is reached only in the limit of the slope tending to positive infinity. The negative margin gives a loss tending to minus infinity. Full rank ensures uniqueness at a finite stationary point. For ordinary logit the positive margin is $M_+(1-d)/4>0$, so its finite solution exists.

Within this construction the difference between the two nonzero margins is $\lambda\Delta/4$. The sharp supremum equals $(\lambda/2) E_{2K}$, with $E_{2K}$ the best uniform polynomial approximation error for $|x|$ of degree $2K$. This follows from the classical sharp $\Delta\leq2E_{2K}$ and rational approximation of distinct contact nodes. The finite derivative comparisons concern the specified point. Local information evaluated elsewhere can differ. [Cai and Low (2011), Lemma 1](https://arxiv.org/pdf/1105.3039)

## Small exact example

Take $P_+$ uniform at $\pm1/2$ and $P_-$ with probability $3/4$ at zero and $1/8$ at each of $\pm1$. Then $E_{P_+} x^2=E_{P_-} x^2=1/4$. Use unit correction and $d=3/4$. With 32 labelled rows the two informative profiles have 8 observations each, giving 2 and 6 positive responses, respectively. The groups at $x=-1/2$ and $1/2$ have 8 observations each, giving 1 and 7 positive responses. Each auxiliary sample keeps the same 16 informative rows. Its other 16 rows follow the indicated distribution. The three auxiliary designs have the same corrected gradient $(0,-3/32)^T$, the same Hessian $\operatorname{diag}(1/4,1/32)$ and the same Newton slope of 3. The exact positive-slope recession rates are $1/32$, zero and $-1/32$. The finite case returns slope $2\log 7$. The boundary infimum is $(11/16)\log2$.
