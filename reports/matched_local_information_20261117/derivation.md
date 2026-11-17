# Matched reference information with opposite attainment

This finite-design construction strengthens the existing score-equivalence path. It retains the same two-sample compensated binary logit objective, with an estimated intercept and one attribute slope. The reference is the specified observed point `(alpha,beta)=(0,0)`. The construction makes no claim that this point is a fitted root, that its empirical sandwich is an actual sampling variance, or that the compared powers minimize a variance criterion.

## Design and path

Let `X=(1,x)`. The two labelled rows are

| x | y | base prediction g |
|---|---|---|
| -1 | 0 | 1/10 |
| 1 | 1 | 9/10 |

The eight auxiliary rows have `x=(-2,0,0,0,0,0,0,2)` and base predictions `(1/4,1/2,1/2,1/2,1/2,1/2,1/2,3/4)`. All predictions are fixed numbers. This is one newly constructed deterministic design, not human data or a new independent random cohort.

Put `a=1/2`. For every `a <= lambda <= 1`, use

$$g_\lambda=1/2+(a/\lambda)(g-1/2),\qquad Q_\lambda=L_m(Y)+\lambda\{L_N(g_\lambda)-L_m(g_\lambda)\}.$$

The same design, labels and reference point are used at every power. Every prediction remains strictly between zero and one. Both groups have exactly the same second-moment matrix `mean(X X^T)=I_2` and zero attribute mean.

## Identical reference information

At the reference point, let the unaveraged labelled and auxiliary score rows be

$$s_{L,i}=X_i\{1/2-y_i+\lambda(g_{\lambda,i}-1/2)\},\qquad s_{U,i}=\lambda X_i(1/2-g_{\lambda,i}).$$

Substitution of the path removes lambda from every row, before any covariance is computed. The labelled rows are `(3/10,-3/10)` and `(-3/10,-3/10)`. The two nonzero auxiliary score rows are `(1/8,-1/4)` and `(-1/8,-1/4)`. The other six are zero. Thus

$$Q_\lambda(0)=\log2,\quad \nabla Q_\lambda(0)=(0,-29/80)^T,\quad b_\lambda=(1/2,29/80)^T.$$

The Hessian at the same reference point is

$$H_\lambda(0)=\frac14\{(1-\lambda)\overline{XX^T}_L+\lambda\overline{XX^T}_U\}=I_2/4.$$

The complete third derivative tensor is zero, as the third derivative of the binary log partition is zero at zero logit. These are equal Taylor polynomials through degree three, not equal functions on a neighborhood. For example, the unit-minus-half-power fourth slope derivative is `-3/16`.

For an explicit empirical sandwich convention, use unbiased within-group score covariances `S_L,S_U` and `J=S_L/m+S_U/N`. Every row is fixed, so both centered and uncentered score-second-moment conventions are invariant across powers. Under the stated convention,

$$J=\operatorname{diag}(4057/44800,3/1792),\qquad H^{-1}JH^{-1}=\operatorname{diag}(4057/2800,3/112).$$

Both matrices are positive definite. The Newton update from the reference is `(0,29/20)` for every power. A finite identical update supplies no claim of a global minimizer or interval coverage.

## Exact attainment boundary

The weighted design is symmetric in x. Its intercept moment is `1/2`. For every fixed beta, alpha=0 is the unique conditional intercept minimizer: its intercept gradient is zero by symmetry and the intercept curvature is strictly positive. On that line,

$$Q_\lambda(0,\beta)=(1-\lambda)\log(2\cosh(\beta/2))+(\lambda/4)\log(2\cosh\beta)+(3\lambda/4)\log2-(29/80)\beta.$$

Its slope derivative is

$$F_\lambda(\beta)=\frac{1-\lambda}{2}\tanh(\beta/2)+\frac{\lambda}{4}\tanh\beta-\frac{29}{80}.$$

This derivative is strictly increasing at every finite beta. It is negative at zero. Its positive-infinity limit is

$$\lim_{\beta\to\infty}F_\lambda(\beta)=\frac{11-20\lambda}{80}.$$

For `1/2 <= lambda < 11/20`, there is a unique positive finite root. Since the positive-weight rows span both fitted parameters, the full objective is strictly convex and this root with alpha=0 is its unique global minimizer. At `lambda=11/20`, the derivative is negative at every finite beta and tends to zero; the infimum is finite and unattained. Its limiting value is `(33/80)log2`. For `lambda>11/20`, the direction `(0,1)` has negative recession `(11-20 lambda)/80`, so the full objective is unbounded below.

At half power, the positive slope recession is `1/80`, and the independently computed slope is approximately `3.67677057698`. At unit power, the recession is `-9/80`. The exact signs, rather than optimizer return flags, establish the distinction.

## A parameter family within the same objective

The matching does not require one isolated prediction vector. Keep the symmetric labelled x values `-1,1`, with labels `0,1` and base predictions `1/2-u,1/2+u`. Give the auxiliary attribute mass `1/(2c^2)` at each of `-c,c`, and mass `1-1/c^2` at zero, where `c>1`. Its extreme predictions are `1/2-v,1/2+v`, with `1/2` at zero. For finite equal-weight designs, choose c so these masses are rational and repeat the zero profile accordingly. Let `0<u,v<1/2` and `a <= lambda <= 1`.

Both groups again have zero first moment and unit second moment. The score rows remain invariant under `g_lambda=1/2+(a/lambda)(g-1/2)`, and the reference Hessian remains `I_2/4`. The slope moment is `b=1/2-a(u-v/c)`. When b is positive, the positive ray changes sign at

$$\lambda_* = \frac{2a(cu-v)}{c-1}.$$

Whenever `a<lambda_*<1`, the same argument yields finite attainment at a and unboundedness at one. The explicit design uses `(c,u,v,a)=(2,2/5,1/4,1/2)`. At fixed c=2, the strict inequalities persist on an open set of `(u,v,a)`, without changing the design counts. This family is an algebraic argument; the verification artifacts numerically evaluate the explicit design only.

## Interpretation and limits

The construction fixes a stronger finite-reference signature than the earlier scalar illustration: every corrected score row, the complete observed Hessian, the complete empirical sandwich and the first Newton update. The attainable-moment boundary still moves. Thus these reference quantities do not determine global attainment in the specified compensated family.

This does not invalidate local asymptotic theory under its assumptions, establish a better estimator, show that a variance-optimal rule makes this particular transition, or prove that this situation is common in applied DCEs. The distinction between local information and global recession is classical. The candidate increment is the exact matching inside the existing compensated choice objective, in support of the paper's broader boundary calculation. Its priority and sufficient publication value remain unconfirmed.
