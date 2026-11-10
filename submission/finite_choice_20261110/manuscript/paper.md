# When prediction assisted logit estimates exist

## Abstract

Predicted choices can supplement observed choices through a correction that preserves the population score. The corrected empirical logit objective can have no finite minimizer. We derive its existence boundary when labelled and auxiliary samples grow with attribute dimension, allowing a utility signal, an estimated intercept and an imperfect frozen predictor. Correction power changes the existence event. In a balanced benchmark with equal sample sizes and proxy error 0.25, the critical dimension to label ratio is 0.355 at unit power and 0.603 at the established variance reference power. Its positive power right limit is 0.664, while exactly zero power gives the ordinary logit threshold 0.5. The limiting boundary applies simultaneously to all powers and governs strictly positive data dependent powers tending to zero. It strictly exceeds the ordinary logit existence boundary for every frozen predictor strictly inside the unit interval. A fixed sample result explains the coefficient growth at small power. Released vaccine choices supply verified exterior moments, and simulations on their attribute support retain seven estimation strategies. Replaying 500 existing samples shows that increased availability can accompany much larger error on shared available fits. The results connect correction weights to existence, coefficient behavior and reliable reporting in choice analysis.

Keywords: binary choice; prediction powered inference; logistic separation; finite auxiliary samples; existence of estimates

## 1 Introduction

Choice analysts increasingly use machine predictions in model development and estimation. Recent work in the Journal of Choice Modelling examines both machine learning in choice analysis and language models as assistants for utility specification (van Cranenburgh et al., 2022; Sfeir et al., 2026). A related use supplies predicted responses for observations without human choice labels. Prediction powered inference corrects these responses using a labelled sample, allowing the population estimating equation to retain its human choice target (Angelopoulos et al., 2023, 2024).

An analyst must also be able to solve the empirical estimating equation. A corrected logit objective can decrease indefinitely along a coefficient direction. Continued optimization then produces increasingly extreme coefficients without reaching a finite optimum. This event is consequential when the human label budget is small relative to the utility specification. Prediction quality, correction power and the number of auxiliary observations all enter the event.

We characterize the finite solution boundary of a specified unpenalized, prediction assisted binary logit objective. Our main result treats an estimated intercept, one Gaussian utility signal and an increasing number of independent Gaussian attributes. The labelled and auxiliary samples grow at a finite ratio. Predictions are frozen functions of the signal and auxiliary information, and can be miscalibrated. An explicit variational expression gives the critical attribute dimension per labelled observation. A balanced specialization reduces the calculation to one scalar optimization.

The result explains a distinction between efficiency calculations and existence calculations. At proxy error 0.25 and equal labelled and auxiliary sample sizes, the balanced unit power boundary is 0.355. Applying the established population variance reference power gives 0.603. The positive power right limit is 0.664, exceeding the ordinary logit endpoint 0.5. The general signal and estimated intercept model also has an upper boundary for all choices of power. Above it, power selection cannot restore a finite unpenalized root. Strictly positive powers tending to zero attain this existence region. Its boundary strictly exceeds ordinary logit even for uninformative or reversed predictions. A fixed sample limit explains how coefficients can approach a separator as their norms grow. The finite sample support audit uses the actual design matrix and also accommodates categorical attributes.

The theoretical tools have clear antecedents. Logistic separation and ordinary logit existence thresholds are established (Candès and Sur, 2020). Statistical dimension locates transitions in random convex feasibility problems (Amelunxen et al., 2014). The unit power uninformative benchmark is a classical Gaussian zonotope absorption event (Kabluchko and Seidel, 2020). Koriyama and Bellec (2025) give general existence results for convex M estimators with finite rowwise loss infima. Positive power correction creates labelled row losses with infimum minus infinity; finite estimability depends on compensation by other rows. We derive the explicit finite auxiliary boundary for that corrected objective. Power tuning and one step estimation come from PPI++ and remain comparison methods throughout the paper.

Unbiased noisy label correction and preservation of convexity are established (Natarajan et al., 2013); corrected empirical risk can become negative in positive unlabelled learning (Kiryo et al., 2017). Finite sample estimation costs of PPI++ are established for mean estimation (Mani et al., 2026), and synthetic likelihood regularization has a developed high dimensional theory (Li and Huang, 2026). DY prior penalized logit guarantees finite estimates for full rank designs and has proportional dimensional theory, including a recent fitted intercept treatment (Sterzinger and Kosmidis, 2026; Sterzinger, 2026). Proper pseudo-label augmentation has nonnegative row recession costs and an existence event invariant to every positive auxiliary weight. The corrected choice loss has negative labelled recession costs whose compensation changes with power. The two sample geometry gives the probability boundary for this event. Generic vanishing regularization and margin limits are established (Rosset et al., 2004); the corrected auxiliary recession function determines the limit studied here.

Two empirical components connect the boundary to choice analysis. First, a pinned public vaccine choice dataset provides corrected objectives whose moments are demonstrably outside their attainable support. The audit also verifies finite solutions from existing power tuning in those same cases. Second, known truth simulations on the released attribute support compare seven strategies under oracle and biased predictions. They separate availability from coverage and retain the strong point performance of soft pooling and penalized alternatives. This reporting supports an analysis decision: specify the target objective, check its finite solution, and evaluate uncertainty for the estimator actually returned.

## 2 Prediction assisted binary logit

### 2.1 Choice target and correction

Let $X$ contain attribute contrasts and any estimated intercept. A binary choice $Y$ has logit probability $\sigma(X^T\beta)$, where $\sigma(u)=(1+e^{-u})^{-1}$. Write the negative log likelihood contribution as

$$\ell(u,y)=\log(1+e^u)-yu. \qquad (1)$$

There are $m$ labelled rows and $N$ independent auxiliary rows from the same distribution. A frozen prediction $g$ is available on both samples. Let $L_m(Y;\beta)$ denote the average labelled loss, and let $L_m(g;\beta)$ and $L_N(g;\beta)$ replace the response by its prediction. For fixed power $0<\lambda\leq1$, the PPI objective is

$$Q_\lambda(\beta)=L_m(Y;\beta)+\lambda\{L_N(g;\beta)-L_m(g;\beta)\}. \qquad (2)$$

The two prediction terms have equal expectations. The population score in (2) equals the human loss score, including when $g$ is imperfect. Under model misspecification its target is the human logistic projection. This identity uses a fixed predictor and power; it does not assert finite sample unbiasedness for a fitted predictor or estimated power.

For a stacked design with labelled rows first, define weights and a corrected moment by

$$w_i=(1-\lambda)/m\ \ (i\leq m),\qquad w_i=\lambda/N\ \ (i>m),$$

$$b_\lambda=m^{-1}X_L^T(Y_L-\lambda g_L)+\lambda N^{-1}X_U^Tg_U. \qquad (3)$$

Then $Q_\lambda(\beta)=\sum_i w_i\log(1+e^{X_i^T\beta})-b_\lambda^T\beta$. The Hessian is positive semidefinite. At unit power only auxiliary rows contribute curvature, while labelled rows still contribute to the corrected moment.

### 2.2 The finite design audit

Define the attainable moment set

$$\mathcal Z_w=\{\sum_i w_i X_i p_i:0\leq p_i\leq1\}. \qquad (4)$$

Proposition 1. Suppose the rows with positive weight span the parameter space. Objective (2) has a unique finite minimizer exactly when $b_\lambda$ is in the interior of $\mathcal Z_w$. If the moment is outside this set, the objective is unbounded below. A boundary moment has no finite stationary point. Without the rank condition, coefficients are not uniquely identified.

This is a classical convex support condition. The gradient of the log partition function maps finite coefficients onto the interior of its attainable moments. Its recession function is

$$Q_\lambda^\infty(d)=\sum_iw_i(X_i^Td)_+-b_\lambda^Td. \qquad (5)$$

An exterior moment supplies a direction with a negative value in (5). An interior moment makes the recession function strictly positive on the unit sphere, establishing coercivity and, with full rank, strict convexity.

Our implementation maximizes a common slack in the constraints $b_\lambda=\sum_iw_iX_ip_i$, $s\leq p_i\leq1-s$. A separate linear program obtains an exterior separating direction and verifies the recession gap. It reports rank deficiency, numerical boundary proximity and unresolved cases separately. The interface preserves the specified objective; power changes, pooling and penalties are explicit subsequent decisions. This audit implements established support geometry and supplies the finite design counterpart to the probabilistic calculation below.

## 3 The existence boundary with finite auxiliary data

### 3.1 Model and main result

Let the contrast vector be $(S,W)$, where $S$ is standard normal and $W$ has $q$ independent standard normal coordinates. Assume

$$P(Y=1\mid S)=\sigma(\gamma S),\qquad |\gamma|<\infty. \qquad (6)$$

The vector $W$ is independent of a fixed dimensional tuple $(S,Y,Z,U)$. Here $Z$ is binary auxiliary information and $U$ is optional additional prediction information. Their conditional law given $(S,Y)$ is fixed. The predictor $g(S,Z,U)$ is measurable, frozen and strictly between zero and one almost surely. Both samples are independent draws from this joint law. The model estimates an intercept, the slope of $S$ and all $q$ remaining slopes. The true intercept and remaining slopes are zero.

Consider $N/m\to\tau>0$ and $q/m\to\kappa$, with a fixed power in $(0,1]$. Define $s_L=1-2Y$, $s_U=2Z-1$ and $r=|Y-g|$. The oriented row intervals are

$$I_L=[\lambda r,1-\lambda+\lambda r], \qquad (7)$$

$$I_U=(\lambda/\tau)[-g,1-g]\quad\text{when }Z=1,$$

$$I_U=(\lambda/\tau)[g-1,g]\quad\text{when }Z=0.$$

For standard normal $H$, write

$$f(a)=E(H-a)_+^2=(1+a^2)\Phi(-a)-a\varphi(a),$$

$$B(h,t,a,b)=1+h^2-f(-h-ta)-f(h+tb). \qquad (8)$$

When the last argument below is an interval, its endpoints enter the last two arguments of $B$. For $\eta=(\eta_0,\eta_1)$ define

$$\Psi(\eta,t)=E B(s_L(\eta_0+\eta_1S),t,I_L)$$

$$\hphantom{\Psi(\eta,t)}+\tau E B(s_U(\eta_0+\eta_1S),t,I_U). \qquad (9)$$

Theorem 1. Under the assumptions above, let

$$\kappa_c=\inf_{\eta\in\mathbb R^2}\ \sup_{t\geq0}\Psi(\eta,t). \qquad (10)$$

If $\kappa<\kappa_c$, the probability that (2) has a unique finite minimizer tends to one. If $\kappa>\kappa_c$, the probability of a finite minimizer tends to zero. The statement leaves the equality case unspecified.

The boundary combines the finite auxiliary ratio, utility signal, predictor distribution and power. Its fixed dimensional optimizations can be evaluated by quadrature and scalar scale maximization. Under joint sign symmetry of $(S,Y,Z,U)$ and $g(-S,1-Z,U)=1-g(S,Z,U)$, convexity and symmetry allow $\eta_0=0$. An asymmetric prediction retains both multiplier coordinates.

The theorem holds for independent Gaussian remaining attributes and a frozen low dimensional predictor. Fitted predictions, repeated choices from a respondent, multinomial menus and general categorical design distributions require separate arguments. For an observed categorical design, Proposition 1 directly checks the stated objective without invoking a Gaussian approximation.

### 3.2 Why the corrected loss requires a separate calculation

The labelled contribution in (2), up to its averaging factor, is

$$\ell_L(u)=(1-\lambda)\log(1+e^u)-(Y-\lambda g)u. \qquad (11)$$

For $Y=0$, taking $u\to-\infty$ sends this contribution to minus infinity. For $Y=1$, taking $u\to+\infty$ has the same effect. At unit power it is affine. The complete objective can still be coercive because auxiliary rows offset those directions. The finite rowwise infimum assumption in Koriyama and Bellec (2025, Assumption 2.5) excludes these labelled losses. Theorem 1 evaluates the compensation using the actual two sample loss structure.

The proof conditions on $(S,Y,Z,U)$ and orients the remaining Gaussian rows by $s_L$ and $s_U$. A finite corrected moment becomes a Gaussian nullspace intersection with the cone generated by the product of (7), subject to the intercept and signal constraints. Classical random subspace theory locates the transition at the statistical dimension of that constrained cone. The new calculation is its limit (10) for the specified corrected logit objective. Appendix A gives the derivation, including tightness of both optimization variables and the connection from feasibility to an interior moment.

### 3.3 Balanced oracle benchmark

Take $\gamma=0$, let $Z$ independently flip $Y$ with probability $\epsilon\in(0,1/2)$, and use $g=\epsilon+(1-2\epsilon)Z$. Symmetry and independence minimize both multiplier coordinates at zero. The finite ratio boundary simplifies to

$$\kappa_c=1+\tau-\min_{t\geq0}F(t), \qquad (12)$$

$$F(t)=(1-\epsilon)\{f(-\lambda\epsilon t)+f((1-\lambda+\lambda\epsilon)t)\}$$

$$\hphantom{F(t)}+\epsilon\{f(-\lambda(1-\epsilon)t)+f((1-\lambda\epsilon)t)\}$$

$$\hphantom{F(t)}+\tau\{f(\lambda(1-\epsilon)t/\tau)+f(\lambda\epsilon t/\tau)\}.$$

At unit power this is

$$\kappa_c=\tau[1-\min_{u\geq0}\{\tau\epsilon(1-\epsilon)u^2+f((1-\epsilon)u)+f(\epsilon u)\}]. \qquad (13)$$

For $\epsilon=0.25$ and $\tau=1$, unit power gives 0.3554. The established population variance reference power $\lambda=\tau/(1+\tau)=0.5$ gives 0.6029. At $\tau=4$ the corresponding boundaries are 0.6120 and 0.7223, with reference power 0.8. As $\tau$ tends to infinity, the unit power boundary tends to $1/[2\pi\epsilon(1-\epsilon)]$, equal to 0.8488 at this error rate. The finite auxiliary calculation materially changes the existence region even for an informative, correctly calibrated predictor.

Zero power uses ordinary logit separation and is handled separately. Its balanced null threshold is 0.5. The interval cone in (7) contains zero at that endpoint, so a zero feasibility check would be uninformative. Our code uses the ordinary separation condition when the established weight formula selects zero.

### 3.4 Utility signal and prediction calibration

For a binary proxy with independent error $\epsilon$, the oracle prediction is

$$g_0(S,Z)=\sigma(\gamma S+(2Z-1)\log((1-\epsilon)/\epsilon)). \qquad (14)$$

We evaluate both this predictor and frozen transformations $g=\sigma(a\log(g_0/(1-g_0))+b)$. The overconfident case has $(a,b)=(2,0)$; shifted log odds use $(1,1)$. Table 1 fixes $\gamma=1.5$ and $\epsilon=0.25$. The comparison power minimizes the existing fixed dimensional orthogonal score variance. It provides a common reference and is held fixed in the phase experiment. It is not claimed to optimize a proportional dimensional inference problem.

Table 1  Theoretical boundaries with a utility signal and estimated intercept

| Auxiliary to label ratio | Prediction | Power | Critical slope to label ratio |
|---|---|---|---|
| 1 | Oracle | 1.0000 | 0.2795 |
| 1 | Oracle | 0.5000 | 0.4637 |
| 1 | Overconfident | 1.0000 | 0.2478 |
| 1 | Overconfident | 0.3386 | 0.4927 |
| 1 | Shifted log odds | 1.0000 | 0.2434 |
| 1 | Shifted log odds | 0.3078 | 0.4810 |
| 4 | Oracle | 1.0000 | 0.4690 |
| 4 | Oracle | 0.8000 | 0.5487 |
| 4 | Overconfident | 1.0000 | 0.4261 |
| 4 | Overconfident | 0.5417 | 0.6046 |
| 4 | Shifted log odds | 1.0000 | 0.4060 |
| 4 | Shifted log odds | 0.4925 | 0.5729 |

At equal sample sizes, overconfidence moves the unit power boundary from 0.2795 to 0.2478; shifting log odds gives 0.2434. The variance reference powers yield boundaries near 0.48 to 0.49. A changed prediction distribution affects the allowable dimension as well as the efficiency calculation. The two free multipliers in (10) account for the signal and the asymmetry of shifted predictions.

Figure 1 evaluates these boundaries using 300 independent draws at each displayed point. Both samples are generated anew on each draw, and the same draw supplies the compared dimensions within a condition. The smooth finite sample transitions sharpen as the labelled sample increases. The complete oracle and imperfect prediction grids include 80 conditions each, with no numerically unresolved feasibility classifications.

![Figure 1](figures/figure1_phase.png)

Figure 1  Finite fit frequencies at auxiliary to label ratio one and utility signal 1.5. Circles and dashed lines use 40 labels; squares and solid lines use 80. Dotted lines show asymptotic boundaries. Error bars give approximate 95% binomial Monte Carlo intervals from 300 draws per point. Both panels use Gaussian attributes.

The boundary can retain predictor information beyond its Brier risk and fixed dimensional covariance. Appendix B constructs two balanced predictions with the same mean, risk and complete covariance function for every fixed power, yet different existence thresholds. Greater dispersion of squared prediction confidence lowers the threshold in that construction. Its gap is about 6% at auxiliary to label ratio 0.05 and 0.12% at ratio one. This consequence distinguishes the summaries mathematically; the displayed effect at larger auxiliary ratios is small.

### 3.5 Positive powers and the zero endpoint

Holding the sample and predictor fixed reveals a simple ordering. Write $H=L_m(Y)$ and $A=Q_1$. For $0<\alpha<\lambda\leq1$,

$$Q_\alpha/\alpha=Q_\lambda/\lambda+(1/\alpha-1/\lambda)H. \qquad (15)$$

The human loss is nonnegative. A coercive objective at power $\lambda$ remains coercive at the smaller positive power $\alpha$. With the rank condition of Proposition 1 this gives nested finite solution events. Exactly zero power removes auxiliary curvature and requires its own separation check.

For the balanced oracle with fixed zero intercept, put

$$G(v)=(1-\epsilon)f(-\epsilon v)+\epsilon f(-(1-\epsilon)v)$$

$$\hphantom{G(v)}+\tau\{f((1-\epsilon)v/\tau)+f(\epsilon v/\tau)\}. \qquad (16)$$

Corollary 1. For $\tau>0$ and $0<\epsilon\leq1/2$, the boundary strictly decreases with positive power and has right limit

$$\kappa_c(0+)=1+\tau-\min_{v\geq0}G(v)>1/2=\kappa_c(0). \qquad (17)$$

At $\tau=1$ and $\epsilon=0.25$, the right limit is 0.664488. An uninformative prediction at $\epsilon=1/2$ also has a gap, with right limit 0.608737. This gap concerns existence and auxiliary stabilization; it does not measure additional choice information.

A fixed sample counterpart describes the coefficient cost. On a strictly separated human sample, suppose the unit power recession function is positive on every nonzero human weak separator. Small positive powers then have finite roots whose norms diverge as power tends to zero. After division by $\log(1/\lambda)$, their limit set minimizes the corrected auxiliary recession cost subject to human margins of at least one. Appendix D states the condition, proves (17) and derives this path. The generic margin mechanism follows established vanishing regularization arguments; the cost and existence boundary belong to the specified corrected loss.

### 3.6 A simultaneous boundary with utility signal and estimated intercept

Retain the model of Section 3.1. Write $h_L=s_L(\eta_0+\eta_1S)$, $h_U=s_U(\eta_0+\eta_1S)$ and $J_U=I_U/\lambda$. Dividing the positive power intervals by power leaves the auxiliary interval $J_U$ fixed. The labelled interval expands to $[r,r+1/\lambda-1]$. Its closed limit gives

$$\Psi_\infty(\eta,v)=E\{1+h_L^2-f(-h_L-vr)\}$$

$$\hphantom{\Psi_\infty(\eta,v)}+\tau E B(h_U,v,J_U),\qquad v\geq0. \qquad (18)$$

Define the limiting and ordinary logit boundaries by

$$\kappa_\infty=\inf_\eta\sup_{v\geq0}\Psi_\infty(\eta,v),\qquad \kappa_0=\inf_\eta E f(h_L). \qquad (19)$$

Theorem 2. Under the assumptions of Theorem 1, $\kappa_c(\lambda)\to\kappa_\infty>\kappa_0$ as positive $\lambda\to0$. For $\kappa<\kappa_\infty$, some fixed power in $(0,1)$ gives a unique finite root with probability tending to one. For $\kappa>\kappa_\infty$, the probability that any power in $[0,1]$ gives a finite root tends to zero. Every sequence of possibly data dependent powers satisfying $0<\lambda_m\leq1$ almost surely and $\lambda_m\to0$ in probability has the same strict two sided existence boundary $\kappa_\infty$. The equality case is unspecified.

The simultaneous upper bound follows from one closed cone containing all positive power cones and the ordinary logit feasibility face. The lower side combines fixed power thresholds with the nesting identity (15). Appendix D proves the required uniform controls without exchanging the infimum and supremum. No rate or independence condition is imposed on the selected positive powers. The frozen predictor remains fixed across sample sizes, strictly inside the unit interval; the result does not cover a changing sequence of trained or hard predictors.

For utility signal 1.5 and proxy error 0.25, the ordinary boundary is 0.3903. With equal sample sizes, the limiting boundaries are 0.5082 for the oracle, 0.5061 for overconfident predictions and 0.4918 for shifted log odds. At auxiliary ratio four they are 0.6263, 0.6249 and 0.5875. Constant and reversed predictions also give strict existence gains, of different sizes. The strict gain establishes stabilization of this objective. Parameter error, choice probability loss and confidence interval performance still require their own evaluation, as Section 5 illustrates.

## 4 Evidence from released vaccine choices

### 4.1 Data and the stated estimands

We use numeric files released with Generative Augmented Inference (Lu et al., 2026), pinned to repository commit d18717206867c7a12ff470879a0dd5d488b300c2. The file contains 4605 human binary choice rows with two 12 column vaccine profiles and 1200 aligned hard AI predictions. Differencing the profiles and removing a common first column yields 11 contrasts. The released study excludes respondents who never select a vaccine; these rows do not represent all originally recruited respondents. Code and data licenses are recorded in the reproduction package.

The reanalysis fixes the aligned 1200 row prefix and evaluates 500 allocations with labelled budgets of 50, 100, 150 and 200 rows. Allocations preserve the source implementation's five row blocks. The released numeric file does not establish participant identifiers, so the comparison is conditional on that block convention. The empirical target is the converged logistic projection on the fixed prefix. It is a reference estimate, distinct from a known population parameter.

Nuisance predictions use five row folds and ridge logistic fits, either within hard AI label strata or using attributes alone. Predictions are held fixed within each objective comparison. This fitted prediction reanalysis illustrates finite support geometry; its distribution is outside the frozen predictor theorem. It also changes the source prefix pool convention, so it is not a reproduction of a published table.

### 4.2 Finite solutions and explicit alternatives

In the auxiliary only objective (2), unit power has six exterior AI moments among 500 allocations at 50 labels; the attributes only comparator has two. Each exterior moment has a separately verified positive recession gap. Existing power tuning from a converged human estimate returns finite roots in all 500 allocations for both predictors. Initializing that same tuning procedure from a verified unit power root leaves six and two initializers unavailable, respectively. The method definition and its initialization both affect availability.

We apply the finite objective audit to all eight recorded exterior cases. Unit power remains exterior in every case; the smallest verified gap is 0.000388. Existing human initialized tuning solves its stated objective in all eight. Soft pooling also solves its own objective, with a corrected unit power score discrepancy of at least 0.1072 across these cases. The examples exhaust the recorded failures of that experiment. They are not an additional prevalence sample or a new estimator comparison.

Component clipping explains the soft pooling alternative in the source style pooled imputation experiment. With inverse label probability weight greater than one, clipping corrected binary one hot targets to the unit interval and renormalizing returns observed choices on labelled rows and predictions on auxiliary rows. This identity holds for every prediction in the unit interval. It removes the original residual correction from the loss being optimized. In the separate pooled imputation comparison, exterior counts are five and two; these counts refer to a different objective and are retained separately.

The released reference vector also requires an explicit optimization convention. It agrees with a 200 step Adam reconstruction within $1.49\times10^{-7}$. A converged fit on all 4605 rows reduces total negative log likelihood by 7.014, with a maximum coefficient difference of 0.1855. Our point comparisons use converged projections and report fixed iteration outputs separately. This check concerns the released reference and the inspected implementation; it does not identify the source of every gain in the published study.

## 5 Availability and uncertainty on the released attribute support

### 5.1 Known truth experiment

To evaluate error and coverage against a known parameter, we sample attribute rows independently with replacement from the fixed 1200 row support. The simulation parameter is its converged 11 coordinate logistic vector. A second signal group multiplies that vector by three. We generate choices from the stated logistic probabilities and binary proxies by independent flips. No new human respondents or language model answers are generated.

The oracle grid crosses $m=20,50,150$, $N=150,600$ and proxy error 0.1 or 0.25, with 500 repetitions per condition. Two additional signal groups use oracle, doubled log odds and shifted log odds predictions, $m=50,150$, $N=600$ and proxy error 0.25. Each of those groups has six conditions and 500 repetitions. Every method shares the attributes, choices and proxy draws within a condition. The full grids and their seeds accompany the aggregate tables.

We compare human maximum likelihood, unit power PPI, human initialized power tuning, ridge initialized power tuning, one step PPI from that ridge start, unit power ridge PPI and soft pooling. Ridge fits use penalty $\|\beta\|^2/(2m)$ on every coordinate. The two power methods use the established PPI++ covariance trace convention at their stated initializers. The one step update uses the same estimated power and initial ridge fit. Soft pooling combines observed labelled choices with frozen auxiliary predictions at equal row weights. Its oracle version has correct conditional mean information.

Two DY prior settings provide a labelled only comparator in the two conditions of Table 2. The estimator fits pseudo-responses $\alpha Y+(1-\alpha)/2$, using $\alpha=m/(m+11)$ and a fixed $\alpha=0.5$ sensitivity. The first follows the dimension based choice used in the authors' supplementary implementation. Both use the same human rows, a zero prior mode and zero initialization, with no auxiliary predictions. They replay the existing 500 draws in each condition; the original seven strategy grids remain unchanged.

For each method we record point availability and interval availability separately. Parameter MSE is the mean of squared coordinate errors, conditional on an available point. Coverage is the fraction of 11 marginal intervals covering their corresponding true coordinates, averaged over repetitions with intervals. Issued and covers assigns zero to a repetition without intervals before averaging over all 500 repetitions. Neither summary is simultaneous coverage. Monte Carlo errors use the repetition as the independent unit. Wald intervals use the implemented score sandwich; ridge intervals do not adjust for shrinkage bias. Appendix C states the covariance formulas and tuning convention.

### 5.2 What a finite output establishes

Table 2 contrasts two prespecified mechanisms: scarcity of human labels with oracle predictions, and overconfidence under stronger utility at a larger label budget. All conditions and error tails remain in the reproduction tables. The selected columns show how availability, parameter error and coverage can lead to different assessments of the same estimator.

Table 2  Point availability and interval performance

| Condition | Estimator | Available | MSE | Coverage | Issued and covers |
|---|---|---|---|---|---|
| 20 labels oracle | Human MLE | 116/500 | 8.5602 | 0.926 | 0.215 |
| 20 labels oracle | Unit PPI | 412/500 | 4.8362 | 0.998 | 0.822 |
| 20 labels oracle | Power human start | 116/500 | 6.0858 | 0.907 | 0.210 |
| 20 labels oracle | Power ridge start | 348/500 | 21.5408 | 0.801 | 0.558 |
| 20 labels oracle | One step | 500/500 | 1.5054 | 0.827 | 0.827 |
| 20 labels oracle | Ridge PPI | 500/500 | 0.2193 | 0.961 | 0.961 |
| 20 labels oracle | Soft pooling | 500/500 | 0.0057 | 0.942 | 0.942 |
| 20 labels oracle | DY dimension rule | 500/500 | 0.9277 | 0.737 | 0.737 |
| 20 labels oracle | DY alpha 0.5 | 500/500 | 0.4397 | 0.740 | 0.740 |
| 150 labels overconfident | Human MLE | 500/500 | 0.1353 | 0.947 | 0.947 |
| 150 labels overconfident | Unit PPI | 500/500 | 0.1326 | 0.971 | 0.971 |
| 150 labels overconfident | Power human start | 500/500 | 0.1050 | 0.942 | 0.942 |
| 150 labels overconfident | Power ridge start | 500/500 | 0.1042 | 0.942 | 0.942 |
| 150 labels overconfident | One step | 500/500 | 0.0992 | 0.945 | 0.945 |
| 150 labels overconfident | Ridge PPI | 500/500 | 0.0860 | 0.958 | 0.958 |
| 150 labels overconfident | Soft pooling | 500/500 | 0.0353 | 0.804 | 0.804 |
| 150 labels overconfident | DY dimension rule | 500/500 | 0.1011 | 0.945 | 0.945 |
| 150 labels overconfident | DY alpha 0.5 | 500/500 | 0.1035 | 0.625 | 0.625 |

With 20 labels and oracle predictions, the human MLE is available in 116 of 500 repetitions. Unit power PPI returns 412 points. Its conditional coverage is 0.998, while issued and covers is 0.822. One step PPI returns 500 points and has coverage 0.827. Ridge PPI and oracle pooling also return every point and have coverage near 0.95 in this condition. The oracle pooling MSE advantage is substantial and is preserved in the comparison.

Under stronger utility, 150 labels and overconfident predictions, all seven estimators return 500 points and intervals. Soft pooling has MSE 0.0353 and coverage 0.8036. Ridge initialized power tuning has MSE 0.1042 and coverage 0.9424; one step PPI has MSE 0.0992 and coverage 0.9451. Exact finite support expectations give a soft pooling population parameter MSE of 0.0181 in this condition. The fixed power corrected population score still vanishes at the true parameter. This comparison isolates the difference between a changed target and corrected estimation uncertainty.

Both DY settings return 500 estimates and intervals in each condition. With 20 labels, dimension based shrinkage gives MSE 0.9277 and diagnostic coverage 0.7369; half power gives 0.4397 and 0.7398. With 150 labels, dimension based DY gives MSE 0.1011 and coverage 0.9451 using labelled data alone. Its paired MSE difference from one step PPI is 0.001935 with Monte Carlo standard error 0.001399 on the same 500 fits. Half power DY has MSE 0.1035 and coverage 0.6249; its exact support population target has parameter MSE 0.0874. These intervals use the ordinary sandwich described in Appendix C. The Gaussian high dimensional corrections of the cited DY papers are not applied to this discrete support.

Soft pooling retains the smallest displayed point errors under the original, weaker signal, including the imperfect predictions. Its coverage in the 150 label cases remains around 0.94. The complete results support distinct choices for distinct objectives: an analyst focused on prediction or parameter error may favor pooling in these conditions, while inference on the human logistic target requires its target discrepancy and coverage to be assessed. Existing ridge and one step procedures also handle many finite computation cases.

![Figure 2](figures/figure2_estimates_intervals.png)

Figure 2  Error and coverage answer different questions. The left column uses 20 labels, oracle predictions and the original signal. The right column uses 150 labels, overconfident predictions and threefold utility. The PPI and pooling methods use 600 auxiliary rows and proxy error 0.25; both DY settings use labelled rows alone. MSE axes are logarithmic; annotations show available points out of 500. Coverage is averaged across 11 marginal intervals, with the reference line at 0.95. Underlying aggregate tables retain interval counts, Monte Carlo errors and all remaining conditions.

### 5.3 Power changes availability and error

We replay all 500 draws of the original oracle condition with 20 labels, 600 auxiliary rows and proxy error 0.25. The covariates, generated choices, proxy draws and truth vector are unchanged. All 1000 stored human and unit power support statuses match the replay. On available estimates, the largest relative MSE discrepancy from the original fits is below $4.8\times10^{-8}$. Positive power roots use a scaled score equation, avoiding an absolute stopping tolerance that becomes permissive as power decreases.

Table 3  The same 500 support samples fitted at different powers

| Power | Available | Mean MSE | Median MSE | 90th percentile MSE |
|---|---|---|---|---|
| 0 human | 116/500 | 8.560 | 4.191 | 22.106 |
| 0.01 | 474/500 | 196.879 | 104.474 | 439.843 |
| 0.1 | 474/500 | 14.630 | 7.147 | 27.851 |
| 0.5 | 467/500 | 3.912 | 1.157 | 4.342 |
| 1 | 412/500 | 4.836 | 1.030 | 7.819 |

MSE and its quantiles in Table 3 condition on each method's available estimates. The human MLE is available in 116 repetitions. Powers 0.1 and 0.01 each admit 474 finite roots; the other 26 corrected objectives fail the limiting positive power feasibility check. The 0.01 fits have a median MSE of 104.474 and a 90th percentile of 439.843. Their larger mean error is accompanied by a substantial shift in the typical available fit.

The shared samples isolate the weight comparison from availability selection. Among the same 412 repetitions with an available unit power estimate, mean MSE is 4.836 at unit power, 1.615 at power 0.5, 9.571 at power 0.1 and 153.487 at power 0.01. The paired mean difference between powers 0.01 and one is 148.650, with Monte Carlo standard error 10.180; 94.66% of these samples have greater MSE at the smaller power. Among the 116 samples where the human MLE exists, power 0.01 instead reduces mean MSE from 8.560 to 4.899. The human separation stratum materially changes the comparison.

The choice probabilities also change materially on those same 412 repetitions. Mean expected excess log loss is 0.715 nats per choice at unit power, 0.434 at power 0.5, 1.475 at power 0.1 and 6.866 at power 0.01. This quantity is the Bernoulli Kullback–Leibler divergence from the known choice probabilities, averaged over the fixed 1200 row support. It evaluates the generated choice population exactly; it supplies no additional human validation sample.

![Figure 3](figures/figure3_power_endpoint.png)

Figure 3  Existence and error respond differently to power. The left panel gives the balanced Gaussian boundary at auxiliary to label ratio one. Open markers at zero give positive power right limits; the filled black marker gives exactly zero power. The right panel compares positive powers on the same 412 replayed samples with available unit power fits, using the original 11 contrast support and 20 labels. Both right panel axes are logarithmic.

The existence ordering supports a diagnostic interpretation of power. Lower positive power can restore a finite root while allowing coefficients to move farther along a human separating direction. Existing penalization, one step estimation and pooling remain relevant alternatives, with their full results in Table 2 and the aggregate grids. Availability alone cannot choose among their estimation targets or establish reliable intervals.

## 6 Implications for choice analysis

Prediction assistance introduces a sample allocation decision alongside the utility specification. Theorem 1 shows how a finite auxiliary budget changes the existence region at a given power. Even a correctly calibrated informative predictor can produce a smaller unit power existence region than ordinary logit when the two samples have equal size. An efficiency reference can improve that region, as the displayed balanced example demonstrates. The positive power ordering and its zero discontinuity show why the largest existence region need not give the most accurate coefficients. Data selected power rules require their own distributional analysis.

For a concrete dataset, the support audit answers whether the stated unpenalized objective can have a finite root. Exterior moments identify an optimization problem that additional iterations cannot solve. Boundary proximity and small curvature identify a different numerical situation and remain separately reported. An analyst can then choose an existing alternative with its objective and initializer documented. Clipping, pooling, a penalty and a one step update each define a specific estimation operation.

Reliable intervals add another requirement. The known truth experiments show both high conditional coverage with unavailable estimates and finite point outputs with lower coverage. Reporting availability, conditional coverage and issued and covers makes those differences visible. Hess et al. (2026) distinguish parameter uncertainty, preference heterogeneity and repeated choice dependence; our reported coverage uses independently generated rows under its stated model. This complements work on the sequential decisions of choice modellers (Nova et al., 2025): the fitted objective and convergence criterion belong in the reproducible specification alongside the utility function.

The Gaussian boundary describes a controlled increasing dimension model. Actual DCE attributes are often discrete and choices are repeated within people. Our finite design support calculation applies to a supplied binary logit loss; the phase formula does not convert those datasets into independent Gaussian observations. Extending the probability boundary to clustered or multinomial choice is a separate theoretical task. The current result supplies an explicit baseline and a direct objective audit for the binary case.

## 7 Conclusion

Prediction assisted binary logit has a finite solution boundary determined jointly by labelled and auxiliary samples, utility signal, prediction distribution and correction power. We derive that boundary for a frozen low dimensional predictor with Gaussian remaining attributes and an estimated intercept. Positive power events are nested. Their upper envelope governs strictly positive powers tending to zero and bounds every selection of power under the same signal and estimated intercept assumptions. It exceeds ordinary logit for every fixed strictly interior predictor, including uninformative predictions. The fixed sample coefficient path and replayed choice samples show the estimation cost that can accompany greater availability. The finite design audit checks the objective actually fitted. These results make the consequences of prediction assistance explicit for choice analysts assessing existence, target fidelity and uncertainty.

## Appendix A Proof of the finite ratio boundary

### A1 Oriented intervals and the finite sample event

The gradient condition for (2) is $X^T(wp)-b_\lambda=0$. On a labelled row, multiply its residual numerator by $s_L$. For $Y=0$ the oriented residual ranges from $\lambda g$ to $1-\lambda+\lambda g$; for $Y=1$ it ranges from $\lambda(1-g)$ to $1-\lambda+\lambda(1-g)$. These are exactly $I_L$. Orienting an auxiliary residual by $s_U$ gives $I_U$, using the finite ratio $\tau_m=N/m$.

Let $D$ be the product of these observed intervals and let $K=\operatorname{cone}(D)$. Every labelled lower endpoint is strictly positive. The cone is closed, pointed and polyhedral. Define $V$ with columns $s$ and $sS$, stacked over the two samples, and

$$C=K\cap\ker(V^T). \qquad (A1)$$

Conditional on the low dimensional observations, the oriented remaining attributes form a $q$ by $(m+N)$ independent standard Gaussian matrix $A$. The corrected moment is feasible exactly when $\ker(A)$ meets $C$ at a nonzero point. A nonzero point of $K$ has positive scale and can be divided by that scale to obtain a point of $D$.

Interior feasibility must accompany the rank condition in Proposition 1. Section A3 shows, with probability tending to one, that $\ker(V^T)$ meets the relative interior of $D$. Conditional on this event, generic Gaussian position makes a nontrivial nullspace intersection meet the relative interior of $C$ almost surely. Boundary only tangency occurs on a null set of orientations for a fixed polyhedral cone. Relative interior is preserved under the corresponding linear images and preimages, connecting the cone event to an interior corrected moment. Below the strict critical ratio the positively weighted design has full column rank almost surely. At unit power the cone span has dimension $N+1$ before the two constraints, implying $\kappa_c\leq\tau$; the strict lower side has enough auxiliary rows for full rank. At smaller positive power, all $m+N$ rows carry curvature.

### A2 Statistical dimension identity

For independent $H\sim N_{m+N}(0,I)$, polyhedral polar calculus and the projection identity give

$$\delta(C)=E_H\inf_\eta\|\Pi_K(H-V\eta)\|^2. \qquad (A2)$$

Indeed $C^\circ=K^\circ+\operatorname{range}(V)$; the sum is closed for these polyhedral cones. For any vector $w$,

$$\|\Pi_K(w)\|^2=\sup_{t\geq0}\sum_i\{w_i^2-\operatorname{dist}(w_i,tI_i)^2\}. \qquad (A3)$$

Each summand equals $\sup_{v\in tI_i}(2w_iv-v^2)$. It is convex in $w_i$ and concave in $t$. Its normal expectation at $w_i=H_i-h$ is $B(h,t,I_i)$ from (8). We preserve the infimum followed by the supremum in (A2) and (A3). Passing the random extrema to population expectations requires the following uniform controls.

### A3 Tightness of the intercept and signal multiplier

For a unit vector $u=(u_0,u_1)$ put $h_u(S)=u_0+u_1S$. Let $\rho_m(u)$ be the sum of the labelled and auxiliary interval support functions at $-s h_u$, divided by $m$. Bounded interval endpoints and an envelope proportional to $1+|S|$ give a uniform law of large numbers on the unit circle. The common prediction contributions cancel between the samples, giving

$$\rho(u)=E\{(-h_u(S))_++Yh_u(S)\}. \qquad (A4)$$

For finite $\gamma$, the conditional choice probability lies strictly inside the unit interval. Every nonzero affine $h_u$ is nonzero with positive probability, so (A4) is positive in every unit direction. Continuity on the unit circle yields $c=\min\rho(u)>0$. With probability tending to one, $\inf_u\rho_m(u)\geq c/2$. Positive support in both directions also places zero in the interior of $V^TD$, establishing the relative interior event used in A1.

In (A3), set $t=1$ and choose each interval endpoint maximizing $(-V_i\eta)v_i$. Normal linear terms and bounded squared endpoints then give, uniformly in $\eta$,

$$\sup_t\Psi_m(\eta,t)\geq c\|\eta\|-O_P(1). \qquad (A5)$$

Here $\Psi_m$ is the empirical sum in (A3) divided by $m$. At zero multiplier the value is at most $\|H\|^2/m=O_P(1)$. Empirical minimizers are bounded in probability. The same support argument bounds the population minimizing set. This is the step that controls increasing multiplier values; pointwise convergence alone would not suffice.

### A4 Tightness of the cone scale

On a compact multiplier set write $w_i=H_i-V_i\eta$ and $a_i=\lambda|Y_i-g_i|$ on labelled rows. For $t\geq0$,

$$\operatorname{dist}(w_i,tI_i)^2\geq(ta_i-w_i)_+^2\geq t^2a_i^2/2-w_i^2.$$

The expectation of $a_i^2$ is strictly positive, and its empirical average converges to that value. Uniformly on the compact set the average squared $w_i$ is bounded in probability. Consequently

$$\Psi_m(\eta,t)\leq2m^{-1}\sum_iw_i^2-(t^2/2)m^{-1}\sum_{i\leq m}a_i^2. \qquad (A6)$$

At $t=0$ the objective is zero. Bound (A6) localizes all maximizing scales in probability, with a corresponding population bound. It remains valid at unit power, where the labelled intervals are singletons.

### A5 Limit and probability transition

On compact multiplier and scale sets, continuity and an integrable envelope formed from $H^2$, $S^2$ and bounded endpoints give uniform laws of large numbers for the two independent samples. The change from $\tau_m$ to $\tau$ is uniform there. Bounds (A5) and (A6) localize both extrema, so

$$\inf_\eta\sup_t\Psi_m(\eta,t)\ \to\ \inf_\eta\sup_t\Psi(\eta,t)$$

in probability. The empirical minimum lies between zero and $\|H\|^2/m$, whose second moments are bounded. Uniform integrability upgrades convergence to $L^1$. Taking conditional expectation over $H$ in (A2) yields $\delta(C)/m\to\kappa_c$ in probability.

The approximate conic kinematic theorem of Amelunxen et al. (2014) now applies conditionally to the Gaussian nullspace of $A$. A fixed positive gap between $q/m$ and $\kappa_c$ dominates its order $\sqrt{m+N}$ transition band. The interior and rank qualifications in A1 give the two strict sides of Theorem 1. No equality case follows from this argument.

### A6 Balanced reduction and numerical checks

In the balanced oracle model, sign symmetry makes the saddle objective even in both multiplier coordinates; convexity makes zero optimal. Normal squared distance to each interval gives the three terms in $F$ of (12). At unit power the labelled singleton contribution is $-t^2\epsilon(1-\epsilon)$. Substituting $u=t/\tau$ gives (13). Its large ratio limit follows by expanding $f$ at zero after writing the minimizer on its order $1/\tau$ scale. Since $f(0)=1/2$ and $f'(0)=-2\varphi(0)$, the limit is $2\varphi(0)^2/[\epsilon(1-\epsilon)]$.

Independent numerical formulations check the algebra. The balanced experiment has 100 original moment versus oriented interval LP matches and 80 phase conditions with 500 draws each. The signal experiment has 120 LP matches, 12 direct cone projection comparisons and 80 phase conditions with 300 draws. The imperfect prediction experiment adds 120 LP matches, 12 projection comparisons and 80 phase conditions with 300 draws. All classified phase draws are retained and no case is numerically unresolved. Independent projection errors are below $4.1\times10^{-14}$ after active face refinement; 100 versus 200 node quadrature agrees within $8.1\times10^{-9}$ in the imperfect grid.

One benchmark has an exact classical interpretation. At unit power with $g=1/2$ and a fixed zero intercept, put $Z_G=m^{-1/2}\sum_{i\leq m}(2Y_i-1)X_i$. This is independent standard Gaussian relative to the auxiliary rows. Finite existence is, almost surely, equivalent to

$$ (N/\sqrt{m})Z_G\in\{\sum_{j=1}^N c_jX_{Uj}: -1\leq c_j\leq1\}. \qquad (A7)$$

For $N\geq p$, Kabluchko and Seidel (2020, Theorem 3.3) give this Gaussian zonotope absorption probability, with squared scale $N^2/m$. This symmetric special case is established geometry. A separate 200 case signed cube LP comparison verifies the reduction; the signal, informative prediction and power dependence in the stated corrected model require the full calculation above.

### A7 Scalar affine reallocation

A scalar affine reallocation based only on the conditioned low dimensional observations cannot repair the labelled row infima while preserving the complete objective. Let $c$ be the stacked vector of added linear loss coefficients, measurable with respect to those observations. Objective equality for all fitted coefficients requires $W^Tc=0$. For $q\geq1$, conditional Gaussian independence gives

$$W^Tc\sim N(0,\|c\|^2 I_q). \qquad (A8)$$

A nonzero $c$ has zero probability of satisfying the requirement. Every labelled row at positive power needs a nonzero shift to make its infimum finite. A shift estimated from $W$ changes the conditional independence model. An objective preserving affine repair cannot satisfy the stated low dimensional conditioning.

## Appendix B Equal risk and covariance with distinct existence thresholds

Let $Y$ be balanced and independent of Gaussian attributes. Conditional on an ancillary confidence amplitude $C\in[0,c_{max}]$, $c_{max}<1/2$, a binary proxy flips $Y$ with probability $\epsilon(C)$. Set $g=1/2+C(2Z-1)$ and define

$$v=EC^2,\qquad K=E[C\{1/2-\epsilon(C)\}],\qquad R=1/4+v-2K.$$

The prediction mean is one half, its variance is $v$, and its covariance with $Y$ is $K$. Its Brier risk is $R$. The established fixed dimensional PPI covariance, including the intercept, is

$$16\{1/4-2\lambda K+\lambda^2v(1+1/\tau)\}I/m. \qquad (B1)$$

Matching $v$ and $K$ matches this entire covariance function and its optimal scalar power. Unit power in Theorem 1 instead gives

$$\kappa_c=\tau[1-\min_{u\geq0}\{\tau Ru^2+E h_u(C)\}],$$

$$h_u(c)=f((1/2+c)u)+f((1/2-c)u). \qquad (B2)$$

Proposition 2. At fixed $R$ and $v$, a strictly larger distribution of $C^2$ in convex order gives a strictly smaller boundary in (B2), provided the two distributions differ.

For $u>0$, define $H_u(x)=h_u(\sqrt{x})$. The identities $f''(a)=2\Phi(-a)$ and $f'''(a)=-2\varphi(a)$ imply

$$h_u'''(c)=2u^3\{\varphi((1/2-c)u)-\varphi((1/2+c)u)\}>0.$$

Also $h_u'(0)=0$, so

$$H_u''(x)=\{c h_u''(c)-h_u'(c)\}/(4c^3)>0,\qquad c=\sqrt{x}.$$

The numerator is the integral from zero to $c$ of $h_u''(c)-h_u''(s)$. A Taylor expansion gives its positive continuous limit at zero. Strict convex order increases $E h_u(C)$ for every positive $u$. The bracket in (B2) has negative derivative at zero and grows quadratically, so its minimizer is positive and finite. Evaluating the strict comparison at the larger dispersion minimizer proves the boundary inequality.

An explicit construction sets $c=0.4$, $h=0.499$, $p=c^2/h^2$ and $K=0.9ph/2$. Prediction A has constant $C=c$ and error $1/2-K/c$. Prediction B has $C=h$ with probability $p$ and error 0.05, and $C=0$ otherwise. Both have $v=0.16$, $K=0.14428858$ and $R=0.12142285$. At $\tau=0.05$ the boundaries are 0.0325295 and 0.0305837; at $\tau=1$ they are 0.3598865 and 0.3594589. All predictions lie strictly inside the unit interval. Convex order is an established tool; the proposition describes its consequence for this corrected objective.

## Appendix C Estimation conventions and reproducibility

### C1 Existing power tuning and covariance

For power tuning at an initializer, put $\mu_i=\sigma(X_i^T\beta)$, $T_i=X_i(\mu_i-Y_i)$ on labelled rows and $P_i=X_i(\mu_i-g_i)$ on all rows. The implementation estimates the pooled information matrix $J$ and uses the PPI++ trace rule

$$A=\operatorname{tr}(J^{-1}\widehat{\operatorname{Cov}}(T_L,P_L)_{sym}J^{-1}),$$

$$D=2(1+m/N)\operatorname{tr}(J^{-1}\widehat{\operatorname{Var}}(P)J^{-1}),$$

$$\widehat\lambda=\operatorname{clip}_{[0,1]}(A/D). \qquad (C1)$$

Here the symmetric covariance is the sum of the two cross covariance matrices; its displayed factor two agrees with that definition. The source style pooled prediction variance uses its sample covariance convention. Missing initializers and nonpositive or unresolved denominators remain explicit statuses.

For a solved fixed power objective, labelled scores are $X\{(1-\lambda)\mu-Y+\lambda g\}$ and auxiliary scores are $\lambda X(\mu-g)$. Their separate sample covariance estimates divided by $m$ and $N$, respectively, form the sandwich middle matrix. The bread is the Hessian of (2). Penalized methods add $I/m$ to that Hessian and use the same random score middle matrix. They do not receive a penalty bias correction. The one step diagnostic evaluates the corresponding sandwich at its returned point. It is not an exact root method.

For soft pooling, the middle matrix is the labelled score covariance weighted by $m/(m+N)^2$ plus the auxiliary score covariance weighted by $N/(m+N)^2$, and the bread is the pooled Hessian. A closed form intercept calculation independently verifies both covariance implementations within $4.5\times10^{-16}$. An exterior one dimensional example verifies the ridge root against a separate scalar solver within $1.9\times10^{-10}.$

### C2 DY prior comparator and diagnostic covariance

The DY estimator minimizes the labelled mean logistic loss with $y_i^*=\alpha Y_i+(1-\alpha)/2$. Strictly interior pseudo-responses and full labelled rank give a unique finite solution. Define $s_i=X_i\{\sigma(X_i^T\beta)-y_i^*\}$ and let $J$ be the mean labelled logistic Hessian. The diagnostic covariance is

$$\widehat{\operatorname{Var}}(\beta)=J^{-1}\widehat{\operatorname{Cov}}(s)J^{-1}/m. \qquad (C2)$$

The sample score covariance uses denominator $m-1$. A separate intercept calculation verifies this formula and the analytic pseudo-response root. The loss multiplier $1/\alpha$ in the published prior expression cancels from the sandwich. Shrinkage bias is retained when assessing coverage of the human parameter. These raw estimates are distinct from the authors' high dimensional rescaling. The recorded code comparison pins their repository commit; no external R code is executed.

### C3 Reproduction contents

The compact reproduction package includes the manuscript source, publication figure builder, complete aggregate tables for the reported grids, independently written estimator and boundary code, a finite objective audit with a synthetic CSV example, and pinned source metadata. Public numeric inputs are downloaded separately and verified against Git blob hashes. The loader permits only specified NumPy numeric pickle globals; downloaded research scripts are not executed. Original research inputs and participant level questionnaire files are excluded.

The quick check runs analytic roots, boundary and rank cases, alternative LP formulations, finite dimensional covariance identities, and mathematical reductions. Full commands regenerate the public allocations and every simulation grid. Source hashes distinguish the completed original finite pool run from a later one dimensional covariance compatibility fix that leaves the original 11 dimensional calculations unchanged. The package documents that difference and preserves the completed aggregate outputs. Counts in the paper refer to choice rows, allocations or generated repetitions as specified, and never convert simulated responses into human respondents.

## Appendix D Powers approaching zero

### D1 Existence on human separating directions

Let $h=H^\infty$ and $a=A^\infty$ for the fixed sample losses in Section 3.5. Define the human weak separation cone $S=\{d:h(d)=0\}$. Assume the stacked design has full column rank. For powers strictly below one this is the curvature rank condition; statements of uniqueness at unit power additionally require full auxiliary rank.

Proposition 3. A finite root exists at some sufficiently small positive power exactly when $a(d)>0$ for every nonzero $d\in S$. When a larger positive power already gives a coercive objective, every smaller positive power gives a coercive objective as well.

Equation (15) proves the nesting statement. On $S$, the identity $Q_\lambda^\infty(d)=\lambda a(d)$ gives necessity. For sufficiency, if $S$ meets the unit sphere, continuity and compactness give a positive lower bound for $a$ on that intersection. It remains positive in a neighborhood. Away from this neighborhood $h$ has a positive minimum, while $a$ is bounded below on the sphere. Every sufficiently small positive power makes $(1-\lambda)h+\lambda a$ positive throughout the sphere. If $S=\{0\}$, $h$ is already positive there. Coercivity and full rank then establish the finite root.

For comparison, proper pseudo-label augmentation $H+cB$ has $h\geq0$ and $B^\infty\geq0$. Its zero recession directions and positively weighted design rank are invariant to every $c>0$. Its finite existence event is weight invariant. Synthetic likelihood regularization, including the existence results of Li and Huang (2026), builds on that positive augmentation structure. The subtracted labelled prediction loss in (2) changes the compensation problem. This comparison is an application of classical separation geometry.

### D2 Proof of the right limit

Substitute $v=\lambda t$ in (12). Its minimized expression becomes $G(v)+R_\lambda(v)$, where

$$R_\lambda(v)=(1-\epsilon)f((\lambda^{-1}-1+\epsilon)v)$$

$$\hphantom{R_\lambda(v)}+\epsilon f((\lambda^{-1}-\epsilon)v). \qquad (D1)$$

For $v>0$ this remainder strictly increases with power. The derivative of the full expression at zero is $-2\varphi(0)/\lambda$, so its minimizer is positive. Comparing the minima at two powers proves strict ordering of their boundaries.

The function $G$ is coercive and has $G(0)=1/2+\tau$ and derivative $2\varphi(0)\{2\epsilon(1-\epsilon)-1\}<0$ at zero. Its minimizer is positive and finite. Nonnegativity of $R_\lambda$ bounds the minimum below by $\min G$. Evaluating at a positive minimizer of $G$, where $R_\lambda$ tends to zero, gives the reverse limiting bound. This proves (17), including its strict gap from one half. The proof concerns the stated iterated limit.

### D3 Coefficient growth on a fixed separated sample

If $H$ has no finite stationary point and $Q_\lambda$ has finite roots for all sufficiently small positive powers, the root norm tends to infinity. A bounded subsequence would imply a finite zero of $\nabla H$ by continuity of $(1-\lambda)\nabla H+\lambda\nabla A$.

Under strict human separation and the positive recession condition of Proposition 3, the growth has a more explicit limit. Let $M$ contain vectors with $(2Y_i-1)X_i^Tv\geq1$ for every labelled row. Put

$$V^*=\operatorname{argmin}_{v\in M}a(v). \qquad (D2)$$

Then the distance from $\beta_\lambda/\log(1/\lambda)$ to $V^*$ tends to zero. A unique minimizer in (D2) gives point convergence. The cost $a$ may be negative away from $S$ and is not assumed to be a norm. This adapts the established vanishing regularization and margin argument of Rosset et al. (2004) to the corrected auxiliary recession cost.

To prove the statement, write $L=\log(1/\lambda)$. For a feasible $v$, $H(Lv)\leq\lambda$. The auxiliary softplus remainder gives $A(Lv)=La(v)+r(Lv)$ with $0\leq r\leq\log2$. Thus $Q_\lambda(Lv)/(\lambda L)$ tends to $a(v)$. If a limiting vector violates a margin constraint, its human loss divided by $\lambda L$ diverges; a margin strictly below one supplies the required exponential lower bound. The auxiliary remainder divided by $L$ vanishes uniformly.

The minimizers are bounded on this scale. Positivity of $a$ on the unit human separation cone implies that the minimum of $\{(1-\lambda)h(u)+\lambda a(u)\}/\lambda$ over unit $u$ has a positive lower bound at small powers. A counterexample sequence would converge to a unit vector in $S$ with a nonpositive limiting cost. Since $H\geq h$ and $A\geq a$, comparison with a feasible $Lv$ gives $\|\beta_\lambda\|=O(L)$. Compact subsequences and the preceding lower and upper bounds place every limit in (D2). Positivity on $S$ also makes $V^*$ compact.

For a one coordinate example with all labelled choices and features equal to one, take $g_L=0.75$ and $g_U=0.25$. The exact root is $\log(2/\lambda-1)$ at every positive power. Equal predictions in the two samples instead leave a boundary target at every power; reversing those predictions makes the target exterior. All three examples have full rank auxiliary curvature. A separate two coordinate calculation verifies (D2) by a linear program and analytic scalar score equations. The normalized coefficients approach $(1,1)$, with scaled score error below $1.6\times10^{-15}$ in the displayed path. Root existence and bounded coefficient behavior remain separate properties.

### D4 All powers with signal and estimated intercept

Use the signs, remaining Gaussian matrix $A$ and multiplier columns $V$ of Appendix A. Divide every positive power interval by $\lambda$. Labelled intervals become $[r_i,r_i+1/\lambda-1]$, with $r_i=|Y_i-g_i|>0$. Auxiliary intervals become $J_{Uj}$, using the observed ratio $\tau_m=N/m$. Define $K_\infty$ by the existence of a scale $v\geq0$ satisfying

$$u_{Li}\geq vr_i,\qquad u_{Uj}\in vJ_{Uj}. \qquad (D3)$$

This is a closed pointed polyhedral cone. At zero scale it includes nonnegative labelled vectors with zero auxiliary coordinates. It is the closure of the increasing union of finite positive power cones. For positive scale, all finite labelled coordinates fit in a sufficiently expanded box; zero scale points follow by approximation. Put $C_\infty=K_\infty\cap\ker(V^T)$. A finite root at any positive power supplies a nonzero point of $\ker(A)\cap C_\infty$. An ordinary logit root supplies such a point on the zero scale face. One constrained cone therefore supplies an upper bound simultaneously for all powers, including zero.

Polyhedral polar calculus gives the identity (A2) with $K_\infty$ and $C_\infty$. For $w=H-V\eta$, its projection norm is

$$\|\Pi_{K_\infty}(w)\|^2=\sup_{v\geq0}[\sum_{i\leq m}\{w_i^2-(vr_i-w_i)_+^2\}$$

$$\hphantom{\|\Pi_{K_\infty}(w)\|^2=}+\sum_{j>m}\{w_j^2-\operatorname{dist}(w_j,vJ_{Uj})^2\}]. \qquad (D4)$$

At zero scale the labelled contribution is $(w_i)_+^2$, and its normal expectation is $f(h_L)$. The full normal expectation divided by $m$ is (18).

Both optimization variables need uniform control. Fix reference power $1/2$. The limiting cone contains every smaller positive power cone, each of which contains the reference cone. Projection norm onto a larger closed convex cone is at least that onto the smaller cone. The reference version of (A5) therefore bounds the empirical limiting and all smaller power objectives below by $c\|\eta\|-O_P(1)$. Their values at zero multiplier are at most $\|H\|^2/m$. This localizes their minimizing multipliers in probability. The population argument places all corresponding minimizing sets in a common compact set.

On a compact multiplier set, the labelled distance obeys $(vr_i-w_i)_+^2\geq v^2r_i^2/2-w_i^2$. Its empirical quadratic coefficient converges to $Er^2>0$. This bounds every maximizing scale in probability, with a corresponding population bound. Strictly interior predictions give positive $Er^2$; no deterministic lower bound on $g$ or $1-g$ is required. Compact uniform laws use the integrable $H^2,S^2$ envelope. The empirical minimum is between zero and $\|H\|^2/m$, with bounded second moments. Uniform integrability and conditional expectation over $H$ yield

$$\delta(C_\infty)/m\to\kappa_\infty. \qquad (D5)$$

The conditional conic kinematic bound now excludes a nonzero intersection when $q/m>\kappa_\infty$. Cone inclusion proves the simultaneous upper side of Theorem 2 without a union over powers.

For the lower side, write $v=\lambda t$ in the finite power objective. Its population expression is

$$\Psi_\lambda(\eta,v)=\Psi_\infty(\eta,v)-E f(h_L+v(r+1/\lambda-1)). \qquad (D6)$$

For every positive $v$, the remainder tends to zero by dominated convergence. Positive scales approximate a zero scale by continuity. Consequently $F_\lambda(\eta)=\sup_v\Psi_\lambda(\eta,v)$ increases pointwise to $F_\infty(\eta)=\sup_v\Psi_\infty(\eta,v)$ as positive power decreases. These functions are finite, convex and continuous in $\eta$. On the common compact set containing the minimizers, monotone convergence is uniform by Dini's theorem. This proves $\kappa_c(\lambda)\to\kappa_\infty$ while retaining the order of the two extrema.

At zero multiplier the labelled contribution is at most $1/2$ and the auxiliary contribution at most $\tau$, giving $\kappa_\infty\leq1/2+\tau<1+\tau$. The stacked design has full column rank below the strict boundary. Choose a fixed $\lambda_0\in(0,1)$ with $\kappa<\kappa_c(\lambda_0)$. If $E_m(\lambda)$ denotes a unique finite root, nesting gives

$$P\{E_m(\lambda_m)\}\geq P\{E_m(\lambda_0)\}-P\{\lambda_m>\lambda_0\}. \qquad (D7)$$

The right side tends to one for strictly positive $\lambda_m\to0$ in probability, including data dependent powers. This proves the lower side without a rate requirement.

It remains to establish the strict ordinary logit gap. Let $M(h)=E(H+h)_+=\varphi(h)+h\Phi(h)>0$ and $h=\eta_0+\eta_1S$. Differentiating (18) at zero scale and cancelling the common prediction population terms gives

$$\partial_v\Psi_\infty(\eta,0)=2E\{YgM(h)+(1-Y)(1-g)M(-h)\}>0. \qquad (D8)$$

Every finite multiplier thus gains strictly over its zero scale value $Ef(h_L)$. The coercive bound above ensures a finite minimizer $\eta_\infty$ of $F_\infty$. Evaluating at that minimizer gives $\kappa_\infty=F_\infty(\eta_\infty)>Ef(h_L(\eta_\infty))\geq\kappa_0$, completing the proof. In the balanced specialization, (18) reduces to $1+\tau-G(v)$ and recovers (17).

### D5 Numerical checks and exact zero selection

The general calculation includes 30 signal, sample ratio and prediction conditions. Two hundred versus 400 node quadrature agrees within $1.7\times10^{-9}$. Independent finite differences check its gradient and Hessian, and the zero scale derivative identity agrees within $1.2\times10^{-16}$. All 120 finite power comparisons lie below the corresponding limiting boundary. All 120 original moment and rescaled interval LPs agree. Twenty-four limiting cone projections, retaining the intercept and signal constraints, match an independent full variable quadratic program within $1.6\times10^{-13}$ in squared norm. Feasibility, orthogonality and an independent dual bound verify the values even when a dual optimizer reports precision loss. Both zero scale cases are retained.

For 120 separate finite designs, the limiting LP has 56 interior moments and 64 exterior moments. Each interior case has a positive power witness checked by the original corrected moment LP. Each exterior case has an independent human weak separator with negative corrected auxiliary recession cost. The identity $Q_\lambda^\infty(d)=\lambda a(d)$ agrees within $1.3\times10^{-13}$ on those directions at three powers. These calculations separately check the two structural possibilities of Proposition 3.

A separate analytic scope check uses the excluded hard predictor $g=Z=1-Y$. Here $r=1$, $s_U=s_L$ and $J_U=[-1/\tau,0]$. The scale derivative is $E\{f'(-h_L-v)-f'(-h_L+v/\tau)\}\leq0$, so the limiting supremum is attained at zero and gives $\kappa_0$. The same cone argument applies in this example because $r=1$ retains scale tightness. The uniform strict gain cannot include this hard endpoint.

The general phase experiment generates 6000 independent Gaussian model draws across 12 conditions, with 40 or 80 labels, auxiliary ratios one or four, utility signal 1.5 and three prediction calibrations. Each draw compares three dimensions and five power events, giving 180 aggregate cells with 500 repetitions and 90000 feasibility calls. The raw table retains 12 unknown solver statuses. Reconstructing their exact draws and checking the original corrected moment LP resolves every case with an explicit certificate. Raw and verified tables are separate; no solver status remains unresolved. One positive power call retains its boundary or numerical proximity classification, separately from interior and exterior calls. The paired nesting checks show no violation.

A selector returning exact zero requires a separate statement. In the balanced fixed zero intercept example with $1/2<\kappa<\kappa_c(0+)$, independently choose zero with probability one half and $1/m$ otherwise. Its finite root probability tends to one half; the strictly positive sequence has probability tending to one. Coefficient error and interval behavior also remain separate from the existence theorem.

The earlier balanced validation is retained: 48 independent constrained projections and 6000 original Gaussian draws replayed at powers $1/\sqrt{m}$ and $1/m$, with 12000 new feasibility calculations. All 12 original limiting aggregates agree, and all 36 new report cells retain 500 draws without unresolved states.

## Data availability

The public vaccine inputs are available from the pinned Generative Augmented Inference repository identified in Section 4. The reproduction package contains download instructions, source hashes, analysis code and aggregate outputs. The paper uses existing released choices and simulated outcomes. The separately retained questionnaire application does not enter the estimates reported here.

## References

Amelunxen, D., Lotz, M., McCoy, M.B., Tropp, J.A. (2014). Living on the edge: Phase transitions in convex programs with random data. Information and Inference 3, 224–294. https://doi.org/10.1093/imaiai/iau005

Angelopoulos, A.N., Bates, S., Fannjiang, C., Jordan, M.I., Zrnic, T. (2023). Prediction-powered inference. Science 382, 669–674. Preprint: https://arxiv.org/abs/2301.09633

Angelopoulos, A.N., Duchi, J.C., Zrnic, T. (2024). PPI++: Efficient prediction-powered inference. arXiv:2311.01453, version 2. https://arxiv.org/abs/2311.01453v2

Candès, E.J., Sur, P. (2020). The phase transition for the existence of the maximum likelihood estimate in high-dimensional logistic regression. Annals of Statistics 48, 27–42. https://arxiv.org/abs/1804.09753

Hess, S., Daly, A., Bliemer, M., Guevara, A., Daziano, R., Dekker, T. (2026). Statistical significance in choice modelling: computation, usage and reporting. Journal of Choice Modelling 60, 100611. https://doi.org/10.1016/j.jocm.2026.100611

Kabluchko, Z., Seidel, H. (2020). Convex cones spanned by regular polytopes. arXiv:2006.09492, version 2. https://arxiv.org/abs/2006.09492v2

Kiryo, R., Niu, G., du Plessis, M.C., Sugiyama, M. (2017). Positive-Unlabeled Learning with Non-Negative Risk Estimator. NIPS 2017 camera ready preprint, arXiv:1703.00593v2. https://arxiv.org/abs/1703.00593v2

Koriyama, T., Bellec, P.C. (2025). Phase transitions for the existence of unregularized M-estimators in single index models. Proceedings of Machine Learning Research 267, 31519–31540. https://proceedings.mlr.press/v267/koriyama25a.html

Li, W., Huang, D. (2026). Regularization Using Synthetic Data for High-Dimensional Inference. arXiv:2407.04194, version 5. https://arxiv.org/abs/2407.04194v5

Lu, C., Wang, M., Zhang, D., Zhang, H. (2026). Generative Augmented Inference. Proceedings of Machine Learning Research 306, 81730–81751. https://proceedings.mlr.press/v306/lu26am.html

Mani, P., Xu, P., Lipton, Z.C., Oberst, M. (2026). No Free Lunch: Non-Asymptotic Analysis of Prediction-Powered Inference. arXiv:2505.20178, version 2. https://arxiv.org/abs/2505.20178v2

Natarajan, N., Dhillon, I.S., Ravikumar, P.K., Tewari, A. (2013). Learning with Noisy Labels. Advances in Neural Information Processing Systems 26, 1196–1204. https://papers.nips.cc/paper_files/paper/2013/hash/3871bd64012152bfb53fdf04b401193f-Abstract.html

Nova, G., van Cranenburgh, S., Hess, S. (2025). Understanding the decision-making process of choice modellers. Journal of Choice Modelling 56, 100562. https://doi.org/10.1016/j.jocm.2025.100562

Rosset, S., Zhu, J., Hastie, T. (2004). Boosting as a Regularized Path to a Maximum Margin Classifier. Journal of Machine Learning Research 5, 941–973. https://www.jmlr.org/papers/v5/rosset04a.html

Sfeir, G., Nova, G., Hess, S., van Cranenburgh, S. (2026). Can large language models assist choice modelling? Insights into prompting strategies and current models’ capabilities. Journal of Choice Modelling 60, 100623. https://doi.org/10.1016/j.jocm.2026.100623

Sterzinger, P. (2026). Proportional-limit asymptotics for Diaconis–Ylvisaker-penalised logistic regression with fitted intercept. arXiv:2609.09831, version 1. https://arxiv.org/abs/2609.09831v1

Sterzinger, P., Kosmidis, I. (2026). Diaconis–Ylvisaker prior penalized likelihood for p/n → κ ∈ (0,1) logistic regression. Biometrika 113(2), asag014. https://doi.org/10.1093/biomet/asag014

van Cranenburgh, S., Wang, S., Vij, A., Pereira, F., Walker, J. (2022). Choice modelling in the age of machine learning—Discussion paper. Journal of Choice Modelling 42, 100340. https://doi.org/10.1016/j.jocm.2021.100340
