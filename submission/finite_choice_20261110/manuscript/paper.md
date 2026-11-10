# When prediction assisted logit estimates exist

## Abstract

Predicted choices can supplement a limited set of observed choices through a correction that preserves the population score. The corrected empirical logit objective can have no finite minimizer. We derive its existence boundary when the labelled sample, auxiliary sample and attribute dimension grow together. The boundary accommodates a utility signal, an estimated intercept and a frozen predictor whose calibration may be imperfect. It quantifies the joint roles of correction power and the auxiliary to label ratio. In a balanced benchmark with equal sample sizes and proxy error 0.25, unit power admits a critical dimension to label ratio of 0.355; the established variance reference power gives 0.603. A released vaccine choice reanalysis supplies verified examples of exterior corrected moments. Simulations on its attribute support retain seven established estimation strategies and report estimate availability, interval issuance, coverage and error separately. A finite objective audit identifies which stated estimators can be computed on a given sample. The results connect prediction assistance to the existence of choice estimates and distinguish that requirement from reliable uncertainty quantification.

Keywords: binary choice; prediction powered inference; logistic separation; finite auxiliary samples; existence of estimates

## 1 Introduction

Choice analysts increasingly use machine predictions in model development and estimation. Recent work in the Journal of Choice Modelling examines both machine learning in choice analysis and language models as assistants for utility specification (van Cranenburgh et al., 2022; Sfeir et al., 2026). A related use supplies predicted responses for observations without human choice labels. Prediction powered inference corrects these responses using a labelled sample, allowing the population estimating equation to retain its human choice target (Angelopoulos et al., 2023, 2024).

An analyst must also be able to solve the empirical estimating equation. A corrected logit objective can decrease indefinitely along a coefficient direction. Continued optimization then produces increasingly extreme coefficients without reaching a finite optimum. This event is consequential when the human label budget is small relative to the utility specification. Prediction quality, correction power and the number of auxiliary observations all enter the event.

We characterize the finite solution boundary of a specified unpenalized, prediction assisted binary logit objective. Our main result treats an estimated intercept, one Gaussian utility signal and an increasing number of independent Gaussian attributes. The labelled and auxiliary samples grow at a finite ratio. Predictions are frozen functions of the signal and auxiliary information, and can be miscalibrated. An explicit variational expression gives the critical attribute dimension per labelled observation. A balanced specialization reduces the calculation to one scalar optimization.

The result explains a distinction between efficiency calculations and existence calculations. At proxy error 0.25 and equal labelled and auxiliary sample sizes, the balanced unit power boundary is 0.355. Its infinite auxiliary limit is 0.849. Applying the established population variance reference power changes the finite ratio boundary to 0.603. These values describe a Gaussian benchmark. The accompanying finite sample audit uses the actual design matrix and remains applicable to categorical attributes.

The theoretical tools have clear antecedents. Logistic separation and ordinary logit existence thresholds are established (Candès and Sur, 2020). Statistical dimension locates transitions in random convex feasibility problems (Amelunxen et al., 2014). Koriyama and Bellec (2025) give general existence results for convex M estimators with finite rowwise loss infima. Positive power correction creates labelled row losses with infimum minus infinity; finite estimability depends on compensation by other rows. We derive the explicit finite auxiliary boundary for that corrected objective. Power tuning and one step estimation come from PPI++ and remain comparison methods throughout the paper.

Corrected empirical risk can become negative in positive unlabelled learning (Kiryo et al., 2017). Finite sample estimation costs of PPI++ are also established for mean estimation (Mani et al., 2026). Here the object of analysis is the finite minimizer of the corrected nonlinear choice loss. The two sample geometry gives its explicit probability boundary.

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

Figure 1  Finite fit frequencies for overconfident and shifted predictions at auxiliary to label ratio one. Circles and dashed lines use 40 labels; squares and solid lines use 80. Vertical dotted lines give the asymptotic boundaries for unit and variance reference powers. Error bars are approximate 95% binomial Monte Carlo intervals for each point. Attributes are Gaussian and the utility signal is 1.5; these are theoretical benchmark simulations.

The boundary can retain predictor information beyond its Brier risk and fixed dimensional covariance. Appendix B constructs two balanced predictions with the same mean, risk and complete covariance function for every fixed power, yet different existence thresholds. Greater dispersion of squared prediction confidence lowers the threshold in that construction. Its gap is about 6% at auxiliary to label ratio 0.05 and 0.12% at ratio one. This consequence distinguishes the summaries mathematically; the displayed effect at larger auxiliary ratios is small.

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

For each method we record point availability and interval availability separately. Parameter MSE is the mean of squared coordinate errors, conditional on an available point. Coverage is the fraction of 11 marginal intervals covering their corresponding true coordinates, averaged over repetitions with intervals. Issued and covers assigns zero to a repetition without intervals before averaging over all 500 repetitions. Neither summary is simultaneous coverage. Monte Carlo errors use the repetition as the independent unit. Wald intervals use the implemented score sandwich; ridge intervals do not adjust for shrinkage bias. Appendix C states the covariance formulas and tuning convention.

### 5.2 What a finite output establishes

Table 2 contrasts two prespecified mechanisms: scarcity of human labels with oracle predictions, and overconfidence under stronger utility at a larger label budget. All conditions and error tails remain in the reproduction tables. The selected columns show how availability, parameter error and coverage can lead to different assessments of the same estimator.

Table 2  Point availability and interval performance with 600 auxiliary rows

| Condition | Estimator | Available | MSE | Coverage | Issued and covers |
|---|---|---|---|---|---|
| 20 labels oracle | Human MLE | 116/500 | 8.5602 | 0.926 | 0.215 |
| 20 labels oracle | Unit PPI | 412/500 | 4.8362 | 0.998 | 0.822 |
| 20 labels oracle | Power human start | 116/500 | 6.0858 | 0.907 | 0.210 |
| 20 labels oracle | Power ridge start | 348/500 | 21.5408 | 0.801 | 0.558 |
| 20 labels oracle | One step | 500/500 | 1.5054 | 0.827 | 0.827 |
| 20 labels oracle | Ridge PPI | 500/500 | 0.2193 | 0.961 | 0.961 |
| 20 labels oracle | Soft pooling | 500/500 | 0.0057 | 0.942 | 0.942 |
| 150 labels overconfident | Human MLE | 500/500 | 0.1353 | 0.947 | 0.947 |
| 150 labels overconfident | Unit PPI | 500/500 | 0.1326 | 0.971 | 0.971 |
| 150 labels overconfident | Power human start | 500/500 | 0.1050 | 0.942 | 0.942 |
| 150 labels overconfident | Power ridge start | 500/500 | 0.1042 | 0.942 | 0.942 |
| 150 labels overconfident | One step | 500/500 | 0.0992 | 0.945 | 0.945 |
| 150 labels overconfident | Ridge PPI | 500/500 | 0.0860 | 0.958 | 0.958 |
| 150 labels overconfident | Soft pooling | 500/500 | 0.0353 | 0.804 | 0.804 |

With 20 labels and oracle predictions, the human MLE is available in 116 of 500 repetitions. Unit power PPI returns 412 points. Its conditional coverage is 0.998, while issued and covers is 0.822. One step PPI returns 500 points and has coverage 0.827. Ridge PPI and oracle pooling also return every point and have coverage near 0.95 in this condition. The oracle pooling MSE advantage is substantial and is preserved in the comparison.

Under stronger utility, 150 labels and overconfident predictions, all seven estimators return 500 points and intervals. Soft pooling has MSE 0.0353 and coverage 0.8036. Ridge initialized power tuning has MSE 0.1042 and coverage 0.9424; one step PPI has MSE 0.0992 and coverage 0.9451. Exact finite support expectations give a soft pooling population parameter MSE of 0.0181 in this condition. The fixed power corrected population score still vanishes at the true parameter. This comparison isolates the difference between a changed target and corrected estimation uncertainty.

Soft pooling retains the smallest displayed point errors under the original, weaker signal, including the imperfect predictions. Its coverage in the 150 label cases remains around 0.94. The complete results support distinct choices for distinct objectives: an analyst focused on prediction or parameter error may favor pooling in these conditions, while inference on the human logistic target requires its target discrepancy and coverage to be assessed. Existing ridge and one step procedures also handle many finite computation cases. The experiments establish no uniformly best estimator.

![Figure 2](figures/figure2_estimates_intervals.png)

Figure 2  Error and coverage answer different questions. The left column uses 20 labels, oracle predictions and the original signal. The right column uses 150 labels, overconfident predictions and threefold utility. Both use 600 auxiliary rows and proxy error 0.25. MSE axes are logarithmic; annotations show available points out of 500. Coverage is averaged across 11 marginal intervals, with the reference line at 0.95. Underlying aggregate tables retain interval counts, Monte Carlo errors and all remaining conditions.

## 6 Implications for choice analysis

Prediction assistance introduces a sample allocation decision alongside the utility specification. Theorem 1 shows how a finite auxiliary budget changes the existence region at a given power. Even a correctly calibrated informative predictor can produce a smaller unit power existence region than ordinary logit when the two samples have equal size. An efficiency reference can improve that region, as the displayed balanced example demonstrates. Data selected power rules require their own distributional analysis.

For a concrete dataset, the support audit answers whether the stated unpenalized objective can have a finite root. Exterior moments identify an optimization problem that additional iterations cannot solve. Boundary proximity and small curvature identify a different numerical situation and remain separately reported. An analyst can then choose an existing alternative with its objective and initializer documented. Clipping, pooling, a penalty and a one step update each define a specific estimation operation.

Reliable intervals add another requirement. The known truth experiments show both high conditional coverage with unavailable estimates and finite point outputs with lower coverage. Reporting availability, conditional coverage and issued and covers makes those differences visible. This complements work on the sequential decisions of choice modellers (Nova et al., 2025): the fitted objective and convergence criterion belong in the reproducible specification alongside the utility function.

The Gaussian boundary describes a controlled increasing dimension model. Actual DCE attributes are often discrete and choices are repeated within people. Our finite design support calculation applies to a supplied binary logit loss; the phase formula does not convert those datasets into independent Gaussian observations. Extending the probability boundary to clustered or multinomial choice is a separate theoretical task. The current result supplies an explicit baseline and a direct objective audit for the binary case.

## 7 Conclusion

Prediction assisted binary logit has a finite solution boundary determined jointly by the labelled sample, finite auxiliary sample, utility signal, prediction distribution and correction power. We derive that boundary for a frozen low dimensional predictor with Gaussian remaining attributes and an estimated intercept. The finite design audit translates existence into a check on the objective being fitted. Released choices and known truth experiments demonstrate why availability, target fidelity and uncertainty need separate evidence. Together these results make the estimation consequences of prediction assistance explicit for choice analysts.

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

Independent numerical formulations check the algebra. The balanced experiment has 100 original moment versus oriented interval LP matches and 80 phase conditions with 500 draws each. The signal experiment has 120 LP matches, 12 direct cone projection comparisons and 80 phase conditions with 300 draws. The imperfect prediction experiment adds 120 LP matches, 12 projection comparisons and 80 phase conditions with 300 draws. All classified phase draws are retained and no case is numerically unresolved. Independent projection errors are below $4.1\times10^{-14}$ after active face refinement; 100 versus 200 node quadrature agrees within $8.1\times10^{-9}$ in the imperfect grid. These checks test the stated calculations and do not establish priority or replace the proof.

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

### C2 Reproduction contents

The compact reproduction package includes the manuscript source, publication figure builder, complete aggregate tables for the reported grids, independently written estimator and boundary code, a finite objective audit with a synthetic CSV example, and pinned source metadata. Public numeric inputs are downloaded separately and verified against Git blob hashes. The loader permits only specified NumPy numeric pickle globals; downloaded research scripts are not executed. Original research inputs and participant level questionnaire files are excluded.

The quick check runs analytic roots, boundary and rank cases, alternative LP formulations, finite dimensional covariance identities, and mathematical reductions. Full commands regenerate the public allocations and every simulation grid. Source hashes distinguish the completed original finite pool run from a later one dimensional covariance compatibility fix that leaves the original 11 dimensional calculations unchanged. The package documents that difference and preserves the completed aggregate outputs. Counts in the paper refer to choice rows, allocations or generated repetitions as specified, and never convert simulated responses into human respondents.

## Data availability

The public vaccine inputs are available from the pinned Generative Augmented Inference repository identified in Section 4. The reproduction package contains download instructions, source hashes, analysis code and aggregate outputs. The paper uses existing released choices and simulated outcomes. The separately retained questionnaire application does not enter the estimates reported here.

## References

Amelunxen, D., Lotz, M., McCoy, M.B., Tropp, J.A. (2014). Living on the edge: Phase transitions in convex programs with random data. Information and Inference 3, 224–294. https://doi.org/10.1093/imaiai/iau005

Angelopoulos, A.N., Bates, S., Fannjiang, C., Jordan, M.I., Zrnic, T. (2023). Prediction-powered inference. Science 382, 669–674. Preprint: https://arxiv.org/abs/2301.09633

Angelopoulos, A.N., Duchi, J.C., Zrnic, T. (2024). PPI++: Efficient prediction-powered inference. arXiv:2311.01453, version 2. https://arxiv.org/abs/2311.01453v2

Candès, E.J., Sur, P. (2020). The phase transition for the existence of the maximum likelihood estimate in high-dimensional logistic regression. Annals of Statistics 48, 27–42. https://arxiv.org/abs/1804.09753

Kiryo, R., Niu, G., du Plessis, M.C., Sugiyama, M. (2017). Positive-Unlabeled Learning with Non-Negative Risk Estimator. NIPS 2017 camera ready preprint, arXiv:1703.00593v2. https://arxiv.org/abs/1703.00593v2

Koriyama, T., Bellec, P.C. (2025). Phase transitions for the existence of unregularized M-estimators in single index models. Proceedings of Machine Learning Research 267, 31519–31540. https://proceedings.mlr.press/v267/koriyama25a.html

Lu, C., Wang, M., Zhang, D., Zhang, H. (2026). Generative Augmented Inference. Proceedings of Machine Learning Research 306, 81730–81751. https://proceedings.mlr.press/v306/lu26am.html

Mani, P., Xu, P., Lipton, Z.C., Oberst, M. (2026). No Free Lunch: Non-Asymptotic Analysis of Prediction-Powered Inference. arXiv:2505.20178, version 2. https://arxiv.org/abs/2505.20178v2

Nova, G., van Cranenburgh, S., Hess, S. (2025). Understanding the decision-making process of choice modellers. Journal of Choice Modelling 56, 100562. https://doi.org/10.1016/j.jocm.2025.100562

Sfeir, G., Nova, G., Hess, S., van Cranenburgh, S. (2026). Can large language models assist choice modelling? Insights into prompting strategies and current models’ capabilities. Journal of Choice Modelling 60, 100623. https://doi.org/10.1016/j.jocm.2026.100623

van Cranenburgh, S., Wang, S., Vij, A., Pereira, F., Walker, J. (2022). Choice modelling in the age of machine learning—Discussion paper. Journal of Choice Modelling 42, 100340. https://doi.org/10.1016/j.jocm.2021.100340
