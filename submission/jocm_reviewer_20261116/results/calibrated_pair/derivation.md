# Calibrated predictors with equal regular information

The example sharpens Appendix B4 of *When prediction assisted logit estimates exist*. It holds the target choice model, the observed human design and conditional calibration fixed. Its object is the existing compensated logit loss. It introduces no estimator or calibration method.

The posterior signals are frozen functions constructed from the known generating law. In particular, the nonzero-signal example uses the generating mu(S), without estimating it from labels. This is an idealized counterexample, not a fitted prediction pipeline or a feasible data-driven rule for selecting predictors.

## Conditional construction

Let X=(1,S,W), with S and the remaining coordinates W independent standard Gaussian variables. Let Y conditional on S have probability mu(S)=sigma(gamma S). Fix 0<h<e<1/2. A binary symmetric signal with error e has marginal success probability q_e=e+(1-2e)mu and posterior probabilities

    g_e(1)=(1-e)mu/q_e,
    g_e(0)=e mu/(1-q_e).

The conditional posterior variance is

    V_e(mu)=(1-2e)^2 mu^2(1-mu)^2 / {q_e(1-q_e)}.

Prediction A uses this moderate signal on every row. Prediction B independently selects a strong signal with error h with probability

    p(mu)=V_e(mu)/V_h(mu)
         ={(1-2e)/(1-2h)}^2 q_h(1-q_h)/{q_e(1-q_e)},

and returns mu otherwise. The selector is independent of Y conditional on S. The strong binary experiment can be garbled into the moderate one: add an independent flip with probability (e-h)/(1-2h). The law of total variance then gives V_h>=V_e, so 0<=p<=1. This also follows from the displayed formula. Selection is allowed to depend on the observed S, and is included among the predictor's observed inputs.

For each predictor, E(Y|S,g)=g. Conditional calibration follows separately on each signal component and on the uninformative component; a mixture of components with the same posterior value preserves it. Also E(g|S)=mu and Var(g|S)=V_e for both predictors. Their joint construction can use identical X,Y and independent proxy errors and selectors. Ordinary human separation is then exactly the same event on each realized pair of samples.

## Complete regular covariance

Put A=mu-Y and B=mu-g. Conditional calibration gives

    E(A|S)=E(B|S)=0,
    E(A^2|S)=mu(1-mu),
    E(AB|S)=E(B^2|S)=V_e(mu),
    E{(Y-g)^2|S}=mu(1-mu)-V_e(mu).

These identities also hold conditional on the full fitted X, because W is independent of the signal construction conditional on S. Thus matching V_e pointwise matches every entry of the truth, prediction and cross score covariance matrices, including fitted intercept and signal, as well as the independent Gaussian slopes. This is stronger than matching their traces or one selected coefficient variance.

Let H=E{XX' mu(1-mu)} and M=E{XX' V_e(mu)}. In the usual fixed-dimensional regular limit for the existing loss, the covariance times the number m of labelled rows is

    H^{-1} [ H - {2 lambda - lambda^2(1+1/tau)} M ] H^{-1}.

Both predictors have exactly the same function of lambda, the same conditional and marginal Brier risk, and the same variance reference power lambda=tau/(1+tau). The calibration identity and variance-optimal power are prior machinery. This expression is a regular reference; it supplies no proportional-dimensional Wald guarantee.

## An analytic balanced counterexample

At gamma=0 choose a confidence amplitude C, use signal error 1/2-C, and set g=1/2+C(2Z-1). Then the predictor is calibrated. Fix c=0.4 and b=0.499. Predictor A has C=c; predictor B has C=b with probability p=c^2/b^2 and C=0 otherwise. Both have v=Cov(Y,g)=0.16 and Brier risk R=0.09. Their full regular covariance functions agree.

At unit power the existing boundary reduces to

    kappa_c=tau [1-min_{u>=0}{tau R u^2 + E h_u(C)}],
    h_u(c)=f((1/2+c)u)+f((1/2-c)u),
    f(a)=E(G-a)_+^2, G standard normal.

For positive u, H_u(x)=h_u(sqrt(x)) is strictly convex. One verification uses

    h_u'''(c)=2u^3{phi((1/2-c)u)-phi((1/2+c)u)}>0,
    H_u''(x)={c h_u''(c)-h_u'(c)}/(4c^3)>0.

The latter has a positive continuous extension at zero. The dispersed C^2 has the same mean and a strictly larger expectation of H_u for every u>0. Each scalar objective has a negative derivative at zero and grows quadratically; its minimizer is positive and finite. Evaluating both objectives at the dispersed minimizer proves kappa_A>kappa_B. Choose any dimension ratio strictly between them. The existing boundary theorem then gives opposite probability limits for finite attainment on the same human choice law, with conditional calibration, Brier risk and the complete regular covariance function matched. This analytic result is a calibrated instance of the existing convex-order proposition, not a new general ordering theorem.

At tau=0.1 the computed unit-power boundaries are 0.0616360691 and 0.0592898088. At tau=4 and power 0.8 their computed order reverses: 1.0953280048 and 1.1117059233. This rules out a universal predictor ordering across powers for this pair. The latter values are floating-point calculations, independently integrated; they are not rigorous interval bounds.

## Nonzero signal and finite-size evidence

For gamma=1, e=0.05, h=0.001 and tau=4, the common variance power is 0.8. The computed boundaries are 1.1886304119 for A and 1.2192432437 for B. The conditional Brier risk averages to 0.0451020062 for both. The full covariance block difference is below 7.5e-16 in the numerical evaluation. A fixed dimension ratio 1.2 lies between the computed boundaries. Nonzero-signal values use the existing nuisance-adjusted saddle profile. Independent adaptive integration with scalar nested roots agrees with the Hermite calculation to below 2.5e-15 on the selected verification cases. Such numerical agreement verifies implementation; it does not certify arithmetic intervals or priority.

Classification accuracy and logarithmic prediction loss differ between these predictors. For the selected nonzero-signal example the computed log losses are about 0.1794 and 0.1359. The matched claims concern calibration, Brier risk and score covariance, and do not assert equality of all prediction quality measures.

The 36 exploratory predictor pairs, all 216 quadrature records and all finite-size conditions remain. The selected example maximized the absolute gap on that exploratory grid. It illustrates a possible mechanism, with no prevalence claim. Each finite-size condition has 200 independent paired draws; different dimension conditions share nuisance columns within a draw. The ratio below counts the Gaussian nuisance slopes, with a fitted intercept and signal added to both designs.

| Labelled rows | Nuisance dimension / labels | A feasible | B feasible | B minus A | Paired Monte Carlo SE |
| --- | --- | --- | --- | --- | --- |
| 40 | 1.05 | 123/200 | 152/200 | 0.145 | 0.0445 |
| 40 | 1.20 | 97/200 | 111/200 | 0.070 | 0.0462 |
| 40 | 1.35 | 78/200 | 66/200 | -0.060 | 0.0423 |
| 80 | 1.05 | 150/200 | 165/200 | 0.075 | 0.0388 |
| 80 | 1.20 | 106/200 | 107/200 | 0.005 | 0.0499 |
| 80 | 1.35 | 62/200 | 51/200 | -0.055 | 0.0455 |

The interval LP reports feasibility; the boundary event has probability zero under the continuous full-rank law. Sixty independent small-design original weighted-support programs crosscheck its interpretation as finite attainment, with 56 interior and 4 exterior cases and no unresolved classifications. All 2400 finite-size classifications are retained. In the central ratio the 80-label frequencies are nearly equal, and the upper ratio reverses the observed ordering at both sizes. These runs do not establish a practically large or uniform small-sample advantage. They preserve the difference between a population counterexample and a useful empirical predictor-selection rule.

## Consequence for the paper

The calibrated construction removes miscalibration as an explanation of the matched-information counterexample. It belongs within the same attainment argument. The finite-size results limit its practical force. It strengthens a precise theoretical statement, while the large samplewise score-equivalence example remains a distinct, already documented illustration. The current 58-page manuscript stays frozen until this extension receives a concise editorial decision. Generic calibration, optimal weighting, convex order, the Gaussian absorption step and the ordinary separation problem retain their attribution. Journal-level novelty and sufficient value require an assessment of this specific consequence against the closest existing results.
