# Predictor–attribute alignment: derivation before numerical evaluation

10 November 2026. This continues the same compensated binary-logit objective. The previous turn made progress by completing the1000case classical finite-power-domain audit and its backup. This record tests the scientific scope of the remaining explicit-boundary contribution. It is not a new behavioral framework or an acceptance assessment.

## A matched question

Let T and R be independent standard normals, Y an independent fair binary response, and g=1/2+c sign(T), where0<c<1/2. Keep g,Y,T and every existing Gaussian noise column W identical within a comparison. An external-information model uses contrasts(1,R,W); an attribute-aligned model uses(1,T,W). Both have the same total number of coefficients and the same marginal contrast law. The complete joint law of(Y,g), prediction risk and confidence distribution are exactly equal, not approximately matched by simulation.

The human population target is zero in both models. The ordinary existence boundary is1/2 in either case. The prediction contains no information about Y. It represents irrelevant variation carried by a predictor; it is a controlled mechanism example, not an empirical model of the private questionnaires or a claim about all trained models.

In the external case the existing theorem applies directly, treating T as external prediction information. In the aligned case T is a fitted model column and must be retained as a constrained index before conditioning on g. The remaining Gaussian columns are still independent. The low-dimensional matrix V has columns s and sT, with s=1−2Y on labelled rows and s=1 on auxiliary rows. The oriented intervals are the same as before: I_L=[lambda r,1−lambda+lambda r], r=|Y−g|, and I_U=(lambda/tau)[−g,1−g].

The candidate aligned threshold is inf_(eta0,etaT) sup_t E B(s_L(eta0+etaT T),t,I_L)+tau E B(eta0+etaT T,t,I_U). The external threshold restricts etaT=0. Thus the aligned value cannot exceed the external value. The existing proof extends by retaining the actual index in V: interval-support cancellation still yields E[(-h)_++Yh]>0 for each nonzero affine h(T), finite Gaussian moments localize the multipliers, and E r²>0 localizes the scale. Conditional random-subspace geometry then applies to the remaining columns. Counting T as one fitted index versus one independent Gaussian noise direction changes dimension by one, not the proportional limit.

## Strict difference at unit power

Joint sign symmetry gives optimal eta0=0 in both cases. Put a=1/2+c,b=1/2−c and R0=1/4+c². The external unit-power threshold is

kappa_external=tau {1−min_(u>=0)[tau R0 u²+f(a u)+f(b u)]}.

The scale optimizer t=tau u is strictly positive and finite. At eta0=etaT=0, the derivative in the T multiplier of the full aligned saddle envelope is

E|T| {−2tc+tau[f'(ta/tau)−f'(tb/tau)]}.

For x>0, f''(x)=2 Phi(−x)<1. Integrating f'' between tb/tau and ta/tau shows that the bracket is strictly negative for every t>0 and c>0. The scale objective is strictly concave at its optimizer, so the envelope derivative has the displayed strictly negative value. A small positive etaT lowers the maximum, proving kappa_aligned<kappa_external at unit power. This conclusion concerns exactly matched prediction marginals and a different prediction–attribute dependence, not better prediction quality.

The generic principle that feature/teacher overlap matters is established in high-dimensional inference. The contribution assessment must check that lineage; this construction does not by itself establish priority. The result, if numerically confirmed, strengthens the corrected-loss model's scope and makes one omitted dependence explicit. It does not solve cross-fitting, imply validity for arbitrary trained black boxes, or prove a new variance or interval result.

## Independently trained linear predictors

For isotropic Gaussian audit rows independent of a training sample, condition on a fitted linear prediction vector theta. Rotate the remaining Gaussian columns so its component orthogonal to the utility direction is one fitted Gaussian index T. A logistic predictor then has the form sigma(aS+bT+d), with a,b,d fixed conditional on training. Additional external proxy information may be retained separately. This elementary rotation justifies a conditional finite-index formulation. A deterministic unconditional threshold further requires convergence of these trained overlaps and uniform localization; neither arbitrary cross-fitting nor a growing index dimension follows automatically.

Cross-prediction is already an established method: Zrnic and Candes(2024), arXiv2309.16598v3, Sections2,3 and5. Its Section2 explicitly fixes the target dimension; it does not by itself supply this proportional finite-root calculation. Koriyama–Bellec(2025) already gives a general single-index existence theory under its stated row-loss assumptions. The current model must retain these attributions; a conditional rotation is not claimed as an independent new training algorithm.

## Evaluation boundary

First evaluate a fixed grid c in{0.1,0.25,0.4},tau in{1,4},lambda in{1,0.5,0.1}. Use symmetric quadrature on the positive and negative halves of the normal index, avoiding a discontinuous sign function at a quadrature node. Compare80and160nodes on each half truncated at12standard deviations, and record the tail bound. Check independent closed-form external unit boundaries, the analytic negative derivative, finite differences, and original corrected-moment versus oriented-box feasibility for120finite paired designs. Keep every numerical failure before any refinement. A further phase experiment will be planned only after this population calculation quantifies the gap; it must retain both paired estimators and all chosen cells, not select repetitions by outcome.
