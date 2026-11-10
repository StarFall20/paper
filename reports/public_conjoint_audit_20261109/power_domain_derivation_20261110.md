# Finite-design admissible correction-power domain: derivation before checks

10 November2026. Same corrected loss, fixed predictions and observed design. This is a classical parametric-support LP consequence, not a new inferential tuning rule or a general conic theorem. The preceding goal turn made progress by adding nearest-prior attribution, an affine-reallocation obstruction and the established DY comparator.

Assume m,N>0 and full column rank of the stacked design X. Unique finite solutions at unit power additionally require full auxiliary rank. At zero power use the ordinary labelled support/rank test separately. No Gaussian or independent-choice sampling assumption is needed for this deterministic objective statement.

Write m*b_lambda=b0+lambda*b1, where b0=X_L^T*Y and b1=−X_L^T*g_L+(m/N)X_U^T*g_U. Set u_i=m*w_i*p_i. The closed moment constraints are

X^T*u=b0+lambda*b1;
0<=u_L<=1−lambda;
0<=u_U<=lambda/tau_m;
0<=lambda<=1.

These are jointly linear in u and lambda. Let L be the maximum lambda over this compact feasible polytope. Zero is always feasible using u_L=Y,u_U=0. A second LP maximizes t>=0 with t<=u_i and u_L+t<=1−lambda,u_U+t<=lambda/tau_m, retaining the equality and lambda bounds. Positive t supplies an interior box witness at some0<lambda<1; because X has full column rank, its image moment is interior and the loss has a unique finite root.

If the maximum t is zero, no interior-power root exists. A unique unit-power root would imply coercivity at smaller positive powers by the existing nesting identity,contradicting this result. Thus no positive power has a unique finite root. L alone cannot decide this case:the existing one-coordinate equal-prediction example is feasible at every power but remains on its moment boundary.

If t is positive, the entire interior-power finite-root domain is exactly0<lambda<L. For powers below the interior witness, use fixed-sample nesting. Between the witness and a feasible maximizer L, convex mixing preserves the moment equality and makes every box inequality strict. If L<1, an interior moment at L would give a positive minimum recession cost on the unit direction sphere. Continuity in power preserves this margin just above L,contradicting maximality. Hence the endpoint L<1 has no finite root. If L=1, check its auxiliary rank and interior moment separately; both inclusion and exclusion of the unit endpoint occur in simple scalar examples.

The closed endpoint has an independent recession description. Write h=H^infty>=0,a=Q_1^infty and Q_lambda^infty=h+lambda(a−h). When an interior witness exists and0<L<1,an endpoint supporting direction d has h(d)>0,a(d)<h(d) and

L=h(d)/(h(d)−a(d)).

For the displayed endpoint LP, an equality-dual vector converted back from column scaling supplies this d. With the LP minimization convention, its normalization gives m*h(d)=L and m*(a(d)−h(d))=−1. Check these identities from the ORIGINAL loss,not from stored dual objective fields. Every power above the ratio has a verified negative recession direction;fixed-power original moment LPs check powers below the endpoint. Retain primal/dual residuals and all numerical states.

For a zero interior-witness optimum, the existing weak-separator characterization supplies a nonzero direction with h(d)=0 and a(d)<=0. An independent LP can find it by forcing each signed coordinate in turn to1 while bounding every direction coordinate in[−1,1];one such normalization represents every nonzero homogeneous direction. Negative a gives unbounded loss for every positive power;zero a is a boundary obstruction. Numerical weak margins and costs retain their tolerances and are not exact rational certificates.

The public audit will replay the original500allocation permutations at50labels for both prediction sources,1000cases total. These include every recorded unit-power exterior case;higher-label cases already have unit roots and follow from nesting. Predictions,original estimators and existing selected powers remain fixed. No root is fitted at the largest feasible endpoint and no largest-feasible-power estimator is proposed. This can clarify admissible weights for a specified target;its practical value and the full paper's contribution still need assessment.
