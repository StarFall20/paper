# Independent discrete attributes in the same compensated choice model

Derived 16 November 2026. This is a scope extension of the current boundary, using an existing universality theorem. It introduces no estimator, no second framework and no claim of a new general universality law. The argument below is an internal mathematical derivation; numerical checks and independent peer review are distinct forms of evidence.

## Result and assumptions

Retain the binary model of Section 3.1, or the fixed-size common-coefficient menu model of Appendix F. Keep the signal and all fixed-dimensional conditioning variables under their stated laws. Replace only the remaining independent Gaussian attributes. Their **uncentered** entries are independent copies of a fixed symmetric variable, independent of the entire original low-dimensional tuple. The variable has mean zero, variance one and a finite absolute moment of some order p>4. This keeps the original independent-sample and conditional choice model intact. The prior transfer theorem allows broader entry classes; no such additional model generalization is needed here.

For the menu model, independence refers to entries before removing a within-menu common utility. Centering discrete profiles creates dependence among alternative contrasts. Applying an entrywise universality theorem to an orthogonally rotated or already centered discrete matrix would not be justified.

Fix a positive power lambda in (0,1], a finite positive auxiliary ratio, fixed menu size and fixed finite signal. Predictions remain frozen and independent of the remaining attributes, conditional on the original fixed-dimensional information. The binary result retains strictly interior predictions. The menu result retains its positive mean squared prediction error assumption and allows simplex-face predictions. Suppose the computed critical ratio kappa_c(lambda) is positive.

**Proposition (discrete-profile extension).** The same kappa_c(lambda) from equation (10), or (F.4), gives the two strict probability limits for a unique finite estimate under the entry assumptions above. Below the boundary a unique finite minimizer exists with probability tending to one. Above it the corrected target is exterior to its attainable moment set with probability tending to one, so the unrestricted objective is unbounded below. The equality case and the finite-sample transition width are unspecified.

All moments and distance profiles defining kappa_c remain those already calculated. The Gaussian variable in that formula is the auxiliary Gaussian used to define a cone's statistical dimension; it does not require the experimental attributes to be Gaussian after the transfer argument.

## 1. Prior theorem used, with reading scope

[Oymak and Tropp, Universality laws for randomized dimension reduction, with applications](https://tropp.caltech.edu/papers/OT17-Universality-Laws-preprint.pdf), author-hosted September 2017 preprint of the 2018 Information and Inference article, Model 2.4 and Theorem I, provide the transfer. The inspected PDF units are physical pages 13 and 17 (printed pages 9 and 13), with the scope discussion on physical page 19. The result controls avoidance and intersection of a compact spherical set by a random kernel under the entry assumptions stated above. It requires its statistical dimension to occupy a fixed positive fraction of the ambient dimension. Its lower side requires spherical convexity. Its probability bound is uniform over the admissible set and matrix. This is established theory. The complete 88-page proof has not been independently rechecked here.

The proposition still needs two estimator-specific bridges: an intersection must provide strictly interior choice probabilities, and the positively weighted discrete design must have full column rank. Gaussian generic position cannot supply either bridge for atomic entries.

## 2. The random map must precede centering

In the binary representation, orienting each remaining attribute entry by the conditioned sign preserves independence and its symmetric distribution. The map A is therefore in the cited matrix class. Its residual cone C=cone(D) intersect ker(V^T) is independent of A after conditioning.

For menus, stack all nJ raw alternative-profile entries in a matrix W_0 with q columns. Each residual block has sum zero. Hence W_0^T u equals the centered-profile map on every admissible residual u. The cone C lies in the n(J-1)-dimensional blockwise zero-sum subspace, intersected with the fixed constant/signal constraints. It is a closed convex cone in the ambient R^(nJ). Use the independent-entry map W_0^T in that ambient space. This preserves the original moment equations without treating centered coordinates as independent.

Statistical dimension is unchanged by adding orthogonal zero coordinates. The existing conditional Gaussian calculation gives delta(C)/m converging to kappa_c. Since n/m tends to 1+tau, its ratio to the raw menu ambient dimension tends to kappa_c/[J(1+tau)]>0. This verifies the nondegeneracy requirement of the cited theorem. The binary ambient denominator is (1+tau).

## 3. Eroded probabilities replace Gaussian general position

For 0<epsilon<1/J, define the contracted simplex Delta_epsilon={p: sum_j p_j=1, p_j>=epsilon}. On labelled menus replace D_L=(1-lambda)Delta-e_Y+lambda g by

    D_L,epsilon=(1-lambda)Delta_epsilon-e_Y+lambda g.

On auxiliary menus use

    D_U,epsilon=(lambda/tau)(Delta_epsilon-g).

At unit power the labelled residual is unchanged because its curvature weight is zero. For binary contrasts the same construction contracts each non-singleton residual interval from the two ends by the corresponding probability margin. Let C_epsilon=cone(product D_i,epsilon) intersect ker(V^T).

These are closed convex cones on the same positive-anchor event as in Appendix F1/A1. A nonzero kernel point in C_epsilon has a positive common scale. Dividing it by that scale yields residuals whose positively weighted choice probabilities are all at least epsilon. Such probabilities place the target in the relative interior of its attainable moment set. A finite minimizer exists on the identified coefficient space; full rank makes it unique.

The fixed-dimensional support lower bound in A3/F1 is positive on the unit multiplier sphere. Contracting the simplex changes its expected support by at most epsilon times a constant times E||V_i||. For sufficiently small fixed epsilon, at least half of that lower bound remains. The multiplier localization is therefore uniform over that interval of epsilon. The labelled residual's distance from zero is not reduced by contraction, so the positive mean residual anchor continues to bound the cone scale. Both optimizations consequently stay in common compact sets.

On these compact sets, the Hausdorff displacement of each scaled residual set is at most a constant times epsilon. The difference of squared distances has an integrable bound proportional to epsilon(1+||H||+||V_i||), with constants depending on the compact scale bound. Uniform laws for the two samples and uniform integrability proceed exactly as in the existing statistical-dimension proof. They give a limit kappa_epsilon for delta(C_epsilon)/m and

    0 <= kappa_c-kappa_epsilon <= C epsilon

for small epsilon. The constant is local to the fixed model and fixed power; it is not a reported numerical error certificate. The cones are nested as epsilon decreases. Thus kappa_epsilon increases to kappa_c.

If kappa<kappa_c, choose a fixed epsilon sufficiently small that kappa<kappa_epsilon. The spherical section of C_epsilon is compact and spherically convex, and its statistical dimension is proportional to the ambient dimension with high probability. Conditional application of the lower side of the cited theorem gives a nonzero kernel point with probability tending to one. The construction above gives strict interior feasibility. If kappa>kappa_c, the upper side applied to C excludes a nonzero kernel intersection with probability tending to one. On the positive-anchor event a feasible residual always supplies such a point. There is consequently no feasible target, and strict separation gives negative recession. Integrating the conditional probability bound proves both assertions. No assumption of a uniformly random discrete kernel has been introduced.

## 4. Discrete rank, including alternative constants

Use the classical Odlyzko pivot-coordinate argument, with its proof included to make the centered design application explicit. An accessible statement and graph-coordinate proof are [Maples, Singularity of random matrices over finite fields, Lemma 2.2](https://user.math.uzh.ch/maples/maples.srmff.pdf), physical page 3; that source states a finite-field version. The same elementary coordinate argument works over the reals. For a vector with independent coordinates whose largest atom is at most a<1, the probability of belonging to any fixed codimension-k linear subspace is at most a^k. Select k pivot coordinates from its linear equations; conditional on all other coordinates each pivot has at most one permitted value. Independence gives the product bound. For continuous coordinates the corresponding bound is zero. Neither the lemma nor its use to expose random columns is a new claim here.

Let B be the number of positively weighted menus: n for lambda<1 and N for lambda=1. The blockwise centered row space has dimension d=B(J-1). The fixed alternative constants and Gaussian signal have rank r_0=J almost surely. Reveal the remaining raw noise columns one at a time. After k independent columns, their span together with the fixed columns has dimension r_0+k. Its preimage under blockwise centering is a codimension d-r_0-k subspace in the raw coordinate space. The conditional probability that the next centered column is dependent is at most a^(d-r_0-k). Summing over q columns gives

    P(rank failure | fixed columns) <= a^(d-r_0-q+1)/(1-a)

whenever d-r_0-q>=0. The binary argument has no block centering and uses d=B and r_0=2. The strict lower side of the computed boundary leaves a positive linear gap: kappa_c<= (1+tau)(J-1) for lambda<1 and kappa_c<=tau(J-1) for lambda=1. The latter follows from the auxiliary-simplex cone-span bound already proved in F1. The rank failure bound consequently tends to zero exponentially.

The bounded p-moment condition supplies a uniform a<1. Symmetry bounds any nonzero atom by 1/2. If mass is concentrated at zero, Holder's inequality and unit variance give P(X!=0)>=nu^[-2p/(p-2)] when E|X|^p<=nu^p. One can take a=max(1/2,1-nu^[-2p/(p-2)]). This is a proof device, not a finite-sample rank prediction for a particular questionnaire. For symmetric two- and three-level independent entries the exact largest atoms are 1/2 and 1/3.

## 5. Scope and contribution decision

The extension removes the Gaussian restriction on independent remaining attribute entries, including symmetric independent two- and three-level attributes. It preserves one compensated objective and one boundary. The samplewise score identity in Proposition 2 is algebraic, and its balanced-model regular covariance uses only the second attribute moments. Combining that existing construction with the extension preserves both the equal local information and the opposite finite-existence limits for independent symmetric discrete noise, including the two- and three-level examples. This is a consequence of the same construction, not a second theory. It is an application of the prior transfer theorem with an interior-feasibility and rank argument; it should be presented as a model-scope proposition inside the existing paper.

It does not cover arbitrary effect coding with dependent dummy columns, feasible-combination restrictions, fixed orthogonal/D-efficient arrays, correlated respondent tasks, nonsymmetric entries, predictors trained on the audit rows or unproved cross-fitting schemes. It also does not transfer the Gaussian risk theorem, finite-sample root frequencies or interval coverage. A larger menu still uses the original Gaussian signal in the statement above. No all-discrete signal-law theorem is claimed here.

The existing machine proxy in the numerical illustration supplies additional outcome information conditional on the fitted attributes. An attribute-only black-box score under a correctly specified conditional choice model does not automatically supply that information. This remains a separate applicability limit already exhibited by the attributes-only control in E5. Extending the attribute law cannot erase it or establish a universal prediction advantage.

The new planned numerical experiment changes only independent noise profile levels on all original 1000 cohorts. It is a coupled design check, not additional human evidence. Matching a large-sample boundary does not imply matching finite-budget frequencies. The current manuscript remains unchanged until the full results and this argument are reconciled.
