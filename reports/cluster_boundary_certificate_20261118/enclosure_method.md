# Enclosing the existing matched block comparison

This calculation revisits the already retained unit-power population screening. It uses the same marginal compensated loss, three tasks per person, outcome dependence 3/4 and proxy error 1/4. It adds no simulated respondent or fitted estimator. The fixed models have identical aggregate-score covariance blocks. Their ordinary-logit boundary is defined to equal one by a scalar correlation root.

At unit power the labelled residual box is a singleton. Summing its 64 outcome/proxy states gives the exact quadratic coefficient c=225/256 in both models. The auxiliary distribution reduces by antipodal Gaussian symmetry to four prediction vectors, with probabilities 25/64 and three copies of 13/64. Therefore the person-normalized boundary is the supremum over t>=0 of

F(t)=-c t^2 + E[||H||^2-dist^2(H,t C D)],

where H is a three-dimensional standard Gaussian, C is any factor satisfying C^T C=Sigma, and D is the corresponding auxiliary box. C can be the upper Cholesky factor. Each auxiliary term is concave in t by joint convexity of the squared-distance minimization. F is consequently 2c-strongly concave. At any positive t0 its supremum is at most F(t0)+F'(t0)^2/(4c). Upper and lower secants at t0 enclose the derivative, using the interval values at t0-h,t0,t0+h.

For a fixed box, f(h)=||h||^2-dist^2(h,t C D) is convex with a 2-Lipschitz gradient. Partition each normal coordinate into fixed cells, including unbounded tail cells, and put M=E[H|cell]. Conditional Jensen and the smoothness bound give

E f(M) <= E f(H) <= E f(M)+E||H-M||^2.

The product-quantizer remainder is three times its one-coordinate conditional variance. This is a deterministic integration enclosure, not a Monte Carlo standard error or a difference between numerical resolutions. Arb encloses the normal cell probabilities, means, second-moment remainder and correlation roots. Cell boundaries and evaluation scales are exact dyadic numbers. Float node and matrix approximations require their own explicit error bounds before claiming a separated interval.

At any feasible box point u the quadratic is q(u)=2z^T u-u^T Sigma u. Its exact optimum lies between q(u) and

q(u)+2 sum_j max{(z-Sigma u)_j(l_j-u_j),(z-Sigma u)_j(r_j-u_j)}.

Thus a primal point with a small original-quadratic dual gap validates the projection independently of how the point was obtained. The implementation must include arithmetic and parameter-enclosure errors; unquantified agreement between solvers is insufficient.

The block theorem still uses existing conditional Gaussian absorption and conic kinematics. A separated numerical interval would establish a specific nonzero gap for these model-defined parameters; it would not establish global originality or practical importance. The earlier screening gap is small. Its magnitude and the stronger existing human-invariant scale construction must guide any decision to add this extension to the article.
