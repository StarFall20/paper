# Complete moments and the compensated loss

12 November 2026. This note compares existing estimation objectives. The explicit proportional-dimension boundary remains the manuscript's candidate contribution. The following observations do not define a new estimator.

## Inspected prior result

Chen, Bai and Zhu, [CC-PPI v2, Sections 2.1–2.3 and Supplement S3.10, Proposition S3.3](https://arxiv.org/html/2610.21837v2), use the stacked moments consisting of the human score m1 and the auxiliary-minus-labelled prediction score m2. Their complete GMM criterion retains both moments. The global certificate is sufficient and uses local curvature and an outside-ball lower bound. Its displayed conditions include a_r>0 and L_r=a_r r-||m1(c)||>0. The fixed-dimensional asymptotic result in Supplement S2.11 retains finite-pilot and regularity assumptions. Their calibration procedure and certificate remain prior work.

## What the certificate implies about the human fit

Let A be the human logistic Hessian. The certificate bounds A(c+t v) below by a_r I for every unit v and 0<=t<=r. On the sphere of radius r,

v' m1(c+r v) = v' m1(c)+integral_0^r v' A(c+t v) v dt >= -||m1(c)||+a_r r = L_r > 0.

Minimize the convex human loss over the closed ball. A minimizer on its sphere has a strictly positive outward derivative, so moving inward lowers the loss. The minimizer lies inside and has zero human score. Positive curvature gives full labelled rank and hence uniqueness. Thus these sufficient conditions also imply a finite human MLE. This is an applicability implication of their displayed conditions, not a claim about all possible complete-GMM solutions. When the human likelihood is separated, this particular certificate cannot hold, even if a complete-moment criterion happens to attain its minimum.

## Exact retained design

Use the endpoint of manuscript Appendix B5: human x=(2,4), y=(1,1), auxiliary x=(1,1), human predictions=(25/44,25/44) and auxiliary predictions=(1/2,1/2). All predicted probabilities are strictly interior. The compensated loss at unit power has recession cost -35/44 in the positive coefficient direction. The human score is strictly negative at every finite coefficient.

Freeze the reference at zero. Form the centered covariance of the stacked score rows, using sample-size divisors and covariance ridge m^-2=1/4. This gives Sigma11=1/2, Sigma12=-3/88, Sigma22=493/1936. Its inverse has entries W11=1972/977, W12=264/977 and W22=3872/977. This is the uncalibrated complete criterion with this explicitly chosen reference. It evaluates a specified frozen weight; CC-PPI additionally specifies its supervised pilot, protected covariance update and final fitting procedure.

With t=exp(beta), the moments are

m1 = t^2/(1+t^2)+2t^4/(1+t^4)-3,

m2 = t/(1+t)-t^2/(1+t^2)-2t^4/(1+t^4)+53/44.

Let K=(1972 m1^2+528 m1 m2+3872 m2^2)/(2*977). At t=3/2, K=43618624129/38838705425. The positive and negative infinite-coefficient limits are 1225/977 and 10729/977. The finite value is strictly below both limits. Continuity gives a compact nonempty sublevel set and a finite global minimizer. Uniqueness is not asserted. The same inputs thus give a separated human likelihood, an unbounded unit-power compensated loss and a finite complete-GMM minimum.

No common asymptotic covariance between these two different objectives is claimed for this selected design. The manuscript's oracle score-equivalence result is a separate comparison within the compensated-loss family.

## Numerical verification and interpretation

The fraction calculation uses no optimizer. A separate direct-array implementation reconstructs the covariance, moments, witness and both tails. Seven BFGS starts all return success with small gradient. Four returned values exceed the exact finite witness; those four cannot be global minima. The numerical routine therefore does not supply the global-attainment proof. The certificate arithmetic is also checked on 4941 center/radius pairs; the all-center implication above is analytic, not inferred from that grid.

The comparison narrows the practical lesson: a finite fit is a property of the chosen empirical objective. The existence boundary of the compensated loss does not establish nonexistence, poor performance or inferential failure of every other prediction-assisted procedure. Existing complete moments, pooling and penalization deserve attribution and explicit scope. The example does not measure their comparative predictive risk.
