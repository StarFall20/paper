# Menu frequencies and predicted choices in the recession function

Represent each menu by its multiset of contrast vectors, translated to a common origin. Translating a menu's utility attributes leaves its logit loss unchanged, and reordering alternatives together with their responses leaves the loss unchanged. Within stratum (k), write (h_k(d)=\max_j x_{kj}^{\mathsf T}d). For sample (s\in\{L,U\}), let \(\pi_{sk}\) be the relative menu frequency and \(\mu_{sk}\) the mean predicted feature moment. In a stratum observed in only one sample, use its observed prediction moment for both slots. Its missing frequency is zero.

Let \(b_Y\) be the human revealed feature moment. The ordinary labelled recession cost is \(H(d)=\sum_k\pi_{Lk}h_k(d)-b_Y^{\mathsf T}d\geq0\). Put \(D_{sk}(d)=h_k(d)-\mu_{sk}^{\mathsf T}d\). The recession of the corrected logit objective at weight \(\lambda\) is

\[
H(d)+\lambda\{C(d)+W(d)\},
\]
\[
C(d)=\sum_k(\pi_{Uk}-\pi_{Lk})\frac{D_{Uk}(d)+D_{Lk}(d)}2,
\qquad
W(d)=\sum_k\frac{\pi_{Uk}+\pi_{Lk}}2\{D_{Uk}(d)-D_{Lk}(d)\}.
\]

Expansion and cancellation recover \(\sum_k\pi_{Uk}D_{Uk}-\sum_k\pi_{Lk}D_{Lk}\) exactly. This is the symmetric composition-and-rate identity applied to directional logit recession. Each \(D_{sk}\) is translation invariant because the predicted probabilities sum to one and the directional maximum shifts by the same amount as the predicted moment. Changing the reference category does not affect the decomposition. An exterior target provides at least one direction with negative total recession. The split into \(C\) and \(W\) describes that particular direction. A different fit can have a different separating direction. [Kitagawa](https://doi.org/10.1080/01621459.1955.10501299)

## Two model interventions

Replacing the two conditional predicted moments by their equal average retains the original menu frequencies. It gives \(H(d)+\lambda C(d)\) exactly for every direction. For each common stratum, replacing the auxiliary menu frequency by the labelled frequency while retaining the separate predictions instead gives the original labelled log partition and the target

\[
b_{\rm matched}=b_Y+\lambda\sum_k \pi_{Lk}(\mu_{Uk}-\mu_{Lk}).
\]

In a labelled stratum missing from the auxiliary sample the contribution is zero by the stated rule. The loss adjustment is linear in the fitted coefficients, so the matched objective has the same Hessian as ordinary logit for every finite coefficient. Full contrast rank of the labelled design and interiority of \(b_{\rm matched}\) in its attainable moment set give a unique finite fit. A negative recession direction gives an unbounded objective, including on exactly matched menus. Under conditional equality of mean predicted moments in the two sample roles given the observed menu design and role assignment, the expected adjustment is zero at every fixed coefficient. This equality is an assumption about the prediction information and sample roles. The observed case comparisons use the finite fitted targets.

For a simple exact example take one binary menu with features 0 and 1 and equal numbers of labelled choices of both alternatives. Its attainable moments are the interval \([0,1]\) and its ordinary target is \(1/2\). Use 3/4 as the labelled prediction for alternative 1 and use 1/2, 1/4 or 1/8 as the auxiliary prediction in three otherwise identical cases. At unit power the corrected targets are 1/4, zero and −1/8, giving a finite fit, an unattained finite infimum and an unbounded objective, respectively. The prediction can be one frozen rule of an observed respondent feature taking different values across roles. Menu frequencies remain identical throughout.

## Prior overlap

The symmetric decomposition is established in rate analysis. Stratified PPI combines a common target distribution over strata and stratum-specific prediction corrections. In the overlapping menu strata, the exact matching intervention uses that architecture with empirical labelled menu shares and a fixed power. The present calculation uses the recession function to diagnose when that adjustment leaves a finite choice fit, and uses released predictions to check how frequently the two interventions change the same objective's existence state. Conditional moment stability is separate from the regular uncertainty results for stratified and nonuniform prediction-assisted inference. [Stratified PPI](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c9fcd02e6445c7dfbad6986abee53d0d-Abstract-Conference.html) · [Nonuniform sampling](https://arxiv.org/abs/2501.18577)
