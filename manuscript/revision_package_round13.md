# Revision package round thirteen

## Recommended title

**Testing Utility-Basis Sufficiency in Discrete Choice Models: A Nuisance-Balanced Utility-Fibre Design for Smart Medical Devices**

This title states the methodological problem, the design object, and the
application testbed. It removes the impression that the paper's main question
is a generic XGBoost versus MNL competition.

## Draft abstract

Discrete choice models can appear adequate when observed tasks occupy a narrow
part of the attribute space. This paper develops a design-based specification
audit for that setting. We construct reflected pairs of attribute profiles
that the candidate model assigns the same complete menu utility vector,
including the opt-out. The pair changes the attribute decomposition while
preserving a predeclared nuisance signature, including value-weighted
comparison distance and display load. The odd choice-probability response
becomes a test for an omitted utility direction; the even response is reported
as a processing and scale diagnostic. The result extends to the
moderate-utility form (F(\Delta V/D)), so the design is not tied to a logit
link. In a smart medical-device attribute space, exhaustive enumeration finds
677 candidate-preserving reflections, 648 pure-odd reflections, and 58
nuisance-balanced reflections. A five-task balanced design rejects a
complexity-only null at .050 compared with .185 for the loose design in a
600-respondent stress test, while omitted-direction power remains 1.000 for
both. With a respondent-level orientation coin, a 150-respondent weak-signal
benchmark gives .490 rejection for the ordinary grid and .870 for the
fibre-optimal grid; the corresponding combined omitted-plus-even-scale rates
are .335 and .785. The five-task block also yields a familywise-controlled
violation profile: under a localized omitted interaction, global power is .330,
profile power is .885, and the active task is correctly localized in .990 of
replications. A respondent-level random-taste negative control gives rejection
rates of .085--.095 at coefficient standard deviations of .1--.2, so this
boundary is reported as relation violation rather than mechanism
identification. The design also identifies the calibration uncertainty created
when candidate coefficients are estimated. The original XGBoost--MNL
comparison is retained as a predictive benchmark. The main contribution is a
reproducible experimental audit for utility-basis sufficiency, with explicit
limits for semantic asymmetry, odd processing, and saturated candidate bases.

## Main-text section map

### 1 Introduction

Open with the support problem in specification testing. State that predictive
fit and richer utility terms learn from existing support. Introduce the
candidate-preserving fibre as a new experimental source of identifying
variation. End with four contributions:

1. nuisance-balanced utility-fibre specification audit;
2. odd/even separation of utility and declared processing mechanisms;
3. fibre-optimal task criterion after nuisance balance;
4. smart-device implementation with calibration and semantic controls;
5. a max-T fibre-violation profile that localizes failures within the tested
   candidate support.

### 2 Related work

Use four compact subsections:

- utility specification and choice-set tests;
- comparison complexity, scale, and processing effects;
- DCE design efficiency, ordering, and attribute attendance;
- machine-learning benchmarks as predictive diagnostics.

Keep the distinction explicit: the paper tests a candidate relation in designed
support. It does not propose a new ML predictor or a latent permutation-
invariance theorem.

### 3 Candidate-preserving design

Define the complete candidate menu vector, the nontrivial fibre, the reflected
involution, the nuisance signature, and the odd/even estimands. State the
moderate-utility extension. State the interpretation rule: rejection is a
relation violation, while non-rejection is local to the tested fibre.

### 4 Fibre-optimal task construction

Describe profile enumeration, candidate-gap tolerance, nuisance-signature
equality, pure odd screening, semantic screening, and the information score
(p(1-p)s^2). Define the five-task grid and the independent calibration pilot.
Place the full pre-registration protocol in the supplement.

### 5 Simulation evidence

Use the following order:

1. size and power of the binary and multinomial parity tests;
2. incremental comparison with observational LR and residual diagnostics;
3. link robustness under logit and probit;
4. ordinary candidate grid versus fibre-optimal grid;
5. calibration-error sensitivity;
6. fibre-violation profile and localization;
7. negative controls for even scale, order, and semantic asymmetry.

Report Monte Carlo replication counts and treat all current values as planning
results until the final locked run is complete.

### 6 Smart-device testbed

Introduce the six attributes from the supplied manuscript. Show the exact
A+/A-/B+/B- task in a table. Verify the complete menu vector, the A--B gap, the
even geometry, and the odd intelligence-cloud interaction. Explain that the
testbed demonstrates constructability; it does not provide population
preference claims.

### 7 Predictive benchmark

Move the original MNL, Random Forest, XGBoost, SHAP, and lambda crossover here.
Frame them as a controlled predictive benchmark that motivates the need for a
specification audit. Keep the numerical crossover in a clearly bounded role.

### 8 Discussion and limitations

Discuss support, calibration, semantic equivalence, odd process effects,
attribute non-attendance, statistical power, and external validation. State the
conditions under which the fibre is empty or the primary contrast is silent.

### 9 Conclusion

Return to the design lesson: a candidate model needs a task support that can
challenge its basis. The proposed audit supplies that support and identifies
its limits.

## Tables and figures to retain

- Table 1: smart-device attributes and levels.
- Table 2: candidate-preserving task-pair verification.
- Table 3: size, power, negative controls, and calibration sensitivity.
- Figure 1: ordinary grid versus fibre-optimal weak-signal power.
- Figure 2: odd and even components under omitted utility and scale nuisance.
- Figure 3: workflow from independent calibration to preregistered paired DCE.
- Figure 4: global versus profile rejection under localized and diffuse omitted
  directions.
- Appendix table: complete five-task profiles and candidate menu vectors.

## Material to demote or remove

- Remove the generic claim that XGBoost becomes preferable after a transferable
  lambda threshold.
- Move SHAP rankings to the predictive benchmark or appendix.
- Remove language implying that simulated attribute importance describes a real
  medical-device market.
- Replace broad “first invariance test” language with the bounded observed-DCE
  specification claim.
- Keep all task replacement, calibration, and semantic decisions outcome-blind.
