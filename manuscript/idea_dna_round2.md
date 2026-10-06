# Idea DNA round two

## Search question

The current paired-task test has a fixed common-shift transformation. The
second search asks whether a stronger object can be developed without turning
the paper into a generic model-comparison or optimal-design paper.

## Candidate ideas

| candidate | defining object | closest boundary | decision |
|---|---|---|---|
| Candidate-preserving metamorphic audit | A set of task transformations derived from a candidate utility basis; each transformation has a model-implied relation between paired choice probabilities | Segura et al.'s metamorphic testing; Breitmoser's choice invariances | retain and develop |
| Maximin null-preserving task design | Choose feasible transformations that maximize the weakest probability separation across a preregistered probe family while preserving the candidate exactly | Huang's discriminatory choice designs and general T-optimal design | retain as a design layer, not a standalone novelty claim |
| Mean-fibre versus coefficient-free audit | Compare transformations that preserve fitted mean utility differences with transformations that preserve differences for every coefficient vector | Literature on taste heterogeneity and specification confounding | exploratory only; estimation error weakens exact inference |
| Policy-equivalence envelope | Report the range of policy outputs across models that pass a predictive tolerance | Robust policy analysis and observational equivalence | backup contribution |
| Automated model-race selector | Search a large utility library with a learner and choose the best model | JOCM assisted specification, Mixed-Logit search, and specification agents | reject as primary contribution |

## Selected object

The selected object is a **candidate-preserving metamorphic specification
audit**. It has three linked components:

1. **Relation construction.** For a candidate utility basis (b(x)), construct
   task pairs whose alternative utility-difference vector is identical under
   the candidate. For an additive basis, a common attribute shift is one exact
   construction. The null relation is equality of the paired choice
   probabilities.
2. **Transformation design.** Before held-out responses are observed, choose
   transformations from a feasible grid by maximizing a maximin score over a
   fixed probe family. The score uses the weakest predicted probability
   separation across nonlinear, threshold, interaction, and out-of-library
   probes, with bounds on attribute movement and respondent burden.
3. **Inference.** Freeze the candidate on development respondents. Use a
   respondent-cluster sign-flip reference on held-out paired contrasts. The
   test remains valid against departures outside the probe family because the
   probe family chooses the design; it does not define the null.

The scientific output is a failure map for the candidate utility basis: which
predeclared transformations violate the candidate-preserving relation and how
much power the design has under controlled departures. A non-rejection is
evidence of compatibility with the tested relation, not proof of correctness.

## Why this is distinct

Metamorphic testing addresses the test-oracle problem by checking how output
changes after an input transformation. The choice-model version derives the
transformation from utility differences and tests a behavioural relation with
cluster randomization. It does not require a fully specified alternative
model.

Huang's discriminatory designs maximize expected separation among a specified
set of competing choice models. The proposed design layer solves a different
problem: it keeps the candidate null exactly true while selecting tasks that
are sensitive to an open set of departures. The distinction must be stated
explicitly; the paper cannot claim to introduce model-discrimination design.

Breitmoser's axiomatic invariances and Fok and Paap's alternative-pair
misspecification tests set the choice-modelling boundary. The contribution is
the operational combination of candidate-preserving cross-task relations,
maximin transformation selection, and respondent-cluster randomization. It is
an implementation and design contribution, not a new invariance theorem or a
general MNL specification test.

## Falsification plan

The extension is worth retaining only if the maximin design improves the
minimum power across probe conditions without increasing null rejection. The
benchmark must include nonlinear, threshold, interaction, cubic or spline
out-of-library departures, random taste coefficients, and an order-effect
placebo. A fixed-shift design is the comparator. If optimization only repeats
the largest feasible shift, the maximin layer adds no scientific value and
should be removed from the main claim.

The mean-fibre versus coefficient-free comparison remains exploratory. A
transformation orthogonal to an estimated coefficient vector can detect
random taste variation, but parameter-estimation error breaks the exact
exchangeability argument. It can be reported as a sensitivity analysis only
after a separate calibration study.

## Required evidence before manuscript promotion

- a deterministic transformation generator with bounds and a recorded seed;
- a design score defined before outcomes are generated;
- fixed-versus-maximin size and power comparisons;
- out-of-library departures and a random-coefficient boundary;
- a task-order placebo and an opt-out design rule;
- an explicit literature comparison with Huang, Breitmoser, Fok and Paap, and
  metamorphic-testing work.

