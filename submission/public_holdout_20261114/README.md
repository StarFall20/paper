# Public conjoint held-out-record diagnostic

Companion to the reviewed 13 November 2026 manuscript; this is a separately
checked follow-up, not an updated main manuscript or a new estimator.

Read [the Chinese findings](findings_zh.md), [the fixed local plan](plan.json),
[the literature update](literature_followup.md), and [the bounded Scholar log](scholar_search_log.json).

The fixed public input has 4605 human choice rows; only the first1200 have
aligned AI labels. Fit and tune on that prefix; evaluate the suffix3405.
Participant identifiers are unavailable. Row disjointness does not certify
independent people or a new population. Allocation MCSE is conditional on this
fixed file. Private questionnaires and original numeric pickles are excluded.

Python3.12.14 and the versions in requirements.txt were used. In an environment
with those dependencies, run from this package directory:

```sh
python analysis/fetch_public_conjoint_audit.py --out local_inputs --manifest source_manifest.json
python analysis/public_holdout_diagnostic.py --input-dir local_inputs --plan plan.json --out regenerated/results --canonical analysis
python analysis/verify_public_holdout.py --input-dir local_inputs --plan plan.json --results regenerated/results --out regenerated/verification --canonical analysis
```

To verify the distributed aggregate coefficients and summaries without fitting
again, use `--results results`. The verifier accepts the compressed JSON records
distributed here. No original response records are needed beyond the separately
downloaded pinned input. Every source numeric file is hash-checked before use;
no downloaded research code runs.

There are14000 method rows, representing500 nested allocations at two label
budgets and two prediction routines. Human-only baseline rows are duplicated
between those routines. They are not14000 independent samples. All8 unavailable
fits and all3000 Firth starts remain. A positive penalized Hessian and fixed
multistart selection do not certify a Firth global optimum.

The main workflow ran once. A second directory rechecked all original-loss
gradients, metrics and summaries from the compressed aggregate coefficients;
this does not claim another complete primary refit. Read verification records.

Source study: Lu, Wang, Zhang and Zhang (2026), Generative Augmented Inference,
https://arxiv.org/abs/2604.14575 ; repository commit in source_manifest.json.
Input licenses are recorded there: source code MIT, source data CC-BY-4.0.
Established Firth and PPI formulas are independently implemented; the manuscript
provides their primary citations. No external package source is copied here.
