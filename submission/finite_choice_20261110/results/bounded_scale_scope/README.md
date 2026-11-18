# Bounded scale robustness of the existing compensated logit boundary

The existing auxiliary-budget counterexample also occurs with a known scale ratio of four. This supplement keeps the manuscript's balanced Gaussian model and compensated objective. It adds a sharp characterization of its two auxiliary endpoint limits, plus verified finite-ratio calculations. The moment inequality is classical. This is a robustness result within the existing contribution, not a separate new theory or a certification of submission readiness.

## Finite auxiliary ratios

The plan fixes the primary candidate before the new calculations: scale D is 1/2 with probability 4/5 and 2 otherwise; proxy error is 2/5; the dimension-to-human-label ratio is 23/50. Thus E D²=1, E D=4/5 and proxy accuracy is 60%. The existing population variance rule is λ=τ/(1+τ). The model has zero utility signal, independent Gaussian remaining attributes and a frozen calibrated proxy. D is known and multiplies the whole utility row, including the intercept. It is not an estimated respondent scale.

| Auxiliary/human ratio τ | Existing power λ | Existence boundary κ_D |
| --- | --- | --- |
| 1/40 | 1/41 | 0.485566236693154 |
| 1/10 | 1/11 | 0.460774479062688 |
| 1/4 | 1/5 | 0.431858889006505 |
| 1 | 1/2 | 0.398297646021542 |
| 4 | 4/5 | 0.405176410542682 |
| 16 | 16/17 | 0.417547602243916 |
| 64 | 64/65 | 0.422514662046865 |

At κ=0.46, τ=1/40 is on the strict finite-estimate side and τ=1 is on the strict no-finite-estimate side. Ordinary human logit remains below its threshold 1/2 at both ratios. These are asymptotic model statements; the table is not a simulation frequency or a DCE failure-rate estimate. The curve decreases and then increases. The fixed-dimensional covariance factor decreases from 3.996097561 to 3.92 between these two ratios, a reduction of about 1.9%. Its entire possible reduction is only 4% for this weak proxy. Existence and this regular covariance reference address different sample limits.

The complete 196-condition grid is in [boundaries.csv](results/boundaries.csv); no conditions are dropped. A secondary, grid-observed case uses error 1/4, hence 75% proxy accuracy, with the same scale law. Its boundaries at τ=1/40 and τ=1 are 0.49118035 and 0.44299102. It also straddles κ=0.46, with a regular covariance reduction of about 12.0%. This case was not the primary candidate fixed in the plan.

## What the support-ratio statement does and does not establish

For known a≤D≤b, r=b/a, the established bounded-variable moment inequality gives (E D)²/E D²≥4r/(1+r)². Combining it with the manuscript's endpoint formula shows that a strict decline from the τ↓0 limit 1/2 to the τ→∞ limit is possible among these scale laws exactly when

$$r>\frac{1+\sqrt{1-\pi e(1-e)}}{1-\sqrt{1-\pi e(1-e)}}.$$

The critical ratios for errors 0.10, 0.25, 0.40 and 0.45 are approximately 12.0642, 4.5719, 2.9683 and 2.7854. [The proof](proof.md) supplies necessity and an attaining endpoint law. This is an endpoint characterization. It is **not** a lower bound on every finite-τ boundary. For example, error 1/4 and r=4 have a large-auxiliary limit 0.54324887 above 1/2, while the finite-τ=1 boundary is 0.44299102. Even constant-scale conditions can have small finite-τ dips for weak proxies. The endpoint criterion cannot be used as a finite-budget safety rule.

## Verification and reproduction

[evaluate.py](evaluate.py) isolates each unique maximizing root by 90 dyadic bisections using 160-bit Arb arithmetic. Strong concavity supplies a global upper bound, including all Gaussian tails. [The independent implementation](verify_independently.py) does not import the producer. It reconstructs the original squared distance to the labelled and auxiliary intervals, verifies roots and values with separately written 256-bit truncated moments, and integrates the original piecewise distances with SciPy. The two interval implementations use the same Arb library; SciPy supplies a different numerical engine, with diagnostics rather than certified integration errors.

All 196 roots, original integrals and global enclosures and all 28 endpoint cases pass. The largest difference from adaptive integration is 4.45×10⁻¹⁶. Printed Arb endpoint strings have their own rounding radii; the verifier uses the outer edges of these strings. Float bounds are outward rounded separately. This is independent implementation checking within this project, not external peer review.

Install [requirements.txt](requirements.txt), then in a fresh directory containing the sources and plan run:

```text
python evaluate.py
python verify_independently.py
python verify_package.py
```

`evaluate.py` intentionally refuses to overwrite an existing boundary file. To replay, copy the source package, remove only the copied results directory, and evaluate there. [replay_verification.json](replay_verification.json) records a separate-copy replay; [MANIFEST.sha256](MANIFEST.sha256) binds all delivered files. The inventory verifier does not repeat the numerical calculations.

No new model fits, Monte Carlo repetitions, respondents or AI answers are introduced. The current reviewed Word manuscript remains unchanged; [proposed appendix text](proposed_appendix.md) is supplied separately pending integration and full document checks. [The literature and contribution audit](literature_audit.md) identifies established components and remaining priority and value questions.
