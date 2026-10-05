# Mixed Logit extension note

The repository now includes `analysis/run_mixed_logit_extension.py`. This extension estimates a single random price coefficient with panel simulated likelihood and keeps the respondent-level 80/20 split used by the corrected benchmark. It uses paired normal draws during estimation and a fixed 80-draw prediction design. It is a targeted heterogeneity check, not the final Mixed Logit specification.

The three-replication extension estimates a mean random-price standard deviation of about 0.36 in the heterogeneity condition and about 0.32 in the combined condition. Mean decision regret is 0.041 in the heterogeneity condition and 0.236 in the combined condition. The pattern is informative: a random price coefficient addresses continuous taste variation, while the combined condition also requires nonlinear and interaction terms.

The final paper should replace this check with a full Mixed Logit model that documents the random-coefficient vector, simulated draws, convergence criteria, WTP recovery, and draw-stability diagnostics. The current outputs are kept separate from the locked primary benchmark.

`analysis/check_mixed_logit_draw_stability.py` compares 20, 40, and 80 paired draws in the heterogeneity and combined conditions. This diagnostic is required before treating simulated likelihood results as stable.
