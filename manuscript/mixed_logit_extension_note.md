# Mixed Logit extension note

The repository now includes `analysis/run_mixed_logit_extension.py`. This extension estimates a single random price coefficient with simulated likelihood and keeps the respondent-level 80/20 split used by the corrected benchmark. It is a targeted heterogeneity check, not the final Mixed Logit specification.

The three-replication extension estimates a mean random-price standard deviation of about 0.36 in the heterogeneity condition and about 0.31 in the combined condition. Mean decision regret is 0.041 in the heterogeneity condition and 0.235 in the combined condition. The pattern is informative: a random price coefficient addresses continuous taste variation, while the combined condition also requires nonlinear and interaction terms.

The final paper should replace this check with a full Mixed Logit model that documents the random-coefficient vector, simulated draws, convergence criteria, WTP recovery, and sensitivity to the number of draws. The current outputs are kept separate from the locked primary benchmark.
