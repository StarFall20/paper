# Power curve round 33

The primary power grid uses 500 replications per cell, 499 hypergeometric within-fibre draws, n ∈ {200, 400, 800}, and η ∈ {0, .05, .10, .15, .20, .30, .45}. Each CSV row includes a Wilson 95% interval. At η = 0.10, support-score rejection is 0.212, 0.320, and 0.616 for n = 200, 400, and 800. At η = 0.20 it is 0.622, 0.930, and 1.000. The η = 0.45 cells are a strong-signal boundary condition.

The score and complete-fibre LR columns agree to Monte Carlo resolution because they test the same saturated conditional alternative through score and deviance forms. The grid is a power-planning result; it is separate from the pre-outcome rank certificate.

The release also contains `results/questionnaire_cluster_power.csv`, generated with eight tasks per respondent, respondent-level randomization, respondent-level random-effects dependence, 200 replications per cell, and 499 complete-randomization draws. Under the planning null (eta = 0), rejection rates are 0.030, 0.045, and 0.050 for 200, 400, and 800 respondents. At eta = 0.20 they are 0.100, 0.245, and 0.420; at eta = 0.30 they are 0.165, 0.520, and 0.845. These are planning values for the proposed instrument and are not human-data evidence.
