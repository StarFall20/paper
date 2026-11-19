# Exact menu matching and finite choice estimation

The released RADIANT/Qwen forecasts let us separate two features of a corrected Logit fit. The auxiliary sample can show different menus, and its forecasts can differ on menus shown in both samples. The analysis keeps all 480 original partitions and compares unit correction, exact adjustment to the labelled menu frequencies and a common predicted moment per shared menu. The two interventions use no hidden human outcomes. [Model and derivation](proof.md) · [Original prediction deposit](https://zenodo.org/records/23143623)

| Labelled respondents | Allocations | Finite unit fits | Finite fits with matched frequencies | Finite fits with common within-menu moments |
|---|---:|---:|---:|---:|
| 15 | 120 | 115 | 114 | 120 |
| 5 | 360 | 268 | 215 | 295 |

At 15 labelled respondents, all five original exterior targets remain exterior after exact frequency matching, and one previously finite fit becomes exterior. When we give both samples the same predicted moment on each shared menu, all five cases become finite. At five respondents, the same prediction intervention recovers 43 original exterior cases and makes 16 originally finite targets exterior. The matched-frequency intervention recovers eight original failures and adds 61. Median labelled-menu coverage is 100% at both budgets. The minima are two thirds and 60%, respectively.

At an original separating direction, the within-menu prediction term is adverse in all five primary failures and in 90 of 92 stress-test failures. The composition term is also adverse in all five primary failures and in 84 of 92 stress-test failures. The terms are evaluated at one observed separating direction per case. The joint estimators can change which direction binds, as shown by the transition counts. [Verified results](verification.json)

Both modified fits use the declared same-label ridge fallback when unpenalized estimation is unavailable. At 15 respondents, matching the menu frequencies changes mean held-out log loss by −0.0213 relative to unit correction and +0.0640 relative to direct ridge on those labels. Giving both samples the same predicted moments changes those losses by −0.0026 and +0.0827, respectively. The original, matched-frequency and common-prediction fits all have higher mean loss than direct MNL given the same human prompt choices and correction labels. Results describe repeated splits of one released study. They do not estimate uncertainty across independent new studies.

## Reproduce

Obtain the two ZIP files and check their input hashes as described in the included [upstream instructions](upstream/README.md). Use Python with NumPy, SciPy, pandas, openpyxl and threadpoolctl. From the extracted supplement, run

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python audit.py --upstream upstream --dependencies dependencies --inputs /path/to/inputs --reference upstream/results/fits.json --out replay.json
python verify.py --upstream upstream --dependencies dependencies --inputs /path/to/inputs --reference upstream/results/fits.json --results replay.json --out replay_verification.json
```

The second program reconstructs the targets and scores from uncompressed menus, checks all existence states and recalculates the held-out losses. The bundle contains fitted coefficients and allocation summaries. It obtains published raw data and forecasts directly from their source archives at reproduction time.
