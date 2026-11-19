# Finite choice estimation with released LLM predictions

The analysis uses Alves’s deposited Qwen3 predictions for four RADIANT choice experiments. It examines the finite-estimate guarantee and held-out predictive loss for a weight set to half the labelled-only bound. The original prediction-powered correction and prediction archive are due to Alves, and the form of the corrected objective follows PPI++. [Alves protocol and code](https://zenodo.org/records/23140920) · [Prediction deposit](https://zenodo.org/records/23143623) · [RADIANT data](https://zenodo.org/records/15420092) · [PPI++](https://doi.org/10.1214/26-AOAS2215)

The source has 24 archived cells across four contexts, two original respondent splits and three predictor coverage levels. We rotate seven of the 35 protected young adults into evaluation and use the other 28 for auxiliary menus. The primary analysis retains the original 15 young adults with correction labels. The five-person stress test divides those 15 into three disjoint groups. Rotations and coverage variants share respondents. The retrospective design supports paired descriptive comparisons on this archive.

We compare ordinary and ridge multinomial logit on the correction labels, three corrected policies, the frozen LLM probabilities and two direct fits that also receive the human choices used in the LLM prompts for the fitted samples. Each prompt reads two observed choices from each of four nearest donor respondents. The matched baseline takes the union of distinct prompt menus across the labelled and auxiliary predictions. Matched counts cover study-specific human examples used in fitting. LLM pretraining, inference costs and outcomes supplied to raw LLM evaluation prompts are outside that comparison.

## Results

| Correction respondents | Cases | Native safe fits | Native unit fits | Unit exterior, safe native | Median prompt menus | Safe minus same-label ridge log loss | Safe minus prompt-matched MNL log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| 15 | 120 | 120 | 115 | 5 | 141 | +0.0124 | +0.0671 |
| 5 | 360 | 240 | 268 | 54 | 124 | +0.2062 | +0.3933 |

Positive differences indicate higher held-out loss for the safe policy. The comparison includes its same-label ridge fallback at zero certified bound. Against ordinary MNL on the correction labels, log loss falls in 68 of 120 common fits at 15 respondents and 199 of 240 at five respondents. Against the prompt-matched MNL, it falls in 25 of 120 and 40 of 360. With five labels, the safe policy uses ridge in 120 cases, including 82 with a finite unit fit. A zero uniform bound permits the realized auxiliary design to support a finite corrected target. The retuned published power has 120 and 252 native fits. Paired differences and Brier scores for all comparators are in `results/summary.json`.

The prompt audit reconstructed 21,600 requests and matched all 21,267 distinct saved cache keys. Reaggregating the three cyclic alternative orders recovered all 7,200 frozen probabilities to an absolute error of at most $1.12\times10^{-16}$. A separate audit recalculated the scores and held-out losses recorded by the fit program. No LLM calls or predictor training took place during the audit.

## Run the audit

Place the two original ZIP files in a separate input directory. The code compares their SHA-256 values with the pinned input hashes before reading their contents.

| File | SHA-256 | Source |
|---|---|---|
| `Estudo2_protocolo_e_codigo.zip` | `3d2e5bcd1d3715d4bc8606dccbce81657c2b70a929184f0e6c3992829328e4a5` | [Alves protocol and code](https://zenodo.org/records/23140920) |
| `saida_confirmatorio_qwen3-30b-a3b-instruct-2507-q4_k_m.zip` | `7a8a093f1bee280732133aa7e8ad1baf57c07063464f39e55f4d08e7ae73274f` | [Alves predictions and cache](https://zenodo.org/records/23143623) |

Run from the extracted supplement using Python with NumPy, pandas, SciPy, openpyxl and threadpoolctl. The pinned Python files in `dependencies/` are included in the supplement.

```sh
python run.py --inputs /path/to/inputs --dependencies dependencies --out replay
python verify.py --inputs /path/to/inputs --dependencies dependencies --out replay
python verify_prompt_cache.py --inputs /path/to/inputs --dependencies dependencies --out replay
```

`results/summary.json`, `results/prompt_cache_summary.json` and the isolated replay receipt record the results. `run.py` generates the 3,840 method records. The archive contains aggregate scores, coefficients and split indices. Study workbooks, prompts, individual predictions and respondent identifiers are held in the separately obtained source archives.
