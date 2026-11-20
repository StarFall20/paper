# Fixed menu frequency sensitivity on public vaccine choices

This audit uses the public design and outcome archive of Lu et al. (2026), pinned at commit d18717206867c7a12ff470879a0dd5d488b300c2. The two forecasts are ridge logit models trained on a separate set of released choices, one against human labels and the other against the released hard AI labels. Both sets of model coefficients are frozen before any frequency scenario is evaluated. A given predictor, labelled allocation, correction weight and entire auxiliary menu library are fixed within each comparison.

The primary audit checks 16,128 corrected objectives. Each scenario uses the same 1,199 distinct ordered menus and 4,800 auxiliary rows counted with multiplicity. For 50 labelled choice records, 94 allocations have a finite ordinary logit estimate. At correction weight 0.75, among those allocations the finite and unbounded states both occur under different frequency scenarios in one allocation with the human-trained predictor and 32 allocations with the AI-trained predictor. The two values come from separate predictor comparisons on the same allocations. The complete seven-scenario ledger includes cases outside the ordinary finite subset.

The follow-up verifies both endpoints and the maximum feasible frequency transfer on 19 selected paths. A separate constrained construction tests 38 positive fractional weight vectors that match the objective value, gradient and Hessian at zero. Thirty-two certify an unbounded objective, including 13 under total variation budget 0.20. The largest discrepancy in the matched quantities is below 3 × 10⁻¹³. Candidate construction uses a direction from the corresponding exterior fit. These counts describe tested witnesses and give no prevalence estimate for the effect.

## Reproduction

Set the archive variable to the extracted submission reproduction directory and the data variable to the public GAI input directory from the archive's download instructions. Set output to a writable result directory. Run from this report directory with the same Python environment as the archive. The default primary evaluator runs linear programs directly. The optional flag uses a strictly positive verified probability representation where available and otherwise falls back to the same linear programs.

    python run.py --archive "$archive" --data "$data" --out "$output" --fast
    python verify_counts.py "$output"
    python check_frontier.py --archive "$archive" --data "$data" --results "$output/results.json" --out "$output/frontiers.json"
    python check_local_diagnostics.py --archive "$archive" --data "$data" --results "$output/results.json" --out "$output/matched_derivatives.json"

The archive reader verifies the pinned source manifest and selected numeric file prior to decoding. It executes no source analysis code. The archive lacks verified respondent identifiers. The five-row ordering block is a source convention. These experiments concern the objective on released menus, without inference about population choice behaviour or vaccine uptake. [Source article](https://proceedings.mlr.press/v306/lu26am.html)
