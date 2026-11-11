# Whole-menu extension: research supplement

11 November 2026. This supplement investigates the same prediction-corrected logit objective as the binary manuscript. It contains a candidate conditional-choice derivation, complete simulated records, numerical checks and an aggregate audit of the existing DCE. It is separate from the current 32-page manuscript and its reproduction archive.

## Question and scope

The question is when the unpenalized compensated loss has a finite minimizer. The model has alternative-specific Gaussian profiles, common attribute coefficients and estimated alternative constants. The fixed number of alternatives is distinct from the growing number of attributes. Predictions are frozen interior probability vectors, independent of the remaining Gaussian attributes conditional on the stated low-dimensional variables. This is not a theorem for arbitrary categorical coefficient matrices, repeated respondent panels or all discrete questionnaire designs.

- [Derivation](menu_boundary_derivation.md): assumptions, residual simplexes, the candidate boundary and positive-power limit.
- [Chinese report](多项选择推广与真实DCE核验_20261111.md): findings and retained qualifications.
- [Phase figure](outputs/menu_phase.png): complete displayed finite-sample illustration.
- [Reading scope and provenance](reading_scope.json): sources and inspected units.

Classical support geometry, conic kinematics, simplex projection and Gaussian rotation retain their established attribution. The candidate contribution is the specific compensated whole-menu calculation and its consequences. Passing the checks below does not establish originality or submission readiness.

## Recorded results

`outputs/menu_phase_trials.csv` contains 6,000 classifications from 1,000 independent simulated model draws. Both powers and three dimensions share a draw. The 12 summary cells each contain 500 repetitions. These are simulated menus, not respondents. All states are retained; this run has no unresolved, boundary or rank-deficient state. The count is of support classifications, not an exact count of individual LP solves.

`outputs/menu_population_grid.csv` retains all 72 integration evaluations across resolutions and scramble seeds. The 18 event conditions consist of 12 fixed-power, four positive-power-limit and two ordinary conditions. Numerical stability is not a rigorous quadrature error bound. The finite-phase plan was selected after this grid and is a mechanism illustration. It was not a preregistered performance study. The population run's `phase_sampling_started:false` records that earlier stage; the separate completed phase status records the subsequent experiment.

`outputs/menu_independent_validation.json` records reconciliation of every phase row, 2,000 samplewise dimension-nesting groups and 12 independent full constrained cone QPs. The diagnostic also checks 360 original-vs-residual LP comparisons, exact binary reductions, derivatives and independent projections. Numerical checks do not replace the remaining proof review.

The actual DCE audit uses 300 human-recruitment records and excludes 60 AI pretests from human labels. Every person repeats the same eight tasks. Whole-person splits and menu-only frozen predictions imply that the two prediction-loss averages are identical for every coefficient vector. Thus their correction cancels at every power in the checked additive menu model. `outputs/balanced_dce_cancellation_checks.json` contains only aggregate checks, source-file hashes and counts. No private rows or answers are included. Versions were administered through different channels; no causal version inference is made.

The original boundary misclassification, a generic-QP coordinate-accuracy failure and a JSON counter serialization failure are retained alongside the corrected computations. `history/` holds the exact first two helper sources; the simulated boundary design and failure records are in `outputs/`. These historical sources are not current entry points.

## Reproduction

Run from the repository root in a scientific Python environment. The recorded environment used Python with NumPy 2.5.3, SciPy 1.18.1, pandas 2.2.3, scikit-learn 1.9.1, Matplotlib 3.11.2 and threadpoolctl. Repository-local imports are under `analysis/`; dependency files reused from the binary work remain there. `source_hashes.json` records the current entry points and their local import closure.

Choose a new writable output directory so the retained results stay unchanged:

```sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/menu-extension-matplotlib
python analysis/menu_corrected_logit.py --out /tmp/menu-extension-replay
python analysis/menu_existence_population.py --out /tmp/menu-extension-replay --bits 10
python analysis/menu_population_grid.py --out /tmp/menu-extension-replay --plan reports/menu_extension_20261111/plans/menu_population_grid_plan_20261111.json
python analysis/menu_existence_phase.py --out /tmp/menu-extension-replay --plan reports/menu_extension_20261111/plans/menu_phase_plan_20261111.json
python analysis/verify_menu_geometry.py --out /tmp/menu-extension-replay
```

The real-DCE identity audit requires the owner's unchanged private exports. It cannot be reproduced from a public archive of aggregate diagnostics alone:

```sh
python analysis/verify_dce_prediction_cancellation.py --input-dir /path/to/private/exports --instrument survey/public_dh36/instrument.json --classification data/dh36_source_classification.json --out /tmp/menu-extension-replay
python analysis/write_menu_extension_report.py --out /tmp/menu-extension-replay
```

The final report writer expects both simulation outputs and the DCE aggregate check. To recreate the public figure/report without access to exports, copy the retained aggregate `outputs/balanced_dce_cancellation_checks.json` into the replay directory. This reuses the recorded DCE evidence; it does not reanalyse private data.

`MANIFEST.sha256` hashes supplement files relative to this directory. `source_hashes.json` hashes code relative to the repository root. Original Monte Carlo outputs are retained; exact bytes can depend on the stated numerical software. No extra questionnaire, LLM responses, recruitment or payment was used for this supplement.

## Follow-up: hard category predictions

[Hard-prediction derivation](hard_prediction_derivation.md) extends the candidate scope to simplex faces,provides a separate exact perfect-label reduction,and isolates a binary versus larger-menu strict-gap condition. The [Chinese follow-up report](硬类别预测与创新复核_20261111.md) distinguishes the candidate from existing multiPPI++ and common-coefficient MNL prediction correction. `hard_outputs/` retains the complete integration and matched availability records,including the actual established fixed-dimensional variance reference for the hard proxy. The earlier posterior-prediction results in `outputs/` remain unchanged.

Run from the repository root,using a fresh output directory:

```sh
python analysis/menu_hard_prediction_boundary.py --out /tmp/hard-menu-replay --plan reports/menu_extension_20261111/plans/menu_hard_prediction_plan_20261111.json
python analysis/hard_menu_variance_reference.py --out /tmp/hard-menu-replay --plan reports/menu_extension_20261111/plans/hard_menu_variance_plan_20261111.json
python analysis/verify_hard_menu_prediction.py --out /tmp/hard-menu-replay
python analysis/hard_menu_matched_phase.py --out /tmp/hard-menu-replay --plan reports/menu_extension_20261111/plans/hard_menu_phase_plan_20261111.json
```

The independent follow-up verifier also checks five frozen historical inputs: the interrupted checkpoint,the exact finite boundary designs,their saved certificates,the interrupted crosscheck design and its initial solver diagnostics. Copy these supplied files before running it;this preserves the historical failure evidence rather than inducing a failure in a different solver version:

```sh
python - <<'PY'
from pathlib import Path
import shutil
source=Path('reports/menu_extension_20261111/hard_outputs')
target=Path('/tmp/hard-menu-replay')
for name in ['hard_menu_matched_phase_interrupted_trials.csv','hard_menu_retained_boundary_designs.json','hard_menu_boundary_certificates.json','hard_menu_interrupted_crosscheck_case.json','hard_menu_initial_phase_attempt.json']:
    shutil.copyfile(source/name,target/name)
PY
python analysis/verify_hard_menu_phase.py --out /tmp/hard-menu-replay --plan reports/menu_extension_20261111/plans/hard_menu_phase_plan_20261111.json
python analysis/write_hard_menu_report.py --out /tmp/hard-menu-replay
```

An interrupted current run can continue with `hard_menu_matched_phase.py --resume` and the same arguments. It restores whole completed draws and saved certificates,then regenerates the random-number prefix to reach the next draw. The recorded run resumed at800completed draws after a solver-unknown crosscheck and a diagnostic-key repair. Initial sources and diagnostics remain in `history/` and `hard_outputs/`. Replaying or resuming does not add statistical replications. The new generalization remains separate from the main manuscript pending proof,priority and practical-contribution assessment. Google Scholar/Scopus access was unsuccessful;`reading_scope.json` records the actual inspected primary units and the unresolved close-title lead.
