# One step comparator audit

This supplement audits the existing one-step comparator against the supplied PPI++ v2 and pinned author code. It retains every original condition: 24 conditions, 500 repetitions each, five methods and 60000 method records on 1500 existing simulated cohort families. No new independent cohort, participant or AI answer is added. [EVIDENCE.md](EVIDENCE.md) explains the results and limits; [source_scope.json](source_scope.json) identifies the precise source versions and reading units.

Use Python 3.12 and the [recorded versions](requirements.txt). From this directory, download the pinned public numeric inputs and verify their hashes:

```sh
python analysis/fetch_public_conjoint_audit.py --out local_inputs --manifest reference/source_manifest.json
```

The download is inbound and retrieves numeric public research files; it executes no downloaded source. The existing restricted numeric loader verifies their manifest before reading them. Downloaded source files, user questionnaires and the uploaded article PDF are excluded from this supplement.

Check the archive and independently reconstruct every recorded update, weight and loss:

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python verify_package.py
python run_reproduction.py --input-dir local_inputs --out regenerated/checks
```

To repeat the primary computation from the original random seeds, then verify it and compare all five scientific output files byte-for-byte, use a new output directory:

```sh
python run_reproduction.py --input-dir local_inputs --out regenerated/full --full
```

The human ridge pilot is common across methods. Four variants separate pooled/labelled information and clipped/unrestricted powers; human ridge is the fifth comparator. These are ridge-start implementations of stated formulas. The [source record](source_record.json) distinguishes them from literal upstream API reproduction and from finite global minimizers. Source-code control-flow observations are static; no upstream source was executed or modified.

Mean losses and Monte Carlo errors concern this retained simulation grid. The results establish neither a general harmfulness theorem for unrestricted weights nor a new estimator. The formal 2026 PPI++ article and Appendices A–E remain separately unresolved. This supplement is prepared for author/reviewer inspection and does not certify submission readiness or global priority.
