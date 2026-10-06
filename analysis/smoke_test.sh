#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root after installing requirements-analysis.txt.
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-/tmp/jocm_smoke}"
mkdir -p "$OUT_DIR"
"$PYTHON_BIN" analysis/run_simulation.py --reps 1 --out "$OUT_DIR/simulation.csv"
"$PYTHON_BIN" analysis/run_ml_extension.py --reps 1 --out "$OUT_DIR/tree.csv"
"$PYTHON_BIN" analysis/run_mixed_logit_extension.py --reps 1 --out "$OUT_DIR/mixed_logit.csv"
"$PYTHON_BIN" analysis/process_gate.py --reps 1 --out "$OUT_DIR/process_gate.csv"
"$PYTHON_BIN" analysis/equivalent_pair_test.py --reps 2 --n-per-pair 30 --bootstrap 50 --out "$OUT_DIR/equivalent_pair.csv"
"$PYTHON_BIN" analysis/observational_equivalence_test.py simulate --reps 1 --bootstrap 20 --out "$OUT_DIR/observational_equivalence.csv"
"$PYTHON_BIN" analysis/exact_paired_task_benchmark.py --reps 1 --bootstrap 20 --respondents 100 --out "$OUT_DIR/exact_paired_task.csv"
"$PYTHON_BIN" analysis/factorial_equivalence_contrast.py --reps 1 --bootstrap 20 --respondents 100 --out "$OUT_DIR/factorial_equivalence.csv"
"$PYTHON_BIN" analysis/randomization_paired_task_test.py --reps 1 --randomization-reps 20 --respondents 100 --out "$OUT_DIR/randomization_paired_task.csv"
"$PYTHON_BIN" - "$OUT_DIR" <<'PY'
import csv, pathlib, sys
root = pathlib.Path(sys.argv[1])
expected = {"simulation.csv": 32, "tree.csv": 16, "mixed_logit.csv": 8, "process_gate.csv": 5, "equivalent_pair.csv": 4, "observational_equivalence.csv": 4, "exact_paired_task.csv": 32, "factorial_equivalence.csv": 7, "randomization_paired_task.csv": 12}
for name, rows in expected.items():
    with (root / name).open(newline="") as f:
        count = sum(1 for _ in csv.DictReader(f))
    if count != rows:
        raise SystemExit(f"{name}: expected {rows} rows, found {count}")
    print(f"{name}: {count} rows")
PY
