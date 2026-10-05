#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root after installing requirements-analysis.txt.
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-/tmp/jocm_smoke}"
mkdir -p "$OUT_DIR"
"$PYTHON_BIN" analysis/run_simulation.py --reps 1 --out "$OUT_DIR/simulation.csv"
"$PYTHON_BIN" analysis/run_ml_extension.py --reps 1 --out "$OUT_DIR/tree.csv"
"$PYTHON_BIN" analysis/run_mixed_logit_extension.py --reps 1 --out "$OUT_DIR/mixed_logit.csv"
python3 - "$OUT_DIR" <<'PY'
import csv, pathlib, sys
root = pathlib.Path(sys.argv[1])
expected = {"simulation.csv": 32, "tree.csv": 16, "mixed_logit.csv": 8}
for name, rows in expected.items():
    with (root / name).open(newline="") as f:
        count = sum(1 for _ in csv.DictReader(f))
    if count != rows:
        raise SystemExit(f"{name}: expected {rows} rows, found {count}")
    print(f"{name}: {count} rows")
PY
