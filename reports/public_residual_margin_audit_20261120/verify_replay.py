"""Replay both fold conventions and bind the audited claims to the supplement."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

from verify_report import verify

HERE = Path(__file__).resolve().parent


def replay(data, out, supplement):
    def run(scheme):
        stage = out / scheme
        runner = HERE / "evaluate_paths.py"
        subprocess.run([
            sys.executable, str(runner), "--data", str(data), "--scheme", scheme,
            "--out", str(stage),
        ], check=True)
        checked = {}
        for name in ("frontier_checks.csv.gz", "policy_rows.csv.gz", "summary.csv"):
            result = (stage / name).read_bytes()
            assert result == (HERE / "results" / scheme / name).read_bytes(), (scheme, name)
            checked[name] = sha256(result).hexdigest()
        observed = json.loads((stage / "verification.json").read_text())
        stored = json.loads((HERE / "results" / scheme / "verification.json").read_text())
        assert observed.pop("runner_sha256") == sha256(runner.read_bytes()).hexdigest()
        stored.pop("runner_sha256")
        assert observed == stored
        return dict(verified_summary=verify(stage, scheme), checks=observed, checked_files=checked)

    assert not out.exists()
    with ThreadPoolExecutor(max_workers=2) as workers:
        reports = dict(zip(("row", "block"), workers.map(run, ("row", "block"))))
    claim = supplement.read_text().split("### I4 ", 1)[1].split("## Data availability", 1)[0]
    with (out / "block/summary.csv").open() as stream:
        selected = {r["method"]: r for r in csv.DictReader(stream)
                    if r["label_rows"] == "50" and r["prediction"] == "AI"}
    for method in ("full", "protected_half_margin", "fixed_half", "fixed_quarter", "all_soft"):
        assert f'{float(selected[method]["mean_loss"]):.5f}' in claim
    assert f'{float(selected["all_soft"]["mean_loss"]) - float(selected["all_soft"]["mean_vs_ridge"]):.5f}' in claim
    assert int(selected["full"]["exterior"]) == 4
    for scheme in reports.values():
        assert scheme["verified_summary"]["protected_fits"] == 2000
    receipt = dict(status="passed", reports=reports, checked_supplement_sha256=sha256(supplement.read_bytes()).hexdigest())
    (out / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--supplement", required=True, type=Path)
    args = parser.parse_args()
    replay(args.data.resolve(), args.out.resolve(), args.supplement.resolve())
