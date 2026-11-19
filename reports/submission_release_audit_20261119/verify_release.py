"""Check the released mathematics and, optionally, the original public menus."""

import argparse
import gzip
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_input(parser, source, plan, key):
    if source is None:
        return None
    source = source.resolve()
    expected = json.loads(plan.read_text())[key]
    if not source.is_file() or sha256(source) != expected:
        parser.error(f"{source} does not match the public input identified in {plan.relative_to(ROOT)}")
    return {"file": source.name, "sha256": expected}


def verify(args, out):
    env = os.environ.copy()
    env.pop("PYTHONOPTIMIZE", None)
    env.update(
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(out / "matplotlib"),
        PYTHONPYCACHEPREFIX=str(out / "cache"),
    )
    checks = {}

    def check(name, script, *flags, cwd=ROOT):
        print(f"Verifying {name}", flush=True)
        log = out / f"{name}.log"
        with log.open("wb") as stream:
            result = subprocess.run(
                [sys.executable, str(ROOT / script), *map(str, flags)],
                cwd=cwd,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            print(log.read_text(errors="replace"), file=sys.stderr)
            raise SystemExit(f"{name} failed. See {log}")
        checks[name] = {"status": "passed", "log": log.name}

    check("inventory", "verify_inventory.py")
    check("headline", "verify_headline_results.py", "--research", ROOT,
          "--manuscript", ROOT / "focused_article.md", "--out", out / "headline.json")
    check("mathematics", "analysis/quick_check.py", "--out", out / "mathematics", "--full-math")
    check("exact_designs", "results/fixed_power_identical_scores/verify.py",
          "--designs", RESULTS / "fixed_power_identical_scores/designs.json",
          "--out", out / "exact_designs.json")
    check("common_newton_point", "results/multiple_diagnostic_points/verify.py",
          "--out", out / "common_newton_point.json")
    check("hessian_power_bounds", "results/hessian_power_bounds/verify.py",
          "--out", out / "hessian_power_bounds.json")
    check("unit_endpoint", "results/unit_correction_endpoint/verify.py",
          "--out", out / "unit_endpoint.json")
    check("unit_archive", "results/unit_correction_endpoint/audit_archives.py",
          "--repo", ROOT, "--out", out / "unit_archive.json")

    if args.food_source:
        panel = RESULTS / "public_panel_dce"
        check("food_panel", "results/public_panel_dce/analysis/verify_panel_audit.py",
              "--input", args.food_source.resolve(), "--results", panel / "results",
              "--out", out / "food_panel", cwd=panel)
        check("food_path", "results/dce_frontier_economic_path/analysis/verify_path.py",
              "--input", args.food_source.resolve(), "--frontier", RESULTS / "power_frontier",
              "--baseline", panel, "--results", RESULTS / "dce_frontier_economic_path/results",
              "--out", out / "food_path")

    if args.swissmetro_source:
        precedent = RESULTS / "swissmetro_corrected_path/results"
        check("swissmetro_path", "results/swissmetro_corrected_path/verify.py",
              "--input", args.swissmetro_source.resolve(), "--results", precedent,
              "--out", out / "swissmetro_path.json")
        archive = RESULTS / "swissmetro_label_budget/results"
        expanded = out / "swissmetro_records"
        expanded.mkdir()
        for name in ("records.csv", "coefficients.jsonl"):
            with gzip.open(archive / f"{name}.gz", "rb") as source, (expanded / name).open("wb") as target:
                shutil.copyfileobj(source, target)
        shutil.copyfile(archive / "summary.json", expanded / "summary.json")
        check("swissmetro_budget", "results/swissmetro_label_budget/verify.py",
              "--input", args.swissmetro_source.resolve(), "--results", expanded,
              "--precedent-summary", precedent / "summary.json",
              "--precedent-allocations", precedent / "allocations.csv",
              "--out", out / "swissmetro_budget.json")
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--food-source", type=Path, help="Pinned public COSCBB.xlsx")
    parser.add_argument("--swissmetro-source", type=Path, help="Pinned public swissmetro.dat")
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error("Run Python with assertions enabled")
    inputs = {
        "food": verify_input(parser, args.food_source, RESULTS / "public_panel_dce/plan.json",
                             "source_sha256"),
        "swissmetro": verify_input(parser, args.swissmetro_source,
                                   RESULTS / "swissmetro_label_budget/plan.json", "source_sha256"),
    }
    out = args.out.resolve()
    if out.exists():
        parser.error(f"Output directory already exists: {out}")
    out.mkdir(parents=True)
    checks = verify(args, out)
    packages = [line.split("==", 1)[0] for line in (ROOT / "requirements.txt").read_text().splitlines()]
    receipt = {
        "status": "passed",
        "archive_manifest_sha256": sha256(ROOT / "MANIFEST.sha256"),
        "python": sys.version.split()[0],
        "package_versions": {name: version(name) for name in packages},
        "checks": checks,
        "public_inputs": inputs,
        "optional_sources_not_supplied": [name for name, source in inputs.items() if source is None],
    }
    output = out / "release_receipt.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"{len(checks)} checks passed. Receipt: {output}", flush=True)


if __name__ == "__main__":
    main()
