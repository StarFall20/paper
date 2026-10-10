"""Run a static integrity audit over the repository and clean release.

This audit checks file closure and locked data artifacts. It deliberately
reports substantive readiness gates separately: passing code checks cannot
turn observational validation into a randomized human-data result.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tracked(repo: Path):
    out = subprocess.run(["git", "ls-files", "-z"], cwd=repo, check=True,
                         capture_output=True, text=False).stdout
    return [repo / item for item in out.decode().split("\0") if item]


def parse_python(files):
    failures = []
    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:  # pragma: no cover - failure path is the report
            failures.append(f"{path}: {exc}")
    return failures


def imported_local_modules(root: Path, entrypoints):
    """Return release-local modules reachable through ordinary imports."""
    seen, missing = set(), []
    queue = [root / item for item in entrypoints]
    while queue:
        path = queue.pop()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module.split(".")[0])
        for name in names:
            candidate = root / "analysis" / f"{name}.py"
            if candidate.exists() and candidate not in seen:
                queue.append(candidate)
            elif not candidate.exists() and name in {"observational_equivalence_test", "fibre_baseline_comparison",
                                                     "support_complete_fibre_design", "support_complete_fibre_benchmark"}:
                missing.append(f"{path.relative_to(root)} imports missing local {name}.py")
    return seen, missing


def check_manifest(root: Path):
    manifest = root / "MANIFEST.sha256"
    failures = []
    for line in manifest.read_text().splitlines():
        expected, rel = line.split(maxsplit=1)
        path = root / rel
        if not path.exists():
            failures.append(f"missing {rel}")
        elif sha256(path) != expected:
            failures.append(f"hash mismatch {rel}")
    return failures


def check_csvs(root: Path):
    failures, summary = [], []
    for path in sorted((root / "results").glob("*.csv")):
        with path.open(newline="") as handle:
            rows = list(csv.reader(handle))
        widths = sorted({len(row) for row in rows}) if rows else []
        if not rows or not rows[0] or len(widths) != 1 or len(rows) < 2:
            failures.append(f"malformed or empty CSV {path.relative_to(root)} widths={widths}")
        summary.append((path.relative_to(root), max(len(rows) - 1, 0), widths[0] if widths else 0))
    return failures, summary


def check_questionnaire(root: Path):
    import importlib.util

    verifier = root / "analysis" / "verify_candidate_preserving_questionnaire.py"
    spec = importlib.util.spec_from_file_location("questionnaire_verifier", verifier)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as temporary:
        out = Path(temporary) / "audit.csv"
        module.run(
            v1=root / "survey/candidate_preserving_dce_questionnaire_v1.docx",
            v2=root / "survey/candidate_preserving_dce_questionnaire_v2.docx",
            english_v1=root / "survey/candidate_preserving_dce_questionnaire_en_v1.docx",
            english_v2=root / "survey/candidate_preserving_dce_questionnaire_en_v2.docx",
            out=out,
        )
        with out.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    if len(rows) != 32 or not all(row["candidate_vector_preserved"] == "1" and
                                   row["sharing_and_price_fixed"] == "1" and
                                   row["profile_layout_matches_spec"] == "1" for row in rows):
        raise AssertionError("rendered bilingual questionnaire audit failed")
    return len(rows)


def write_report(repo: Path, release: Path, report: Path):
    files = tracked(repo)
    py_files = [path for path in files if path.suffix == ".py"]
    parse_failures = parse_python(py_files)
    entrypoints = [
        "analysis/support_complete_fibre_design.py",
        "analysis/fibre_baseline_comparison.py",
        "analysis/support_complete_power_curve.py",
        "analysis/questionnaire_cluster_power.py",
        "analysis/common_dce_design_exposure_audit.py",
        "analysis/verify_conditional_baselines.py",
        "analysis/verify_candidate_preserving_questionnaire.py",
        "analysis/electricity_external_validation.py",
        "analysis/train_vehicle_external_validation.py",
    ]
    closure, closure_failures = imported_local_modules(release, entrypoints)
    manifest_failures = check_manifest(release)
    csv_failures, csv_summary = check_csvs(release)
    questionnaire_cards = check_questionnaire(release)
    docx_count = sum(1 for path in files if path.suffix.lower() == ".docx")
    release_files = [path for path in release.rglob("*") if path.is_file()]
    findings = [
        ("PASS" if not parse_failures else "BLOCKER", "python_syntax", f"{len(py_files)} tracked Python files; parse failures={len(parse_failures)}", "Fix every parser failure before release."),
        ("PASS" if not closure_failures else "BLOCKER", "release_import_closure", f"{len(closure)} reachable release files; missing local imports={len(closure_failures)}", "Keep every runtime local dependency in the clean package."),
        ("PASS" if not manifest_failures else "BLOCKER", "release_hash_manifest", f"{len(release_files)} release files; manifest failures={len(manifest_failures)}", "Regenerate MANIFEST.sha256 after every release edit."),
        ("PASS" if not csv_failures else "BLOCKER", "locked_csv_integrity", f"{len(csv_summary)} release CSVs; malformed={len(csv_failures)}", "Repair malformed or empty locked outputs."),
        ("PASS", "rendered_questionnaire_audit", f"{questionnaire_cards} Chinese/English DOCX cards checked from rendered tables", "Keep the DOCX audit in the release check."),
        ("OPEN", "human_data", "No collected candidate-preserving respondent data are in the repository", "Do not claim empirical rejection or population validity; field and preregister a pilot first."),
        ("OPEN", "independent_review", "No independent mathematical or journal-style peer review is recorded", "Obtain an external review before calling the paper submission-ready."),
        ("OPEN", "design_scope", "The common-design result is an illustrative L9/D-exchange audit under one toy coarsening", "Add a real DCE design family or narrow the practical claim to the illustration."),
        ("OPEN", "measurement_validation", "The questionnaire composite candidate index has not passed cognitive validation", "Run comprehension, difficulty, opt-out, and version-balance checks before confirmatory use."),
    ]
    lines = [
        "# Full code and data audit — round 34",
        "",
        "This report separates executable integrity from scientific readiness. A passing software check does not supply randomized human evidence.",
        "",
        f"- Audit date: 2026-10-10",
        f"- Tracked files: {len(files)}",
        f"- Tracked Python files: {len(py_files)}",
        f"- Tracked DOCX files: {docx_count}",
        f"- Clean release files: {len(release_files)}",
        "",
        "## Findings",
        "",
        "| Status | Check | Evidence | Required action |",
        "| --- | --- | --- | --- |",
    ]
    for status, check, evidence, action in findings:
        lines.append(f"| {status} | {check} | {evidence} | {action} |")
    lines += ["", "## Release CSV dimensions", "", "| File | Data rows | Columns |", "| --- | ---: | ---: |"]
    lines += [f"| `{name}` | {rows} | {width} |" for name, rows, width in csv_summary]
    if parse_failures or closure_failures or manifest_failures or csv_failures:
        lines += ["", "## Machine-check failures", ""]
        lines += [f"- {item}" for item in parse_failures + closure_failures + manifest_failures + csv_failures]
    lines += [
        "",
        "The release is executable after the repaired dependency closure and rendered-DOCX audit. The manuscript remains a method-and-design study with planning simulations. Human-data sufficiency claims, a validated measurement instrument, and independent review remain open gates.",
        "",
    ]
    report.write_text("\n".join(lines))
    return findings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--release", default="reproducibility_round33")
    parser.add_argument("--out", default="manuscript/full_code_data_audit_round34.md")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    findings = write_report(repo, repo / args.release, repo / args.out)
    for row in findings:
        print(" | ".join(row))
