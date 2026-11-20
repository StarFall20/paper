"""Replay the archive calculation and bind it to the manuscript claims."""

import argparse
import hashlib
import json
from pathlib import Path
from subprocess import run
import sys
from tempfile import TemporaryDirectory


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(args):
    root = args.archive.resolve()
    with TemporaryDirectory(prefix="jeffreys-replay-") as tmp:
        out = Path(tmp) / "results.json"
        run([sys.executable, str(Path(__file__).with_name("audit.py")), "--archive", str(root),
             "--data", str(args.data.resolve()), "--out", str(out)], check=True)
        assert out.read_bytes() == args.reference.read_bytes()
        record = json.loads(out.read_text())
    main = root / "focused_article.md"
    supplement = root / "focused_supplement.md"
    assert "19 selected vaccine designs" in main.read_text()
    assert "same 50 labelled records and 1,199 auxiliary menus" in main.read_text()
    assert all(anchor in supplement.read_text() for anchor in ("Proposition K1.", "Matching arbitrary finite orders",
                                                               "$-0.000344$", "$-0.002117$"))
    assert record["exact_toy"]["penalized_directional_slope"] == -1 / 64
    proof = dict(status="passed", public_rays=record["public_rays"],
                 certified_unbounded=record["public_rays_still_certify_unbounded"],
                 exact_common_support_orders=[r["matched_order"] for r in record["exact_common_support_constructions"]],
                 exact_toy_and_multinomial_replayed=True, source_result_sha256=digest(args.reference),
                 article_sha256=digest(main), supplement_sha256=digest(supplement))
    args.out.write_text(json.dumps(proof, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["archive", "data", "reference", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    replay(parser.parse_args())
