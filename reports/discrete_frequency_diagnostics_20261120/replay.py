"""Recreate and verify the filed assertions against the archived inputs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from verify import verify

HERE = Path(__file__).resolve().parent


def replay(out):
    expected = json.loads((HERE / "results.json").read_text())
    observed = verify()
    # Normal floating point differences in root finding can vary across platforms.
    for a, b in zip(observed["cases"], expected["cases"]):
        for key in ("probability_margin", "finite_slope"):
            actual, reference = a.pop(key), b.pop(key)
            assert (actual is None and reference is None) or abs(actual - reference) < 1e-10
    # The JSON round trip also normalizes integer derivative-order keys to strings.
    assert json.loads(json.dumps(observed)) == expected
    root = HERE.parents[1]
    article, supplement = (root / f"focused_{name}.md" for name in ("article", "supplement"))
    assert "nine discrete profiles and all derivatives through order five agree (Appendix J)" in article.read_text()
    note = (HERE / "supplement_note.md").read_text().rstrip()
    assert note in supplement.read_text()
    for row in observed["cases"]:
        assert "$" + row["largest_feasible_weight"] + "$" in note
    result = dict(status="passed", exact_derivative_orders=5, profiles=9, positive_integer_designs=3,
                  checks=observed, source_sha256={p.name: sha256(p.read_bytes()).hexdigest()
                                                  for p in (article, supplement)})
    out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    replay(parser.parse_args().out)
