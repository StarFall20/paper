"""Re-run the exact archived allocations and bind the claims to the manuscript."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def replay(archive, food, out):
    out.mkdir(parents=True, exist_ok=False)
    regenerated = out / "results.json"
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    subprocess.run([sys.executable, str(HERE / "verify.py"), "--archive", str(archive),
                    "--food", str(food), "--out", str(regenerated)], env=env, check=True)
    data = regenerated.read_bytes()
    assert data == (HERE / "results.json").read_bytes()
    result = json.loads(data)["summary"]
    assert result["total_allocations"] == 384 and result["direct_state_checks"] == 128
    assert result["opposite_pairs_for_weight"]["0.75"] == 12
    supplement = archive / "focused_supplement.md"
    article = archive / "focused_article.md"
    assert (HERE / "supplement_note.md").read_text().rstrip() in supplement.read_text()
    (out / "verification.json").write_text(json.dumps(dict(
        status="passed", authoritative_summary=result, results_sha256=sha256(data).hexdigest(),
        source_sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in (article, supplement)},
    ), indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("archive", "food", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    replay(a.archive.resolve(), a.food.resolve(), a.out.resolve())
