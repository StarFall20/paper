"""Verify the two forecast conventions from the released food workbook."""

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
    regen = out / "results.json"
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    subprocess.run([sys.executable, str(HERE / "verify.py"), "--archive", str(archive), "--food", str(food),
                    "--out", str(regen)], check=True, env=env)
    data = regen.read_bytes()
    assert data == (HERE / "results.json").read_bytes()
    result = json.loads(data)
    assert result["ordinary_finite_groups"] == 37
    assert result["direct_probability_state_checks"] == 400
    assert result["changed_state_pairs"]["shared_forecast"]["0.9"] == 7
    supplement = archive / "focused_supplement.md"
    article = archive / "focused_article.md"
    assert (HERE / "supplement_note.md").read_text().rstrip() in supplement.read_text()
    provenance = dict(status="passed", results_sha256=sha256(data).hexdigest(),
                      source_sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in (article,supplement)},
                      summary={k:v for k,v in result.items() if k!="groups"})
    (out / "verification.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("archive", "food", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    replay(args.archive.resolve(), args.food.resolve(), args.out.resolve())
