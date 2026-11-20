"""Replay the same-menu comparison from pinned public inputs."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def replay(archive, food, swiss, out):
    out.mkdir(parents=True, exist_ok=False)
    regenerated = out / "results.json"
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    subprocess.run([sys.executable, str(HERE / "verify.py"), "--archive", str(archive),
                    "--food-data", str(food), "--swiss-data", str(swiss), "--out", str(regenerated)],
                   check=True, env=env)
    data = regenerated.read_bytes()
    assert data == (HERE / "results.json").read_bytes()
    parsed = json.loads(data)
    common = parsed["common_forecast_by_exact_auxiliary_menu"]
    assert [common[key]["counts"]["opposite_state_pairs"] for key in ("1", "0.5")] == [70, 0]
    assert sum(len(v["allocation_records"]) for v in common.values()) == 768
    supplement = archive / "focused_supplement.md"
    assert (HERE / "supplement_note.md").read_text().rstrip() in supplement.read_text()
    article = archive / "focused_article.md"
    (out / "verification.json").write_text(json.dumps(dict(
        status="passed", checked_configurations=768, unit_weight_opposite_pairs=70, half_weight_opposite_pairs=0,
        reports_sha256=sha256(data).hexdigest(),
        source_sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in (article, supplement)},
    ), indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("archive", "food", "swiss", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    replay(a.archive.resolve(), a.food.resolve(), a.swiss.resolve(), a.out.resolve())
