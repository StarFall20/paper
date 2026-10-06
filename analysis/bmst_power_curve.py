"""Sample-size and focal-pair sensitivity for the one-member BMST test.

This is a planning analysis. It uses the locked assignment generator and
varies respondents and the number of focal pairs. The output is reported as
empirical size/power, not as a new estimate of the main effect.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from bmst_assignment_benchmark import evaluate, generate


CONDITIONS = ("additive", "nonlinear", "interaction")


def generate_with_pairs(seed: int, respondents: int, focal_pairs: int,
                        condition: str):
    """Generate the locked three-pair design, then retain a focal subset.

    The existing generator has three focal pairs. For planning purposes, the
    first pair is always retained; additional pairs are retained in order.
    This keeps the pair construction identical to the benchmark while making
    the task-count effect explicit.
    """
    data = generate(seed, respondents=respondents, condition=condition)
    if focal_pairs >= 3:
        return data
    ids, y, x, orientations, focal = data
    kept = np.zeros_like(focal, dtype=bool)
    for rid in np.unique(ids):
        rows = np.flatnonzero((ids == rid) & focal)
        kept[rows[:focal_pairs]] = True
    return ids, y, x, orientations, kept


def run(reps: int = 30, respondents=(150, 300, 600),
        focal_pairs=(1, 3), randomization_reps: int = 199,
        out: str = "results/bmst_power_curve.csv"):
    rows = []
    for n in respondents:
        for pairs in focal_pairs:
            for condition in CONDITIONS:
                # The full-three-pair arm is the estimand used in the paper;
                # the one-pair arm is a task-count sensitivity analysis.
                rejects = []
                clusters = []
                for rep in range(reps):
                    data = generate_with_pairs(
                        20261030 + 100000 * n + 1000 * pairs + rep,
                        respondents=n, focal_pairs=pairs, condition=condition,
                    )
                    pvalue, nclusters = evaluate(
                        data, 20500000 + n + pairs * 100 + rep,
                        randomization_reps,
                    )
                    rejects.append(int(pvalue < 0.05))
                    clusters.append(nclusters)
                rows.append({
                    "respondents": n,
                    "focal_pairs": pairs,
                    "condition": condition,
                    "replications": reps,
                    "rejection_rate": float(np.mean(rejects)),
                    "mean_clusters": float(np.mean(clusters)),
                })
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fields = list(rows[0])
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"wrote {len(rows)} BMST power-curve rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--respondents", type=int, nargs="+",
                        default=[150, 300, 600])
    parser.add_argument("--focal-pairs", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--randomization-reps", type=int, default=199)
    parser.add_argument("--out", default="results/bmst_power_curve.csv")
    args = parser.parse_args()
    run(**vars(args))
