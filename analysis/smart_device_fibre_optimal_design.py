"""Fibre-optimal design criterion for the smart-device supplement.

The criterion maximizes the local Fisher signal for an odd omitted interaction
while preserving the candidate menu and an even distance nuisance. It is a
within-fibre analogue of model-discrimination design: rival mechanisms are
separated only after the candidate predicts the same menu.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import smart_device_fibre_candidate_grid as grid


def logistic(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def information_score(row):
    # At eta=0, the derivative of the odd probability contrast is
    # p(1-p) times the odd omitted-gap signal.  We use the binary A-vs-B
    # component so the score remains transparent and pre-outcome.
    gap = float(row["gap"])
    p = float(logistic(gap))
    variance = p * (1.0 - p)
    signal = float(row["signal"])
    return variance * signal * signal


def choose_fibre_optimal(candidates, targets=(0.2, 0.4, 0.6, 0.8, 1.0)):
    chosen = []
    for target in targets:
        pool = [row for row in candidates if abs(row["gap"] - target) <= 0.06]
        if not pool:
            pool = candidates
        row = max(pool, key=lambda x: (information_score(x),
                                       -x["hamming"], -x["distance"]))
        chosen.append(dict(row, target_gap=target,
                           fisher_odd_signal=information_score(row)))
    return chosen


def run(out="results/smart_device_fibre_optimal_design.csv"):
    candidates = [row for row in grid.enumerate_candidates()
                  if abs(row["odd_gap_plus"] + row["odd_gap_minus"]) < 1e-9]
    chosen = choose_fibre_optimal(candidates)
    rows = []
    for idx, row in enumerate(chosen, 1):
        rows.append({
            "task": idx, "target_gap": row["target_gap"],
            "candidate_gap": row["gap"], "odd_signal": row["signal"],
            "fisher_odd_signal": row["fisher_odd_signal"],
            "distance": row["distance"], "hamming_changes": row["hamming"],
            "candidate_A": row["u_a"], "candidate_B": row["u_b"],
            "candidate_menu": str((round(row["u_a"], 6),
                                    round(row["u_b"], 6), -0.42)),
            "odd_gap_plus": row["odd_gap_plus"],
            "odd_gap_minus": row["odd_gap_minus"],
        })
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print("candidate_pool", len(candidates))
    for row in rows:
        print(row["task"], "gap", row["candidate_gap"],
              "fisher_odd_signal", round(row["fisher_odd_signal"], 6))
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/smart_device_fibre_optimal_design.csv")
    run(**vars(parser.parse_args()))
