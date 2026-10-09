"""Nuisance-balanced reflection design for the smart-device testbed.

The earlier grid matched raw squared distance only.  This module adds the
weighted L1 comparison-complexity distance implied by the candidate additive
utility, together with display-level complexity controls.  A retained pair
must preserve the full candidate menu, the odd interaction, and this nuisance
signature.  The old fibre-optimal grid remains the intentionally looser
comparator.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import smart_device_fibre_candidate_grid as grid
import smart_device_fibre_design as base
import smart_device_fibre_optimal_design as optimal


def nuisance_signature(a, b):
    """Return declared even complexity features for an A--B comparison."""
    component_changes = [
        base.COEFFICIENTS[k][x] - base.COEFFICIENTS[k][y]
        for k, (x, y) in enumerate(zip(a, b))
    ]
    raw_changes = [x - y for x, y in zip(a, b)]
    return (
        round(sum(abs(x) for x in component_changes), 9),  # weighted L1
        sum(x != 0 for x in raw_changes),                 # changed attributes
        sum(abs(x) for x in raw_changes),                 # raw L1 load
        sum(x > 1e-9 for x in component_changes),         # positive directions
        sum(x < -1e-9 for x in component_changes),        # negative directions
    )


def is_balanced(row):
    return nuisance_signature(row["a_plus"], row["b_plus"]) == \
        nuisance_signature(row["a_minus"], row["b_minus"])


def balanced_pool(candidates):
    return [row for row in candidates
            if abs(row["odd_gap_plus"] + row["odd_gap_minus"]) < 1e-9
            and is_balanced(row)]


def choose_balanced(candidates, targets=(0.32, 0.40, 0.57, 0.80, 0.98)):
    chosen = []
    for target in targets:
        pool = [row for row in candidates if abs(row["gap"] - target) <= 0.06]
        if not pool:
            raise ValueError(f"no balanced task near target gap {target}")
        row = max(pool, key=lambda x: (optimal.information_score(x),
                                       -x["hamming"], -x["distance"]))
        chosen.append(dict(row, target_gap=target,
                           nuisance_signature=str(
                               nuisance_signature(row["a_plus"], row["b_plus"]))))
    return chosen


def run(out="results/smart_device_nuisance_balanced_design.csv"):
    all_candidates = grid.enumerate_candidates()
    candidates = balanced_pool(all_candidates)
    rows = []
    for task, row in enumerate(choose_balanced(candidates), 1):
        rows.append({
            "task": task,
            "target_gap": row["target_gap"],
            "candidate_gap": row["gap"],
            "odd_signal": row["signal"],
            "candidate_A": row["u_a"],
            "candidate_B": row["u_b"],
            "candidate_menu": str((round(row["u_a"], 6),
                                    round(row["u_b"], 6), base.OPTOUT_UTILITY)),
            "squared_distance": row["distance"],
            "hamming_changes": row["hamming"],
            "nuisance_signature": row["nuisance_signature"],
            "A_plus": str(row["a_plus"]),
            "A_minus": str(row["a_minus"]),
            "B_plus": str(row["b_plus"]),
            "B_minus": str(row["b_minus"]),
        })
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("candidate_pool", len(all_candidates))
    print("pure_odd_pool", sum(abs(r["odd_gap_plus"] + r["odd_gap_minus"]) < 1e-9
                               for r in all_candidates))
    print("nuisance_balanced_pool", len(candidates))
    print("selected_gaps", [round(row["candidate_gap"], 2) for row in rows])
    print("selected_signals", [round(row["odd_signal"], 2) for row in rows])
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/smart_device_nuisance_balanced_design.csv")
    run(**vars(parser.parse_args()))
