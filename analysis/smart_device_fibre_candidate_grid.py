"""Enumerate a pre-outcome grid of smart-device parity tasks.

The grid uses the supplied manuscript's attribute levels and additive MNL
coefficients. It selects one exact candidate-preserving reflection near each
of five target A--B gaps, subject to an odd intelligence/cloud or home/support
interaction, equal squared raw-level distance, and a compact transformation.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import smart_device_fibre_design as base


def enumerate_candidates():
    profiles = []
    for profile in itertools.product(range(3), repeat=6):
        profiles.append((profile, base.candidate_utility(profile),
                         base.omitted_interaction(profile)))
    groups = defaultdict(list)
    for row in profiles:
        groups[row[1]].append(row)
    candidates = []
    for ua, group_a in groups.items():
        for a_plus, a_minus in itertools.combinations(group_a, 2):
            dh_a = a_plus[2] - a_minus[2]
            hamming_a = sum(x != y for x, y in zip(a_plus[0], a_minus[0]))
            if abs(dh_a) < 0.3:
                continue
            for ub, group_b in groups.items():
                gap = round(ua - ub, 6)
                if not 0.2 <= gap <= 1.0:
                    continue
                for b_plus, b_minus in itertools.combinations(group_b, 2):
                    dh_b = b_plus[2] - b_minus[2]
                    hamming_b = sum(x != y for x, y in zip(b_plus[0], b_minus[0]))
                    if abs(dh_a + dh_b) > 1e-9:
                        continue
                    if abs(dh_a - dh_b) < 0.5:
                        continue
                    d_plus = base.squared_distance(a_plus[0], b_plus[0])
                    d_minus = base.squared_distance(a_minus[0], b_minus[0])
                    if abs(d_plus - d_minus) > 1e-9:
                        continue
                    candidates.append({
                        "gap": gap, "signal": abs(dh_a - dh_b) / 2.0,
                        "distance": d_plus,
                        "hamming": hamming_a + hamming_b,
                        "a_plus": a_plus[0], "a_minus": a_minus[0],
                        "b_plus": b_plus[0], "b_minus": b_minus[0],
                        "u_a": ua, "u_b": ub,
                        "odd_gap_plus": a_plus[2] - b_plus[2],
                        "odd_gap_minus": a_minus[2] - b_minus[2],
                    })
    # Remove exact duplicates, then use a compactness-first ordering.
    unique, seen = [], set()
    for row in sorted(candidates, key=lambda x: (x["hamming"], x["distance"],
                                                  -x["signal"], x["gap"])):
        key = (row["a_plus"], row["a_minus"], row["b_plus"], row["b_minus"])
        if key not in seen:
            seen.add(key); unique.append(row)
    return unique


def choose_grid(candidates, targets=(0.2, 0.4, 0.6, 0.8, 1.0)):
    chosen = []
    for target in targets:
        row = min(candidates, key=lambda x: (abs(x["gap"] - target),
                                              x["hamming"], x["distance"],
                                              -x["signal"]))
        chosen.append(dict(row, target_gap=target))
    return chosen


def profile_text(profile):
    return " / ".join(base.LABELS[attr][level]
                      for attr, level in zip(base.ATTRIBUTES, profile))


def run(out="results/smart_device_fibre_candidates.csv"):
    candidates = enumerate_candidates()
    chosen = choose_grid(candidates)
    rows = []
    for idx, row in enumerate(chosen, 1):
        ua, ub = row["u_a"], row["u_b"]
        rows.append({
            "task": idx, "target_gap": row["target_gap"], "candidate_gap": row["gap"],
            "candidate_A": ua, "candidate_B": ub,
            "candidate_menu": str((round(ua, 6), round(ub, 6), base.OPTOUT_UTILITY)),
            "distance_plus": row["distance"], "distance_minus": row["distance"],
            "odd_gap_plus": row["odd_gap_plus"], "odd_gap_minus": row["odd_gap_minus"],
            "odd_signal": row["signal"], "hamming_changes": row["hamming"],
            "A_plus": profile_text(row["a_plus"]),
            "A_minus": profile_text(row["a_minus"]),
            "B_plus": profile_text(row["b_plus"]),
            "B_minus": profile_text(row["b_minus"]),
        })
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print("enumerated", len(candidates), "unique candidate-preserving reflections")
    for row in rows:
        print(row["task"], "gap", row["candidate_gap"], "signal", row["odd_signal"],
              "distance", row["distance_plus"], "changes", row["hamming_changes"])
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/smart_device_fibre_candidates.csv")
    run(**vars(parser.parse_args()))
