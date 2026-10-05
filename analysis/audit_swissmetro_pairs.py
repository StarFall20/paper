"""Audit repeated and approximately model-equivalent Swissmetro tasks.

The raw Swissmetro file is not committed. This entry point reads a local
tab-separated copy and reports exact repeated attribute profiles plus
within-respondent pairs whose candidate-model utility-difference vectors are
close. Pair construction uses attributes and frozen coefficients only; choice
labels are never used to form pairs.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict


ATTRIBUTE_FIELDS = (
    "TRAIN_TT", "TRAIN_CO", "TRAIN_HE", "SM_TT", "SM_CO", "SM_HE",
    "SM_SEATS", "CAR_TT", "CAR_CO", "TRAIN_AV", "SM_AV", "CAR_AV",
)


def read_rows(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = []
        for row in reader:
            for key in ATTRIBUTE_FIELDS + ("ID",):
                row[key] = float(row[key])
            rows.append(row)
    return rows


def utility_signature(row, b_time=-1.277859, b_cost=-1.083790,
                      asc_train=-0.701187, asc_car=-0.154633, scale=100.0):
    train = asc_train + b_time * row["TRAIN_TT"] / scale + b_cost * row["TRAIN_CO"] / scale
    swissmetro = b_time * row["SM_TT"] / scale + b_cost * row["SM_CO"] / scale
    car = asc_car + b_time * row["CAR_TT"] / scale + b_cost * row["CAR_CO"] / scale
    return train - swissmetro, car - swissmetro


def run(path, out="results/swissmetro_pair_audit.csv", tolerances=(0.01, 0.02, 0.05, 0.1)):
    rows = read_rows(path)
    by_id = defaultdict(list)
    for row in rows:
        by_id[int(row["ID"])].append(row)
    out_rows = []
    for tolerance in [0.0, *tolerances]:
        pair_count = 0
        respondent_count = 0
        exact_profile_count = 0
        for respondent_rows in by_id.values():
            signatures = [utility_signature(row) for row in respondent_rows]
            local_pairs = 0
            seen = set()
            for i, left in enumerate(respondent_rows):
                profile = tuple(left[key] for key in ATTRIBUTE_FIELDS)
                if profile in seen:
                    exact_profile_count += 1
                seen.add(profile)
                for j in range(i + 1, len(respondent_rows)):
                    right = signatures[j]
                    if max(abs(signatures[i][k] - right[k]) for k in (0, 1)) <= tolerance:
                        local_pairs += 1
            pair_count += local_pairs
            respondent_count += int(local_pairs > 0)
        out_rows.append({
            "tolerance": tolerance,
            "rows": len(rows),
            "respondents": len(by_id),
            "within_respondent_pairs": pair_count,
            "respondents_with_pair": respondent_count,
            "exact_profile_repeats": exact_profile_count,
        })
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        writer.writeheader(); writer.writerows(out_rows)
    print(f"wrote Swissmetro pair audit to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="local path to swissmetro.dat")
    parser.add_argument("--out", default="results/swissmetro_pair_audit.csv")
    args = parser.parse_args()
    run(args.path, args.out)
