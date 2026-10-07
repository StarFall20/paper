"""Find the smallest attribute-support repair for a blind fibre direction.

The FQC audit reports when a departure is constant on every candidate fibre.
This companion returns a concrete support change that makes the departure vary.
It is a design-repair certificate, not a post-outcome model-selection step.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
from collections import defaultdict

import numpy as np

import structural_fibre_design as structural


def endpoint_interaction(profile):
    mode, _intelligence, support, _evidence, _data, _price = profile
    return int(mode == 2 and support == 2)


def fibre_variation(max_level):
    groups = defaultdict(list)
    for profile in itertools.product(range(max_level + 1), repeat=6):
        groups[tuple(structural.candidate_basis(profile))].append(
            (profile, endpoint_interaction(profile)))
    best = None
    for basis, members in groups.items():
        values = [v for _profile, v in members]
        spread = max(values) - min(values)
        if spread <= 0:
            continue
        candidate = {
            "max_level": max_level,
            "basis": basis,
            "spread": spread,
            "members": members,
            "group_count": len(groups),
        }
        if best is None or (spread, len(members)) > \
                (best["spread"], len(best["members"])):
            best = candidate
    return best, len(groups)


def run(out="results/fibre_support_repair_certificate.csv"):
    pool, raw, matrix, _scales = structural.prepare_pool()
    endpoint_column = np.asarray([
        0.5 * ((endpoint_interaction(row["a_plus"]) -
                endpoint_interaction(row["b_plus"])) -
               (endpoint_interaction(row["a_minus"]) -
                endpoint_interaction(row["b_minus"])))
        for row in pool
    ])
    rows = [{"domain_max_level": 2,
             "candidate_fibre_endpoint_spread": 0,
             "pool_endpoint_odd_nonzero": int(np.any(np.abs(endpoint_column) > 1e-12)),
             "status": "blind"}]
    repair = None
    for max_level in (2, 3, 4):
        best, group_count = fibre_variation(max_level)
        rows.append({"domain_max_level": max_level,
                     "candidate_fibre_endpoint_spread": 0 if best is None else best["spread"],
                     "pool_endpoint_odd_nonzero": int(np.any(np.abs(endpoint_column) > 1e-12)) if max_level == 2 else "",
                     "status": "blind" if best is None else "testable",
                     "group_count": group_count,
                     "example_basis": "" if best is None else str(tuple(int(x) for x in best["basis"])),
                     "example_profiles": "" if best is None else str([x[0] for x in best["members"][:6]])})
        if repair is None and best is not None:
            repair = (max_level, best)
    if repair is None:
        raise RuntimeError("no support repair found")
    max_level, best = repair
    rows.append({"domain_max_level": "repair",
                 "candidate_fibre_endpoint_spread": best["spread"],
                 "pool_endpoint_odd_nonzero": "",
                 "status": "add_level",
                 "group_count": best["group_count"],
                 "example_basis": str(tuple(int(x) for x in best["basis"])),
                 "example_profiles": str([x[0] for x in best["members"][:6]])})
    fields = sorted({key for row in rows for key in row})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print("current_pool", len(pool), "current_endpoint_odd_nonzero",
          bool(np.any(np.abs(endpoint_column) > 1e-12)))
    print("minimal_repair_max_level", max_level)
    print("repair_basis", tuple(int(x) for x in best["basis"]))
    print("repair_spread", best["spread"])
    print("repair_profiles", [x[0] for x in best["members"][:6]])
    print("wrote", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/fibre_support_repair_certificate.csv")
    run(**vars(parser.parse_args()))
