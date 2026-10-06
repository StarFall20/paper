"""Compare fixed and maximin paired-task designs in the randomization benchmark.

The design-score audit is outcome-free.  This companion run checks its actual
finite-sample consequences under the same candidate-preserving randomization
test used by the main prototype.  It reports null rejection and power for the
fixed three-shift instrument, the constrained maximin design, and the
unconstrained collapse boundary.  The latter is intentionally included as a
negative control: a generic criterion that repeats one large shift is not a
credible experimental design.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from metamorphic_design_score import design_score, scores
from randomization_paired_task_test import evaluate
from exact_paired_task_benchmark import simulate


BASE_SHIFTS = ((20, 0), (0, 60), (20, 50))
CONSTRAINED_SHIFTS = ((5, 45), (20, 0), (35, 45))
UNCONSTRAINED_SHIFTS = ((40, 60), (40, 60), (40, 60))
DESIGNS = {
    "fixed": BASE_SHIFTS,
    "constrained_maximin": CONSTRAINED_SHIFTS,
    "unconstrained_maximin": UNCONSTRAINED_SHIFTS,
}


def named_shifts(shifts):
    return {f"shift_{i}": np.asarray(shift, dtype=float)
            for i, shift in enumerate(shifts, 1)}


def run(reps=50, randomization_reps=199, respondents=300,
        out="results/metamorphic_design_benchmark.csv"):
    rows = []
    conditions = ("additive", "nonlinear", "threshold", "interaction",
                  "random_price", "nonlinear_random_price",
                  "interaction_random_price", "cubic")
    for design_name, shifts in DESIGNS.items():
        for condition in conditions:
            for rep in range(reps):
                data = simulate(20261007 + rep, respondents=respondents,
                                condition=condition,
                                shift_specs=named_shifts(shifts))
                oracle_term = {"nonlinear": "nonlinear", "threshold": "threshold",
                               "interaction": "interaction",
                               "nonlinear_random_price": "nonlinear",
                               "interaction_random_price": "interaction"}.get(condition)
                candidates = [None] if oracle_term is None else [None, oracle_term]
                for term in candidates:
                    records = evaluate(*data, term=term,
                                       randomization_reps=randomization_reps,
                                       seed=20600000 + rep)
                    record = next(row for row in records if row["pair_type"] == "all")
                    rows.append({"design": design_name, "condition": condition,
                                 "rep": rep,
                                 "candidate": "oracle_term" if term else "additive",
                                 "pairs": record["pairs"],
                                 "pvalue": record["pvalue"],
                                 "reject": int(record["pvalue"] < 0.05)})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    for design_name in DESIGNS:
        for condition in conditions:
            subset = [row for row in rows if row["design"] == design_name
                      and row["condition"] == condition
                      and row["candidate"] == "additive"]
            print(design_name, condition,
                  "rejection_rate", round(float(np.mean([row["reject"] for row in subset])), 3))
    print(f"wrote {len(rows)} metamorphic design-benchmark rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--randomization-reps", type=int, default=199)
    parser.add_argument("--respondents", type=int, default=300)
    parser.add_argument("--out", default="results/metamorphic_design_benchmark.csv")
    run(**vars(parser.parse_args()))
