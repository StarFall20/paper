"""Stress-test complexity balance under a moderate-utility choice rule.

The binary data-generating process follows

    P(A>B) = F(Delta V / d_L1),

where d_L1 is the weighted attribute distance.  The loose reflection grid
holds the candidate gap and squared raw distance fixed but leaves d_L1
different across the two members.  The nuisance-balanced grid holds d_L1
fixed as well.  This is a direct negative-control test for the
comparison-complexity objection.
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
import smart_device_fibre_design as base
import smart_device_fibre_optimal_design as optimal
import smart_device_nuisance_balanced_design as balanced


def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def weighted_l1(a, b):
    return float(sum(abs(base.COEFFICIENTS[k][x] - base.COEFFICIENTS[k][y])
                     for k, (x, y) in enumerate(zip(a, b))))


def task_records(design):
    candidates = grid.enumerate_candidates()
    if design == "loose":
        pure = [row for row in candidates
                if abs(row["odd_gap_plus"] + row["odd_gap_minus"]) < 1e-9]
        return optimal.choose_fibre_optimal(pure)
    return balanced.choose_balanced(balanced.balanced_pool(candidates))


def randomization_pvalue(scores, reps, rng):
    observed = abs(float(np.mean(scores)))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(reps, len(scores)))
    draws = np.abs((signs @ scores) / len(scores))
    return float((1.0 + np.sum(draws >= observed)) / (reps + 1.0))


def evaluate(tasks, condition, respondents, permutation_reps, seed):
    rng = np.random.default_rng(seed)
    eta = 0.9 if condition == "omitted_plus_complexity" else 0.0
    signs = np.sign([task["odd_gap_plus"] for task in tasks])
    scores = np.zeros(respondents)
    for rid in range(respondents):
        orientation = 1 if rng.integers(0, 2) else -1
        score = 0.0
        for j, task in enumerate(tasks):
            a = task["a_plus"] if orientation > 0 else task["a_minus"]
            b = task["b_plus"] if orientation > 0 else task["b_minus"]
            gap = base.candidate_utility(a) - base.candidate_utility(b)
            gap += eta * (base.omitted_interaction(a)
                          - base.omitted_interaction(b))
            probability = float(sigmoid(gap / weighted_l1(a, b)))
            choice_a = rng.binomial(1, probability)
            contrast = 1.0 if choice_a else -1.0
            score += orientation * signs[j] * contrast
        scores[rid] = score / len(tasks)
    pvalue = randomization_pvalue(
        scores, permutation_reps, np.random.default_rng(seed + 1001))
    return int(pvalue < 0.05), pvalue


def run(reps=100, respondents=300, permutation_reps=199,
        out="results/smart_device_complexity_balance_binary_benchmark.csv"):
    rows = []
    for design in ("loose", "balanced"):
        tasks = task_records(design)
        for condition in ("complexity_only", "omitted_plus_complexity"):
            for rep in range(reps):
                reject, pvalue = evaluate(
                    tasks, condition, respondents, permutation_reps,
                    98300000 + rep)
                rows.append({"design": design, "condition": condition,
                             "rep": rep, "reject": reject,
                             "pvalue": pvalue, "task_count": len(tasks),
                             "respondents": respondents})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for design in ("loose", "balanced"):
        for condition in ("complexity_only", "omitted_plus_complexity"):
            subset = [r for r in rows if r["design"] == design and
                      r["condition"] == condition]
            print(design, condition, "rejection_rate",
                  round(float(np.mean([r["reject"] for r in subset])), 3))
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--respondents", type=int, default=300)
    parser.add_argument("--permutation-reps", type=int, default=199)
    parser.add_argument("--out", default="results/smart_device_complexity_balance_binary_benchmark.csv")
    run(**vars(parser.parse_args()))
