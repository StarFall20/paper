"""Random-taste boundary check for the nuisance-balanced fibre design.

The task fibre is constructed from a calibrated population candidate.  This
benchmark asks whether respondent-level random taste, left outside that
candidate, is detected as a relation violation.  It is a boundary condition,
not evidence that random taste is an omitted interaction.
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
import smart_device_nuisance_balanced_design as balanced


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -35, 35)))


def weighted_l1(a, b):
    return float(sum(abs(base.COEFFICIENTS[k][x] - base.COEFFICIENTS[k][y])
                     for k, (x, y) in enumerate(zip(a, b))))


def mean_coefficients():
    return np.asarray([base.COEFFICIENTS[k][level]
                       for k in range(6) for level in (1, 2)], dtype=float)


def value(profile, beta):
    total = 0.0
    for k, level in enumerate(profile):
        if level:
            total += beta[2 * k + level - 1]
    return float(total)


def randomization_pvalue(scores, reps, rng):
    observed = abs(float(np.mean(scores)))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(reps, len(scores)))
    draws = np.abs((signs @ scores) / len(scores))
    return float((1.0 + np.sum(draws >= observed)) / (reps + 1.0))


def evaluate(tasks, condition, respondents, permutation_reps, seed):
    rng = np.random.default_rng(seed)
    beta_bar = mean_coefficients()
    sd = {"homogeneous": 0.0, "random_taste_0.1": 0.1,
          "random_taste_0.2": 0.2, "omitted": 0.0}[condition]
    eta = 0.9 if condition == "omitted" else 0.0
    scores = np.zeros(respondents)
    signs = np.sign([task["odd_gap_plus"] for task in tasks])
    for rid in range(respondents):
        beta = beta_bar + rng.normal(0.0, sd, size=len(beta_bar))
        orientation = 1 if rng.integers(0, 2) else -1
        score = 0.0
        for j, task in enumerate(tasks):
            a = task["a_plus"] if orientation > 0 else task["a_minus"]
            b = task["b_plus"] if orientation > 0 else task["b_minus"]
            gap = value(a, beta) - value(b, beta)
            gap += eta * (base.omitted_interaction(a)
                          - base.omitted_interaction(b))
            p = float(sigmoid(gap / weighted_l1(a, b)))
            choice_a = rng.binomial(1, p)
            contrast = 1.0 if choice_a else -1.0
            score += orientation * signs[j] * contrast
        scores[rid] = score / len(tasks)
    pvalue = randomization_pvalue(
        scores, permutation_reps, np.random.default_rng(seed + 101))
    return int(pvalue < 0.05), pvalue


def run(reps=200, respondents=600, permutation_reps=999,
        out="results/smart_device_random_taste_balance_benchmark.csv"):
    candidates = grid.enumerate_candidates()
    tasks = balanced.choose_balanced(balanced.balanced_pool(candidates))
    rows = []
    for condition in ("homogeneous", "random_taste_0.1",
                      "random_taste_0.2", "omitted"):
        for rep in range(reps):
            reject, pvalue = evaluate(
                tasks, condition, respondents, permutation_reps,
                99700000 + rep)
            rows.append({"condition": condition, "rep": rep,
                         "reject": reject, "pvalue": pvalue,
                         "respondents": respondents,
                         "task_count": len(tasks)})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for condition in ("homogeneous", "random_taste_0.1",
                      "random_taste_0.2", "omitted"):
        subset = [r for r in rows if r["condition"] == condition]
        print(condition, "rejection_rate",
              round(float(np.mean([r["reject"] for r in subset])), 3))
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--respondents", type=int, default=600)
    parser.add_argument("--permutation-reps", type=int, default=999)
    parser.add_argument("--out", default="results/smart_device_random_taste_balance_benchmark.csv")
    run(**vars(parser.parse_args()))
