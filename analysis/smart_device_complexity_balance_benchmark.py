"""Compare loose and nuisance-balanced reflection designs.

The data-generating process includes an even comparison-complexity scale based
on the weighted L1 distance.  The loose design matches raw squared distance,
while the balanced design also matches the L1 complexity signature.  Under a
null with no omitted utility, the loose design can reject because its two
reflections have different complexity.  The benchmark makes that boundary
visible and then measures omitted-direction power under the same nuisance.
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


def softmax(values):
    values = np.asarray(values, dtype=float)
    values -= np.max(values)
    out = np.exp(values)
    return out / out.sum()


def weighted_l1(a, b):
    return float(sum(abs(base.COEFFICIENTS[k][x] - base.COEFFICIENTS[k][y])
                     for k, (x, y) in enumerate(zip(a, b))))


def task_records(name):
    candidates = grid.enumerate_candidates()
    if name == "loose":
        pure = [row for row in candidates
                if abs(row["odd_gap_plus"] + row["odd_gap_minus"]) < 1e-9]
        return optimal.choose_fibre_optimal(pure)
    return balanced.choose_balanced(balanced.balanced_pool(candidates))


def task_probability(task, plus, eta, complexity_strength):
    a = task["a_plus"] if plus else task["a_minus"]
    b = task["b_plus"] if plus else task["b_minus"]
    ua = base.candidate_utility(a) + eta * base.omitted_interaction(a)
    ub = base.candidate_utility(b) + eta * base.omitted_interaction(b)
    distance = weighted_l1(a, b)
    denominator = 1.0 + complexity_strength * distance
    return softmax([ua / denominator, ub / denominator,
                    base.OPTOUT_UTILITY / denominator])


def randomization_pvalue(scores, reps, rng):
    observed = abs(float(np.mean(scores)))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(reps, len(scores)))
    draws = np.abs((signs @ scores) / len(scores))
    return float((1.0 + np.sum(draws >= observed)) / (reps + 1.0))


def evaluate(tasks, condition, respondents, permutation_reps, seed,
             complexity_strength):
    eta = 0.9 if condition == "omitted" else 0.0
    rng = np.random.default_rng(seed)
    scores = np.zeros(respondents)
    signs = np.sign([task["odd_gap_plus"] for task in tasks])
    for rid in range(respondents):
        orientation = 1 if rng.integers(0, 2) else -1
        score = 0.0
        for j, task in enumerate(tasks):
            probs = task_probability(task, orientation > 0, eta,
                                     complexity_strength)
            choice = int(rng.choice(3, p=probs))
            contrast = 1.0 if choice == 0 else (-1.0 if choice == 1 else 0.0)
            score += orientation * signs[j] * contrast
        scores[rid] = score / len(tasks)
    pvalue = randomization_pvalue(
        scores, permutation_reps, np.random.default_rng(seed + 1001))
    return int(pvalue < 0.05), pvalue


def run(reps=100, respondents=300, permutation_reps=199,
        complexity_strength=0.20,
        out="results/smart_device_complexity_balance_benchmark.csv"):
    rows = []
    for design in ("loose", "balanced"):
        tasks = task_records(design)
        for condition in ("null_complexity", "omitted_complexity"):
            for rep in range(reps):
                reject, pvalue = evaluate(
                    tasks, "omitted" if condition.startswith("omitted") else "null",
                    respondents, permutation_reps, 97100000 + rep,
                    complexity_strength)
                rows.append({"design": design, "condition": condition,
                             "rep": rep, "reject": reject, "pvalue": pvalue,
                             "task_count": len(tasks),
                             "respondents": respondents,
                             "complexity_strength": complexity_strength})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for design in ("loose", "balanced"):
        for condition in ("null_complexity", "omitted_complexity"):
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
    parser.add_argument("--complexity-strength", type=float, default=0.20)
    parser.add_argument("--out", default="results/smart_device_complexity_balance_benchmark.csv")
    run(**vars(parser.parse_args()))
