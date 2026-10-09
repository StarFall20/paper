"""Compare ordinary candidate grid and fibre-optimal smart-device designs.

Both designs use five candidate-preserving A/B/opt-out tasks. The benchmark
keeps respondent count, task count, and sign-flip inference fixed, then compares
power for the existing odd interaction and size under an even scale nuisance.
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
import smart_device_fibre_optimal_design as optimal
import smart_device_fibre_design as base


def softmax(values):
    values = np.asarray(values, dtype=float)
    values -= np.max(values)
    ex = np.exp(values)
    return ex / ex.sum()


def randomization_pvalue(scores, reps, rng):
    observed = abs(float(np.mean(scores)))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(reps, len(scores)))
    draws = np.abs((signs @ scores) / len(scores))
    return float((1.0 + np.sum(draws >= observed)) / (reps + 1.0))


def task_probability(task, condition, plus, eta_strength):
    if plus:
        a, b = task["a_plus"], task["b_plus"]
    else:
        a, b = task["a_minus"], task["b_minus"]
    eta = eta_strength if condition in ("omitted", "combined") else 0.0
    scale = 0.03 if condition in ("even_scale", "combined") else 0.0
    ua = base.candidate_utility(a) + eta * base.omitted_interaction(a)
    ub = base.candidate_utility(b) + eta * base.omitted_interaction(b)
    d = base.squared_distance(a, b)
    denom = 1.0 + scale * d
    return softmax([ua / denom, ub / denom, base.OPTOUT_UTILITY / denom])


def task_records(which):
    candidates = grid.enumerate_candidates()
    if which == "grid":
        return grid.choose_grid(candidates)
    pure = [row for row in candidates
            if abs(row["odd_gap_plus"] + row["odd_gap_minus"]) < 1e-9]
    return optimal.choose_fibre_optimal(pure)


def evaluate(tasks, condition, respondents, permutation_reps, seed, eta_strength):
    rng = np.random.default_rng(seed)
    p_plus = [task_probability(task, condition, True, eta_strength) for task in tasks]
    p_minus = [task_probability(task, condition, False, eta_strength) for task in tasks]
    signs = np.sign([task["odd_gap_plus"] for task in tasks])
    scores = np.zeros(respondents)
    for rid in range(respondents):
        # The randomization unit is the respondent block: one orientation
        # coin is shared across all tasks in the block.
        orientation = 1 if rng.integers(0, 2) else -1
        respondent_score = 0.0
        for j, (pp, pm) in enumerate(zip(p_plus, p_minus)):
            probs = pp if orientation > 0 else pm
            choice = int(rng.choice(3, p=probs))
            contrast = 1.0 if choice == 0 else (-1.0 if choice == 1 else 0.0)
            respondent_score += orientation * signs[j] * contrast
        scores[rid] = respondent_score / len(tasks)
    pvalue = randomization_pvalue(
        scores, permutation_reps, np.random.default_rng(seed + 1001))
    return int(pvalue < 0.05), pvalue


def run(reps=100, respondents=400, permutation_reps=199,
        out="results/smart_device_multitask_benchmark.csv"):
    rows = []
    for eta_strength in (0.3, 1.0):
        for design_name in ("grid", "fibre_optimal"):
            tasks = task_records(design_name)
            for condition in ("null", "omitted", "even_scale", "combined"):
                for rep in range(reps):
                    reject, pvalue = evaluate(
                        tasks, condition, respondents, permutation_reps,
                        96000000 + rep, eta_strength)
                    rows.append({"eta_strength": eta_strength,
                                 "design": design_name, "condition": condition,
                                 "rep": rep, "reject": reject, "pvalue": pvalue,
                                 "task_count": len(tasks),
                                 "respondents": respondents})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    for eta_strength in (0.3, 1.0):
        for design_name in ("grid", "fibre_optimal"):
            for condition in ("null", "omitted", "even_scale", "combined"):
                subset = [r for r in rows if r["eta_strength"] == eta_strength and
                          r["design"] == design_name and
                          r["condition"] == condition]
                print(eta_strength, design_name, condition, "rejection_rate",
                      round(float(np.mean([r["reject"] for r in subset])), 3))
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--respondents", type=int, default=400)
    parser.add_argument("--permutation-reps", type=int, default=199)
    parser.add_argument("--out", default="results/smart_device_multitask_benchmark.csv")
    run(**vars(parser.parse_args()))
