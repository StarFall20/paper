"""Estimate a fibre-violation profile from the balanced task block.

The primary fibre test can reject a candidate relation, but a researcher also
needs to know where in the candidate menu the relation fails.  This script
keeps the same nuisance-balanced tasks and reports a global odd contrast plus
a simultaneous max-T profile.  A localized omitted interaction is used only
to validate the diagnostic; it is not treated as an identified structural
term in the method.
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
    x = np.clip(np.asarray(x, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def weighted_l1(a, b):
    return float(sum(abs(base.COEFFICIENTS[k][x] - base.COEFFICIENTS[k][y])
                     for k, (x, y) in enumerate(zip(a, b))))


def tasks():
    candidates = grid.enumerate_candidates()
    return balanced.choose_balanced(balanced.balanced_pool(candidates))


def task_scores(task_rows, condition, respondents, seed):
    rng = np.random.default_rng(seed)
    matrix = np.zeros((respondents, len(task_rows)))
    for rid in range(respondents):
        orientation = 1 if rng.integers(0, 2) else -1
        for j, task in enumerate(task_rows):
            a = task["a_plus"] if orientation > 0 else task["a_minus"]
            b = task["b_plus"] if orientation > 0 else task["b_minus"]
            gap = base.candidate_utility(a) - base.candidate_utility(b)
            omitted_gap = (base.omitted_interaction(a)
                           - base.omitted_interaction(b))
            if condition == "localized":
                eta = 0.9 if j == 2 else 0.0
            elif condition == "diffuse":
                eta = 0.9
            else:
                eta = 0.0
            p = float(sigmoid((gap + eta * omitted_gap) /
                              weighted_l1(a, b)))
            choice_a = rng.binomial(1, p)
            contrast = 1.0 if choice_a else -1.0
            sign = np.sign(task["odd_gap_plus"])
            matrix[rid, j] = orientation * sign * contrast
    return matrix


def max_t_pvalue(matrix, reps, seed):
    means = matrix.mean(axis=0)
    se = matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0])
    observed = float(np.max(np.abs(means / se)))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]),
                       size=(reps, matrix.shape[0]))
    draws = np.abs((signs @ matrix) / matrix.shape[0]) / se
    pvalue = float((1.0 + np.sum(np.max(draws, axis=1) >= observed)) /
                   (reps + 1.0))
    return int(pvalue < 0.05), pvalue, means, se


def global_pvalue(matrix, reps, seed):
    scores = matrix.mean(axis=1)
    observed = abs(float(scores.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(reps, len(scores)))
    draws = np.abs((signs @ scores) / len(scores))
    pvalue = float((1.0 + np.sum(draws >= observed)) / (reps + 1.0))
    return int(pvalue < 0.05), pvalue


def run(reps=200, respondents=600, permutation_reps=999,
        out="results/smart_device_fibre_violation_profile.csv"):
    task_rows = tasks()
    rows = []
    for condition in ("null", "localized", "diffuse"):
        for rep in range(reps):
            matrix = task_scores(task_rows, condition, respondents,
                                 99400000 + rep)
            global_reject, global_p = global_pvalue(
                matrix, permutation_reps, 99500000 + rep)
            profile_reject, profile_p, means, se = max_t_pvalue(
                matrix, permutation_reps, 99600000 + rep)
            peak = int(np.argmax(np.abs(means / se)))
            active = 2 if condition == "localized" else -1
            rows.append({"condition": condition, "rep": rep,
                         "global_reject": global_reject,
                         "global_pvalue": global_p,
                         "profile_reject": profile_reject,
                         "profile_pvalue": profile_p,
                         "peak_task": peak + 1,
                         "active_task": active + 1 if active >= 0 else "",
                         "localized_correct": int(active >= 0 and peak == active),
                         "respondents": respondents,
                         "task_count": len(task_rows)})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for condition in ("null", "localized", "diffuse"):
        subset = [r for r in rows if r["condition"] == condition]
        print(condition,
              "global", round(float(np.mean([r["global_reject"] for r in subset])), 3),
              "profile", round(float(np.mean([r["profile_reject"] for r in subset])), 3),
              "localization", round(float(np.mean([r["localized_correct"] for r in subset])), 3))
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--respondents", type=int, default=600)
    parser.add_argument("--permutation-reps", type=int, default=999)
    parser.add_argument("--out", default="results/smart_device_fibre_violation_profile.csv")
    run(**vars(parser.parse_args()))
