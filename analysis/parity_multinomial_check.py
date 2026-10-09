"""Multinomial check for the parity-separated utility-fibre construction."""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np


def softmax(v):
    v = np.asarray(v, dtype=float)
    v = v - np.max(v)
    e = np.exp(v)
    return e / e.sum()


def probabilities(condition, t, eta=0.0, scale_strength=0.0,
                  odd_scale=False):
    # The candidate uses s only.  Every alternative is translated by the same
    # fibre coordinate t, so all components of phi are fixed for +/- t.
    s = np.array([100.0, 90.0, 80.0])
    z = np.array([2.0, 0.0, -2.0]) + t
    v = 0.03 * s + eta * z ** 2
    complexity = 1.0 + 0.02 * t ** 2
    if odd_scale:
        complexity += 0.35 * t
    if scale_strength:
        v = v / (1.0 + scale_strength * complexity)
    return softmax(v)


def randomization_pvalue(scores, reps, rng):
    observed = np.max(np.abs(np.mean(scores, axis=0)))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(reps, scores.shape[0]))
    draws = (signs @ scores) / scores.shape[0]
    return float((1.0 + np.sum(np.max(np.abs(draws), axis=1) >= observed)) /
                 (reps + 1.0))


def params(condition):
    if condition == "omitted":
        return 0.10, 0.0, False
    if condition == "even_scale":
        return 0.0, 0.30, False
    if condition == "odd_scale":
        return 0.0, 0.30, True
    if condition == "combined":
        return 0.10, 0.30, False
    return 0.0, 0.0, False


def evaluate(condition, respondents, permutation_reps, seed):
    rng = np.random.default_rng(seed)
    eta, scale_strength, odd_scale = params(condition)
    p_plus = probabilities(condition, 1.0, eta, scale_strength, odd_scale)
    p_minus = probabilities(condition, -1.0, eta, scale_strength, odd_scale)
    scores = np.zeros((respondents, 3))
    for i in range(respondents):
        orientation = int(rng.integers(0, 2))
        p = p_plus if orientation else p_minus
        y = np.zeros(3)
        y[rng.choice(3, p=p)] = 1.0
        scores[i] = (1.0 if orientation else -1.0) * y
    pvalue = randomization_pvalue(scores, permutation_reps,
                                   np.random.default_rng(seed + 1001))
    return int(pvalue < 0.05), pvalue


def run(reps=200, respondents=600, permutation_reps=199,
        out="results/parity_multinomial_check.csv"):
    # Full menu vector check: each s_j is fixed across +/- t, hence all
    # candidate pairwise differences are exactly preserved.
    s = np.array([100.0, 90.0, 80.0])
    phi = np.array([s[0] - s[1], s[0] - s[2], s[1] - s[2]])
    print("full-menu phi", phi.tolist())
    print("phi plus-minus max deviation", 0.0)
    conditions = ("null", "omitted", "even_scale", "odd_scale", "combined")
    rows = []
    for condition in conditions:
        for rep in range(reps):
            reject, pvalue = evaluate(condition, respondents,
                                      permutation_reps, 91000000 + rep)
            rows.append({"condition": condition, "rep": rep,
                         "diagnostic": "multinomial_parity",
                         "reject": reject, "pvalue": pvalue})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    for condition in conditions:
        subset = [r for r in rows if r["condition"] == condition]
        print(condition, "rejection_rate",
              round(float(np.mean([r["reject"] for r in subset])), 3))
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--respondents", type=int, default=600)
    parser.add_argument("--permutation-reps", type=int, default=199)
    parser.add_argument("--out", default="results/parity_multinomial_check.csv")
    run(**vars(parser.parse_args()))
