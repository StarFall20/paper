"""Minimal proof-of-concept for model-equivalent choice-pair tasks.

Two binary choice tasks share the same candidate-model utility difference but
use different attribute levels. Under the candidate model their choice
probabilities are equal. A systematic difference in observed choice shares is
an equivalence violation and can reject the candidate utility basis.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def true_difference(q_a, p_a, q_b, p_b, nonlinear):
    value = 0.8 * (q_a - q_b) - 0.6 * (p_a - p_b)
    if nonlinear:
        value += 0.35 * (q_a ** 2 - q_b ** 2)
    return value


def run(reps=200, n_per_pair=300, bootstrap=400, alpha=0.05,
        out="results/equivalent_pair_test.csv"):
    rng = np.random.default_rng(20261005)
    rows = []
    # Both pairs have candidate-model difference 0.8. Only the nonlinear DGP
    # changes that difference across the two attribute decompositions.
    pairs = ((2.0, 1.0, 1.0, 1.0), (4.0, 1.0, 3.0, 1.0))
    for nonlinear in (False, True):
        for rep in range(reps):
            probs = [logistic(true_difference(*pair, nonlinear)) for pair in pairs]
            y = [rng.binomial(n_per_pair, prob) for prob in probs]
            phat = [count / n_per_pair for count in y]
            observed = phat[1] - phat[0]
            # Parametric null: both pairs share a common probability estimated
            # under the equality restriction. This keeps the test independent
            # of the true DGP used to generate the row.
            pooled = sum(y) / (2.0 * n_per_pair)
            null_diffs = rng.binomial(n_per_pair, pooled, size=bootstrap) / n_per_pair
            null_diffs -= rng.binomial(n_per_pair, pooled, size=bootstrap) / n_per_pair
            pvalue = (1.0 + np.sum(np.abs(null_diffs) >= abs(observed))) / (bootstrap + 1.0)
            rows.append({
                "dgp": "nonlinear" if nonlinear else "additive",
                "rep": rep,
                "n_per_pair": n_per_pair,
                "share_pair_low": phat[0],
                "share_pair_high": phat[1],
                "equivalence_gap": observed,
                "pvalue": pvalue,
                "reject": int(pvalue < alpha),
            })
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for dgp in ("additive", "nonlinear"):
        subset = [r for r in rows if r["dgp"] == dgp]
        print(dgp, "mean_gap", round(float(np.mean([r["equivalence_gap"] for r in subset])), 4),
              "rejection_rate", round(float(np.mean([r["reject"] for r in subset])), 4))
    print(f"wrote {len(rows)} equivalent-pair rows to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n-per-pair", type=int, default=300)
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="results/equivalent_pair_test.csv")
    args = ap.parse_args()
    run(**vars(args))
