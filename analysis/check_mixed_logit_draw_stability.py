"""Check how the targeted Mixed Logit extension changes with draw count."""
from __future__ import annotations
import argparse
import csv
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_mixed_logit_extension import fit_mixed_logit, score_mixed_logit
from run_simulation import CONDITIONS, feature_matrix, make_data


def run(reps=2, n=400, tasks=12, out="results/mixed_logit_draw_stability.csv"):
    target_conditions = {"heterogeneity", "combined"}
    draw_levels = (20, 40, 80)
    rows = []
    for ci, cond in enumerate(CONDITIONS):
        if cond.name not in target_conditions:
            continue
        for rep in range(reps):
            seed = 20501005 + ci * 1000 + rep
            x, z, choices, v = make_data(seed, n, tasks, cond)
            X = feature_matrix(x, z, structured=False).reshape(n, tasks, 3, -1)
            ids = np.arange(n)
            np.random.default_rng(seed + 7).shuffle(ids)
            cut = int(0.8 * n)
            train_ids, test_ids = ids[:cut], ids[cut:]
            for draws in draw_levels:
                beta, sigma = fit_mixed_logit(X[train_ids], choices[train_ids], draws=draws, seed=seed + 31)
                vals = score_mixed_logit(X[test_ids], choices[test_ids], v[test_ids], beta, sigma,
                                         draws=draws, seed=seed + 41)
                rows.append({"condition": cond.name, "replication": rep, "draws": draws,
                             "random_price_sd": sigma, "accuracy": vals[0], "logloss": vals[1],
                             "brier": vals[2], "share_rmse": vals[3], "decision_regret": vals[4]})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} draw-stability rows to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--tasks", type=int, default=12)
    ap.add_argument("--out", default="results/mixed_logit_draw_stability.csv")
    args = ap.parse_args()
    run(args.reps, args.n, args.tasks, args.out)
