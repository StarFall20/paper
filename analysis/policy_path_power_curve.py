"""Sample-size power and false-alert curves for policy-path triage."""
from __future__ import annotations

import argparse
import csv
import os
import numpy as np

from policy_path_triage import diagnostics
from recoverability_triage import respondent_split
from run_simulation import CONDITIONS, make_data


def one_design(n, tasks, calibration_reps, eval_reps, seed0):
    cal = []
    for rep in range(calibration_reps):
        seed = seed0 + rep
        x, z, y, _ = make_data(seed, n, tasks, CONDITIONS[0])
        tr, va, _ = respondent_split(n, seed + 7000)
        d, _ = diagnostics(x, z, y, tr, va, tasks)
        cal.append(d)
    d_threshold = float(np.quantile([r["path_structured_l1"] for r in cal], .99))
    q_threshold = float(np.quantile([r["cluster_score"] for r in cal], .975))
    rows = []
    for ci, cond in enumerate(CONDITIONS[:5]):
        policy, hetero, unresolved = [], [], []
        for rep in range(eval_reps):
            seed = seed0 + 100000 + ci * 1000 + rep
            x, z, y, _ = make_data(seed, n, tasks, cond)
            tr, va, _ = respondent_split(n, seed + 7000)
            d, _ = diagnostics(x, z, y, tr, va, tasks)
            p = d["path_structured_l1"] > d_threshold
            h = d["cluster_score"] > q_threshold
            policy.append(p); hetero.append(h); unresolved.append(p and h)
        rows.append({"n": n, "tasks": tasks, "condition": cond.name,
                     "replications": eval_reps,
                     "policy_alert_rate": float(np.mean(policy)),
                     "heterogeneity_alert_rate": float(np.mean(hetero)),
                     "unresolved_rate": float(np.mean(unresolved)),
                     "threshold_policy_path": d_threshold,
                     "threshold_cluster_score": q_threshold})
    return rows


def run(sizes=(200, 400, 800), tasks=12, calibration_reps=50,
        eval_reps=12, out="results/policy_path_power_curve.csv"):
    rows = []
    for idx, n in enumerate(sizes):
        rows.extend(one_design(n, tasks, calibration_reps, eval_reps,
                               24100000 + idx * 1000000))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for r in rows:
        print(r)
    print("wrote", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="200,400,800")
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--calibration-reps", type=int, default=50)
    parser.add_argument("--eval-reps", type=int, default=12)
    parser.add_argument("--out", default="results/policy_path_power_curve.csv")
    args = parser.parse_args()
    run(tuple(int(x) for x in args.sizes.split(",")), args.tasks,
        args.calibration_reps, args.eval_reps, args.out)
