"""Calibrate the process-aware gate and evaluate its mechanism routing.

The existing process prototype uses exploratory cutoffs.  This script replaces
those cutoffs with a one-sided parametric-bootstrap calibration under the
pooled independent-task MNL null.  It then evaluates the frozen cutoffs on
mechanism-specific simulations.  The script is deliberately lightweight and
does not alter the locked primary benchmark.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from process_gate import CONDITIONS, feature_matrix, fit_and_diagnose, make_data
from run_simulation import N_ALTERNATIVES, fit_mnl, softmax


def fit_beta(x, z, y, ids, tasks):
    features = feature_matrix(x, z)
    rows = np.concatenate([ids * tasks + q for q in range(tasks)])
    return fit_mnl(features[rows], np.concatenate([y[ids, q] for q in range(tasks)]))


def simulate_independent_choices(x, z, beta, rng):
    """Generate choices under the fitted independent-task MNL null."""
    features = feature_matrix(x, z)
    n, tasks, _, _ = x.shape
    linear = np.einsum("ntjp,p->ntj", features.reshape(n, tasks, N_ALTERNATIVES, -1), beta)
    probabilities = softmax(linear.reshape(-1, N_ALTERNATIVES)).reshape(n, tasks, N_ALTERNATIVES)
    choices = np.empty((n, tasks), dtype=int)
    for index, row in enumerate(probabilities.reshape(-1, N_ALTERNATIVES)):
        choices.flat[index] = rng.choice(N_ALTERNATIVES, p=row)
    return choices


def split_ids(n, seed):
    ids = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    cut = int(0.8 * n)
    return ids[:cut], ids[cut:]


def calibrate_null(bootstrap_reps, n, tasks, seed):
    """Return bootstrap cutoffs and the null draws used to obtain them."""
    x, z, y, _, _ = make_data(seed, n, tasks, CONDITIONS[0])
    train_ids, test_ids = split_ids(n, seed + 77)
    beta = fit_beta(x, z, y, train_ids, tasks)
    rng = np.random.default_rng(seed + 7000)
    draws = []
    for _ in range(bootstrap_reps):
        y_boot = simulate_independent_choices(x, z, beta, rng)
        q, r, _, _ = fit_and_diagnose(x, z, y_boot, train_ids, test_ids, tasks)
        draws.append((q, r))
    draws = np.asarray(draws)
    return float(np.quantile(draws[:, 0], 0.95)), float(np.quantile(draws[:, 1], 0.95)), draws


def evaluate_frozen_gate(reps, n, tasks, q_cutoff, r_cutoff, seed):
    rows = []
    for ci, condition in enumerate(CONDITIONS):
        for rep in range(reps):
            data_seed = seed + ci * 1000 + rep
            x, z, y, _, _ = make_data(data_seed, n, tasks, condition)
            train_ids, test_ids = split_ids(n, data_seed + 77)
            q, r, _, _ = fit_and_diagnose(x, z, y, train_ids, test_ids, tasks)
            q_signal = int(q > q_cutoff)
            r_signal = int(r > r_cutoff)
            if q_signal and r_signal:
                route = "process"
            elif q_signal and not r_signal:
                route = "defer_taste_vs_ANA"
            elif r_signal:
                route = "sequence_without_score_signal"
            else:
                route = "pooled_or_utility"
            rows.append({
                "condition": condition.name,
                "replication": rep,
                "q_price": q,
                "inertia_gap": r,
                "q_signal": q_signal,
                "sequence_signal": r_signal,
                "route": route,
                "true_ana": int(condition.ana),
                "true_inertia": int(condition.inertia),
                "true_random_price": int(condition.random_price),
            })
    return rows


def run(bootstrap_reps=200, eval_reps=100, n=400, tasks=12,
        out="results/process_gate_calibration.csv",
        null_out="results/process_gate_null_bootstrap.csv"):
    q_cutoff, r_cutoff, null_draws = calibrate_null(bootstrap_reps, n, tasks, 20261004)
    rows = evaluate_frozen_gate(eval_reps, n, tasks, q_cutoff, r_cutoff, 20262004)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    with open(null_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["q_price", "inertia_gap"])
        writer.writerows(null_draws.tolist())
    with open(out + ".meta", "w") as f:
        f.write(f"q_price_95={q_cutoff:.12g}\n")
        f.write(f"inertia_gap_95={r_cutoff:.12g}\n")
        f.write(f"bootstrap_reps={bootstrap_reps}\n")
        f.write(f"evaluation_reps={eval_reps}\n")
        f.write(f"n={n}\n")
        f.write(f"tasks={tasks}\n")
    print(f"q_price_95={q_cutoff:.6f} inertia_gap_95={r_cutoff:.6f}")
    print(f"wrote calibrated process-gate results to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-reps", type=int, default=200)
    parser.add_argument("--eval-reps", type=int, default=100)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--out", default="results/process_gate_calibration.csv")
    parser.add_argument("--null-out", default="results/process_gate_null_bootstrap.csv")
    args = parser.parse_args()
    run(args.bootstrap_reps, args.eval_reps, args.n, args.tasks, args.out, args.null_out)
