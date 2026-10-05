"""Exploratory respondent-score gate for continuous taste heterogeneity.

The gate fits a behavioural MNL on development respondents and evaluates the
out-of-fold respondent-level score for the price coefficient.  The ratio of
the squared score norm to its model-based information is close to one under
pooling and rises when respondent-specific price sensitivity is present.  The
script is a development diagnostic; the final paper must calibrate the gate
with a parametric bootstrap and pre-register its threshold.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from run_simulation import CONDITIONS, PRICE_INDEX, feature_matrix, fit_mnl, make_data


def score_overdispersion(x4: np.ndarray, y2: np.ndarray, beta: np.ndarray) -> float:
    """Return a respondent-level price-score overdispersion ratio."""
    n, tasks, _, _ = x4.shape
    probabilities = np.exp(
        np.einsum("ntjp,p->ntj", x4, beta)
        - np.max(np.einsum("ntjp,p->ntj", x4, beta), axis=2, keepdims=True)
    )
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    price = x4[:, :, :, PRICE_INDEX]
    chosen = price[np.arange(n)[:, None], np.arange(tasks)[None, :], y2]
    expected = (probabilities * price).sum(axis=2)
    score = (chosen - expected).sum(axis=1)
    information = ((probabilities * price**2).sum(axis=2) - expected**2).sum(axis=1)
    return float(np.sum(score**2) / np.clip(np.sum(information), 1e-12, None))


def fit_and_score(x: np.ndarray, z: np.ndarray, y: np.ndarray,
                  train_ids: np.ndarray, test_ids: np.ndarray,
                  tasks: int, structured: bool) -> float:
    features = feature_matrix(x, z, structured=structured)
    train = np.concatenate([features[train_ids * tasks + q] for q in range(tasks)])
    y_train = np.concatenate([y[train_ids, q] for q in range(tasks)])
    beta = fit_mnl(train, y_train)
    test_rows = (test_ids[:, None] * tasks + np.arange(tasks)[None, :]).reshape(-1)
    test_x = features[test_rows].reshape(len(test_ids), tasks, features.shape[1], features.shape[2])
    return score_overdispersion(test_x, y[test_ids], beta)


def run(reps: int = 30, n: int = 400, tasks: int = 12,
        out: str = "results/heterogeneity_gate_30rep.csv") -> None:
    rows = []
    for ci, condition in enumerate(CONDITIONS):
        for rep in range(reps):
            seed = 20261004 + ci * 1000 + rep
            x, z, choices, _ = make_data(seed, n, tasks, condition)
            respondent_ids = np.arange(n)
            rng = np.random.default_rng(seed + 77)
            rng.shuffle(respondent_ids)
            cut = int(0.8 * n)
            train_ids, test_ids = respondent_ids[:cut], respondent_ids[cut:]
            q_additive = fit_and_score(x, z, choices, train_ids, test_ids, tasks, structured=False)
            q_structured = fit_and_score(x, z, choices, train_ids, test_ids, tasks, structured=True)
            rows.append({
                "condition": condition.name,
                "replication": rep,
                "q_additive": q_additive,
                "q_structured": q_structured,
                "true_heterogeneity": int(condition.heterogeneity),
            })
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote heterogeneity-gate diagnostics to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--out", default="results/heterogeneity_gate_30rep.csv")
    args = parser.parse_args()
    run(args.reps, args.n, args.tasks, args.out)
