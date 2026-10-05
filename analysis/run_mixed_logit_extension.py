"""Random-coefficient price extension for the controlled choice benchmark.

The extension keeps the grouped respondent split used by the primary scaffold.
It estimates a single random price coefficient with simulated likelihood, using
fixed standard-normal draws within each respondent. Results are an extension
check until the full Mixed Logit specification, WTP recovery, and convergence
report are added.
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_simulation import CONDITIONS, feature_matrix, make_data


def _probabilities(X, beta, sigma, draws):
    # X: respondent x task x alternative x feature
    price = X[:, :, :, 4]
    base = np.einsum("ntjp,p->ntj", X, beta)
    util = base[:, None, :, :] + sigma * draws[:, :, None, None] * price[:, None, :, :]
    util = util - util.max(axis=3, keepdims=True)
    expu = np.exp(util)
    return expu / expu.sum(axis=3, keepdims=True)


def fit_mixed_logit(X, y, steps=90, draws=20, lr=0.035, seed=0):
    """Fit beta and a positive random price scale by EM-style gradients."""
    n, tasks, _, p = X.shape
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, draws))
    beta = np.zeros(p)
    log_sigma = np.log(0.35)
    target_x = X[np.arange(n)[:, None], np.arange(tasks)[None, :], y]
    target_price = target_x[:, :, 4]
    for _ in range(steps):
        sigma = float(np.exp(np.clip(log_sigma, -4.0, 2.0)))
        prob = _probabilities(X, beta, sigma, q)
        chosen = np.take_along_axis(prob, y[:, None, :, None].repeat(draws, axis=1), axis=3)[:, :, :, 0]
        log_joint = np.log(np.clip(chosen, 1e-12, 1.0)).sum(axis=2)
        log_joint -= log_joint.max(axis=1, keepdims=True)
        weights = np.exp(log_joint)
        weights /= weights.sum(axis=1, keepdims=True)

        expected_x = np.einsum("ndtj,ntjp->ndtp", prob, X)
        score_beta = target_x[:, None, :, :] - expected_x
        score_beta = score_beta.sum(axis=2)
        expected_price = np.einsum("ndtj,ntj->ndt", prob, X[:, :, :, 4])
        score_sigma = (target_price[:, None, :] - expected_price).sum(axis=2)
        grad_beta = np.einsum("nd,ndp->p", weights, score_beta) / n
        grad_log_sigma = np.einsum("nd,nd->", weights * q, score_sigma) * sigma / n
        beta += lr * np.clip(grad_beta, -5.0, 5.0)
        log_sigma += lr * float(np.clip(grad_log_sigma, -5.0, 5.0))
    return beta, float(np.exp(np.clip(log_sigma, -4.0, 2.0)))


def score_mixed_logit(X, y, v, beta, sigma, draws=80, seed=0):
    n, tasks, _, _ = X.shape
    q = np.random.default_rng(seed).normal(size=(n, draws))
    prob = _probabilities(X, beta, sigma, q)
    pr = prob.mean(axis=1)
    yflat = y.reshape(-1)
    pflat = pr.reshape(-1, 3)
    vflat = v.reshape(-1, 3)
    pred = pflat.argmax(1)
    acc = float((pred == yflat).mean())
    logloss = float(-np.log(np.clip(pflat[np.arange(len(yflat)), yflat], 1e-12, 1)).mean())
    brier = float(((pflat - np.eye(3)[yflat]) ** 2).sum(1).mean())
    shares = pflat.mean(0)
    observed = np.bincount(yflat, minlength=3) / len(yflat)
    share_rmse = float(np.sqrt(np.mean((shares - observed) ** 2)))
    regret = float(np.mean(vflat.max(1) - vflat[np.arange(len(yflat)), pred]))
    return acc, logloss, brier, share_rmse, regret


def run(reps=3, n=400, tasks=12, out="results/mixed_logit_extension.csv"):
    rows = []
    for ci, cond in enumerate(CONDITIONS):
        for rep in range(reps):
            seed = 20401005 + ci * 1000 + rep
            x, z, choices, v = make_data(seed, n, tasks, cond)
            X = feature_matrix(x, z, structured=False).reshape(n, tasks, 3, -1)
            ids = np.arange(n)
            np.random.default_rng(seed + 7).shuffle(ids)
            cut = int(0.8 * n)
            train_ids, test_ids = ids[:cut], ids[cut:]
            beta, sigma = fit_mixed_logit(X[train_ids], choices[train_ids], seed=seed + 31)
            vals = score_mixed_logit(X[test_ids], choices[test_ids], v[test_ids], beta, sigma, seed=seed + 41)
            rows.append({"condition": cond.name, "replication": rep, "model": "Mixed_Logit_random_price",
                         "random_price_sd": sigma, "accuracy": vals[0], "logloss": vals[1],
                         "brier": vals[2], "share_rmse": vals[3], "decision_regret": vals[4]})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} model-replication rows to {out}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--tasks", type=int, default=12)
    ap.add_argument("--out", default="results/mixed_logit_extension.csv")
    args = ap.parse_args()
    run(args.reps, args.n, args.tasks, args.out)
