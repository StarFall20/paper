"""Exploratory benchmark for transportable utility specification.

The object is an environment-invariant utility core.  Candidate terms are
screened on several source environments using leave-one-environment-out loss
and a penalty for instability in environment-specific utility coefficients.
The target environment reverses a source-only proxy correlation.  This is a
method-transfer audit from invariant risk minimisation, not a new flexible
choice model.  The output is exploratory: the coefficient-dispersion penalty
is not scale-normalized and must be frozen from source-only evidence before
any confirmatory interpretation.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np


TERMS = ("quality", "price", "proxy", "quality_sq")
SUBSETS = (
    ("quality", "price"),
    ("quality", "price", "proxy"),
    ("quality", "price", "quality_sq"),
    ("quality", "price", "proxy", "quality_sq"),
)


def softmax(u):
    z = u - np.max(u, axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def make_design(x, subset):
    """Alternative-specific terms plus two alternative constants."""
    blocks = []
    for term in subset:
        if term == "quality":
            blocks.append(x[:, :, :, 0:1])
        elif term == "price":
            blocks.append(x[:, :, :, 1:2])
        elif term == "proxy":
            blocks.append(x[:, :, :, 2:3])
        elif term == "quality_sq":
            blocks.append(x[:, :, :, 0:1] ** 2)
        else:
            raise ValueError(term)
    attrs = np.concatenate(blocks, axis=3)
    n, t, j, _ = attrs.shape
    asc = np.broadcast_to(np.eye(j)[None, None, :, :j - 1], (n, t, j, j - 1))
    return np.concatenate([attrs, asc], axis=3)


def fit_mnl(x, y, subset, steps=260, lr=0.09, l2=1e-3):
    X = make_design(x, subset)
    p = X.shape[3]
    beta = np.zeros(p)
    for _ in range(steps):
        pr = softmax(np.einsum("ntjp,p->ntj", X, beta).reshape(-1, X.shape[2]))
        pr = pr.reshape(X.shape[0], X.shape[1], X.shape[2])
        target = np.eye(X.shape[2])[y]
        grad = np.einsum("ntj,ntjp->p", pr - target, X) / (X.shape[0] * X.shape[1])
        beta -= lr * (grad + l2 * beta)
    return beta


def predict(x, subset, beta):
    X = make_design(x, subset)
    return softmax(np.einsum("ntjp,p->ntj", X, beta).reshape(-1, X.shape[2])).reshape(X.shape[0], X.shape[1], X.shape[2])


def generate(seed, environment, respondents=180, tasks=8):
    rng = np.random.default_rng(seed)
    n, t, j = respondents, tasks, 3
    latent_q = rng.normal(0.0, 1.0, (n, t, j))
    q_obs = latent_q + rng.normal(0.0, 0.75, (n, t, j))
    price = rng.normal(0.0, 1.0, (n, t, j))
    rho = (0.9, 0.65, 0.4, -0.9)[environment]
    proxy = rho * latent_q + rng.normal(0.0, 0.45, (n, t, j))
    x = np.stack([q_obs, price, proxy], axis=3)
    asc = np.array([0.0, -0.12, -0.20])
    v = 0.8 * latent_q - 0.9 * price + asc
    p = softmax(v.reshape(-1, j)).reshape(n, t, j)
    y = np.array([rng.choice(j, p=row) for row in p.reshape(-1, j)]).reshape(n, t)
    return x, y, v


def logloss(pr, y):
    n, t, j = pr.shape
    return float(-np.log(np.clip(pr[np.arange(n)[:, None], np.arange(t)[None, :], y], 1e-12, 1.0)).mean())


def evaluate_subset(source, target, subset, invariant_lambda):
    env_models = []
    env_losses = []
    for x, y, _ in source:
        beta = fit_mnl(x, y, subset)
        env_models.append(beta)
        env_losses.append(logloss(predict(x, subset, beta), y))
    heldout_losses = []
    for holdout in range(len(source)):
        x_train = np.concatenate([source[i][0] for i in range(len(source)) if i != holdout], axis=0)
        y_train = np.concatenate([source[i][1] for i in range(len(source)) if i != holdout], axis=0)
        beta = fit_mnl(x_train, y_train, subset)
        heldout_losses.append(logloss(predict(source[holdout][0], subset, beta), source[holdout][1]))
    coef_gap = float(np.mean(np.std(np.asarray(env_models), axis=0)))
    cv_loss = float(np.mean(heldout_losses))
    score = cv_loss + invariant_lambda * coef_gap
    beta_pool = fit_mnl(np.concatenate([s[0] for s in source], axis=0), np.concatenate([s[1] for s in source], axis=0), subset)
    target_x, target_y, target_v = target
    target_pr = predict(target_x, subset, beta_pool)
    policy_x = np.array(target_x, copy=True)
    policy_x[:, :, 0, 1] -= 0.75
    policy_pr = predict(policy_x, subset, beta_pool)
    true_pr = softmax(target_v.reshape(-1, target_v.shape[2])).reshape(target_v.shape)
    policy_v = np.array(target_v, copy=True)
    policy_v[:, :, 0] += 0.9 * 0.75
    true_policy_pr = softmax(policy_v.reshape(-1, policy_v.shape[2])).reshape(policy_v.shape)
    predicted_effect = float(np.mean(policy_pr[:, :, 0]) - np.mean(target_pr[:, :, 0]))
    true_effect = float(np.mean(true_policy_pr[:, :, 0]) - np.mean(true_pr[:, :, 0]))
    return {
        "subset": "+".join(subset),
        "cv_loss": cv_loss,
        "coef_gap": coef_gap,
        "score": score,
        "target_loss": logloss(target_pr, target_y),
        "target_share0": float(np.mean(target_pr[:, :, 0])),
        "true_share0": float(np.mean(true_pr[:, :, 0])),
        "policy_effect": predicted_effect,
        "true_policy_effect": true_effect,
        "policy_effect_error": abs(predicted_effect - true_effect),
    }


def run(reps=30, respondents=180, tasks=8, invariant_lambda=0.15, rep_start=0,
        out="results/transportable_utility_benchmark.csv"):
    rows = []
    for rep in range(rep_start, rep_start + reps):
        source = [generate(20261201 + rep * 10 + e, e, respondents, tasks) for e in range(3)]
        target = generate(20261201 + rep * 10 + 3, 3, respondents, tasks)
        metrics = [evaluate_subset(source, target, subset, invariant_lambda) for subset in SUBSETS]
        standard = min(metrics, key=lambda m: m["cv_loss"])
        invariant = min(metrics, key=lambda m: m["score"])
        for method, selected in (("loeo_cv", standard), ("invariant_core", invariant)):
            row = {"rep": rep, "method": method,
                   "invariant_lambda": invariant_lambda, **selected}
            rows.append(row)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    for method in ("loeo_cv", "invariant_core"):
        subset = [r for r in rows if r["method"] == method]
        print(method, "target_loss", round(float(np.mean([r["target_loss"] for r in subset])), 3),
              "target_share_gap", round(float(np.mean(np.abs(np.asarray([r["target_share0"] for r in subset]) - np.asarray([r["true_share0"] for r in subset])))), 3),
              "policy_effect_error", round(float(np.mean([r["policy_effect_error"] for r in subset])), 3),
              "proxy_selected", round(float(np.mean(["proxy" in r["subset"] for r in subset])), 3))
    print(f"wrote {len(rows)} transportable-utility rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--respondents", type=int, default=180)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--invariant-lambda", type=float, default=0.15)
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--out", default="results/transportable_utility_benchmark.csv")
    run(**vars(parser.parse_args()))
