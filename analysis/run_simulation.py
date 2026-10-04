"""Pure-NumPy Monte Carlo benchmark for the JOCM revision.

The script keeps the simulation runnable in a minimal environment. It compares
an additive MNL with a structured MNL that receives prespecified nonlinear and
interaction terms. Optional scikit-learn learners can be added later without
changing the data-generating process or validation code.
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Condition:
    name: str
    nonlinear: bool
    threshold: bool
    interaction: bool
    heterogeneity: bool


CONDITIONS = [
    Condition("additive", False, False, False, False),
    Condition("nonlinear", True, False, False, False),
    Condition("threshold", False, True, False, False),
    Condition("interaction", False, False, True, False),
    Condition("heterogeneity", False, False, False, True),
    Condition("nonlinear_threshold", True, True, False, False),
    Condition("nonlinear_interaction", True, False, True, False),
    Condition("combined", True, True, True, True),
]


def softmax(u):
    z = u - u.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def make_data(seed, n=400, tasks=12, condition=None):
    rng = np.random.default_rng(seed)
    j = 3
    # alternative attributes: quality, support, evidence, data, price
    x = rng.normal(size=(n, tasks, j, 5))
    x[:, :, 2, :] = 0.0  # opt-out has no product attributes
    x[:, :, :, 4] = np.abs(x[:, :, :, 4]) + 0.5
    z = rng.normal(size=(n, 2))  # observed income and digital experience
    z[:, 0] = (z[:, 0] > 0).astype(float)
    z[:, 1] = (z[:, 1] > 0).astype(float)
    base = np.array([0.55, 0.35, 0.45, 0.25, -0.85])
    v = np.einsum("ntjp,p->ntj", x, base)
    v[:, :, 2] += -0.35
    if condition.nonlinear:
        v += 0.55 * x[:, :, :, 0] ** 2 - 0.30 * x[:, :, :, 4] ** 2
    if condition.threshold:
        v += -0.90 * np.maximum(x[:, :, :, 4] - 1.25, 0.0)
    if condition.interaction:
        v += 0.75 * x[:, :, :, 0] * x[:, :, :, 1]
        v += 0.55 * x[:, :, :, 2] * z[:, None, None, 1]
    if condition.heterogeneity:
        # Unobserved random price sensitivity. This is deliberately distinct
        # from the observed covariates used by the interaction condition.
        random_price = rng.normal(0.0, 0.45, size=n)
        v += random_price[:, None, None] * x[:, :, :, 4]
    v[:, :, 2] += 0.15 * z[:, None, 0]
    p = softmax(v.reshape(-1, j)).reshape(n, tasks, j)
    choices = np.array([rng.choice(j, p=pp) for pp in p.reshape(-1, j)]).reshape(n, tasks)
    return x, z, choices, v


def feature_matrix(x, z, structured=False):
    n, t, j, _ = x.shape
    flat_x = x.reshape(n * t, j, 5)
    flat_z = np.repeat(z, t, axis=0)
    optout = np.broadcast_to((np.arange(j) == 2).astype(float)[None, :, None], (n * t, j, 1))
    cols = [flat_x, optout]
    if structured:
        cols += [flat_x[:, :, 0:1] ** 2, flat_x[:, :, 4:5] ** 2]
        cols += [np.maximum(flat_x[:, :, 4:5] - 1.25, 0.0)]
        cols += [flat_x[:, :, 0:1] * flat_x[:, :, 1:2]]
        cols += [flat_x[:, :, 2:3] * flat_z[:, None, 1:2]]
        cols += [flat_x[:, :, 4:5] * flat_z[:, None, 0:1]]
    return np.concatenate(cols, axis=2)


def fit_mnl(X, y, steps=350, lr=0.08, l2=1e-3):
    p = X.shape[2]
    beta = np.zeros(p)
    for _ in range(steps):
        pr = softmax(np.einsum("tjp,p->tj", X, beta))
        target = np.zeros_like(pr)
        target[np.arange(len(y)), y] = 1.0
        grad = np.einsum("tj,tjp->p", pr - target, X) / len(y) + l2 * beta
        beta -= lr * grad
    return beta


def score(X, y, v_true, beta):
    pr = softmax(np.einsum("tjp,p->tj", X, beta))
    pred = pr.argmax(1)
    acc = float((pred == y).mean())
    logloss = float(-np.log(np.clip(pr[np.arange(len(y)), y], 1e-12, 1)).mean())
    brier = float(((pr - np.eye(3)[y]) ** 2).sum(1).mean())
    shares = pr.mean(0)
    observed = np.bincount(y, minlength=3) / len(y)
    share_rmse = float(np.sqrt(np.mean((shares - observed) ** 2)))
    regret = float(np.mean(v_true.max(1) - v_true[np.arange(len(y)), pred]))
    return acc, logloss, brier, share_rmse, regret


def choose_candidate_terms(xx, choices, train_ids, tasks, seed):
    """Nested forward selection using respondent-level validation."""
    rng = np.random.default_rng(seed)
    shuffled = np.array(train_ids, copy=True)
    rng.shuffle(shuffled)
    cut = max(1, int(0.75 * len(shuffled)))
    inner, valid = shuffled[:cut], shuffled[cut:]
    groups = [[6], [7], [8], [9], [10], [11]]
    names = ["quality_sq", "price_sq", "price_hinge", "quality_support", "evidence_digital", "price_income"]
    selected = list(range(6))
    remaining = list(range(len(groups)))

    def stack(ids, cols):
        mat = xx[:, :, cols]
        tr = np.concatenate([mat[ids * tasks + q] for q in range(tasks)])
        yy = np.concatenate([choices[ids, q] for q in range(tasks)])
        return tr, yy

    tr0, y0 = stack(inner, selected)
    beta0 = fit_mnl(tr0, y0)
    va0, yv = stack(valid, selected)
    best_loss = score(va0, yv, np.zeros((len(yv), 3)), beta0)[1]
    while remaining:
        trials = []
        for gi in remaining:
            cols = selected + groups[gi]
            tr, yy = stack(inner, cols)
            b = fit_mnl(tr, yy, steps=220)
            va, _ = stack(valid, cols)
            loss = score(va, yv, np.zeros((len(yv), 3)), b)[1]
            trials.append((loss, gi))
        loss, gi = min(trials)
        if loss + 1e-4 < best_loss:
            selected += groups[gi]
            remaining.remove(gi)
            best_loss = loss
        else:
            break
    return selected, "+".join(["base"] + [names[i] for i, g in enumerate(groups) if all(c in selected for c in g)])


def run(reps=24, n=400, tasks=12, out="results/simulation_results.csv"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for ci, cond in enumerate(CONDITIONS):
        for rep in range(reps):
            seed = 20261004 + ci * 1000 + rep
            x, z, choices, v = make_data(seed, n, tasks, cond)
            idx = np.arange(n)
            rng = np.random.default_rng(seed + 77)
            rng.shuffle(idx)
            cut = int(0.8 * n)
            train_ids, test_ids = idx[:cut], idx[cut:]
            for model, structured in [("MNL", False), ("Structured_MNL", True), ("ML_assisted_spec", True)]:
                xx = feature_matrix(x, z, structured)
                selected = list(range(xx.shape[2]))
                selected_terms = "base+all_candidates" if model == "Structured_MNL" else "base"
                if model == "ML_assisted_spec":
                    selected, selected_terms = choose_candidate_terms(xx, choices, train_ids, tasks, seed + 101)
                xx = xx[:, :, selected]
                train = np.concatenate([xx[train_ids * tasks + q] for q in range(tasks)])
                ytrain = np.concatenate([choices[train_ids, q] for q in range(tasks)])
                test = np.concatenate([xx[test_ids * tasks + q] for q in range(tasks)])
                ytest = np.concatenate([choices[test_ids, q] for q in range(tasks)])
                vtest = np.concatenate([v[test_ids, q] for q in range(tasks)])
                beta = fit_mnl(train, ytrain)
                vals = score(test, ytest, vtest, beta)
                rows.append({"condition": cond.name, "replication": rep, "model": model,
                             "accuracy": vals[0], "logloss": vals[1], "brier": vals[2],
                             "share_rmse": vals[3], "decision_regret": vals[4],
                             "selected_terms": selected_terms})
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--tasks", type=int, default=12)
    ap.add_argument("--out", default="results/simulation_results.csv")
    args = ap.parse_args()
    rows = run(args.reps, args.n, args.tasks, args.out)
    print(f"wrote {len(rows)} model-replication rows to {args.out}")
