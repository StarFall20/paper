"""Prototype policy-equivalence audit for the JOCM revision.

The audit asks whether models that are predictively tied on grouped holdout
choices imply the same policy contrast. It is intentionally independent of the
Random Forest and process-gate prototypes: candidate specifications are fixed
before fitting and the output is a policy-equivalence radius.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_simulation import (  # noqa: E402
    CONDITIONS,
    N_ALTERNATIVES,
    OPT_OUT_INDEX,
    PRICE_INDEX,
    feature_matrix,
    fit_mnl,
    make_data,
    softmax,
)


MODEL_COLUMNS = {
    "base": tuple(range(6)),
    "quality_sq": tuple(list(range(6)) + [6]),
    "price_sq": tuple(list(range(6)) + [7]),
    "price_hinge": tuple(list(range(6)) + [8]),
    "quality_support": tuple(list(range(6)) + [9]),
    "evidence_digital": tuple(list(range(6)) + [10]),
    "price_income": tuple(list(range(6)) + [11]),
    "all": tuple(range(12)),
}


def _true_v(x, z, condition):
    """Return deterministic systematic utility for non-heterogeneity DGPs."""
    v = np.einsum("ntjp,p->ntj", x, np.array([0.55, 0.35, 0.45, 0.25, -0.85]))
    v[:, :, OPT_OUT_INDEX] += -0.35
    if condition.nonlinear:
        v += 0.55 * x[:, :, :, 0] ** 2 - 0.30 * x[:, :, :, PRICE_INDEX] ** 2
    if condition.threshold:
        v += -0.90 * np.maximum(x[:, :, :, PRICE_INDEX] - 1.25, 0.0)
    if condition.interaction:
        v += 0.75 * x[:, :, :, 0] * x[:, :, :, 1]
        v += 0.55 * x[:, :, :, 2] * z[:, None, None, 1]
    v[:, :, OPT_OUT_INDEX] += 0.15 * z[:, None, 0]
    return v


def _probabilities(x, z, beta, columns):
    xx = feature_matrix(x, z, structured=True)[:, :, columns]
    return softmax(np.einsum("tjp,p->tj", xx, beta)).reshape(x.shape[0], x.shape[1], N_ALTERNATIVES)


def _fit_and_score(x, z, y, ids, columns):
    tasks = x.shape[1]
    xx = feature_matrix(x, z, structured=True)[:, :, columns]
    rows = np.concatenate([xx[ids * tasks + q] for q in range(tasks)])
    yy = np.concatenate([y[ids, q] for q in range(tasks)])
    beta = fit_mnl(rows, yy, steps=420, lr=0.08)
    pr = softmax(np.einsum("tjp,p->tj", rows, beta))
    logloss = float(-np.log(np.clip(pr[np.arange(len(yy)), yy], 1e-12, 1.0)).mean())
    return beta, logloss


def _policy_effect(x, z, beta, columns, delta=0.75):
    base = _probabilities(x, z, beta, columns)
    cf = x.copy()
    cf[:, :, :2, PRICE_INDEX] += delta
    shifted = _probabilities(cf, z, beta, columns)
    # The estimand is the change in the market share of product 0.
    return float((shifted[:, :, 0] - base[:, :, 0]).mean())


def run(reps=30, n=400, tasks=12, epsilon=0.01, delta=0.75,
        out="results/policy_equivalence_prototype.csv"):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    rows = []
    # Heterogeneity is excluded here because make_data intentionally does not
    # expose the latent random coefficient. A separate heterogeneity extension
    # is required before this audit is used for that mechanism.
    conditions = [c for c in CONDITIONS if not c.heterogeneity]
    for ci, condition in enumerate(conditions):
        for rep in range(reps):
            seed = 20261004 + ci * 10000 + rep
            x, z, y, _ = make_data(seed, n=n, tasks=tasks, condition=condition)
            rng = np.random.default_rng(seed + 91)
            ids = np.arange(n); rng.shuffle(ids)
            cut = int(0.8 * n)
            train_ids, test_ids = ids[:cut], ids[cut:]
            fitted = {}
            for name, columns in MODEL_COLUMNS.items():
                beta, loss = _fit_and_score(x, z, y, train_ids, columns)
                # Validation loss is computed on respondent-held-out tasks.
                xx = feature_matrix(x, z, structured=True)[:, :, columns]
                rows_test = np.concatenate([xx[test_ids * tasks + q] for q in range(tasks)])
                yy_test = np.concatenate([y[test_ids, q] for q in range(tasks)])
                pr_test = softmax(np.einsum("tjp,p->tj", rows_test, beta))
                val_loss = float(-np.log(np.clip(pr_test[np.arange(len(yy_test)), yy_test], 1e-12, 1.0)).mean())
                policy = _policy_effect(x[test_ids], z[test_ids], beta, columns, delta=delta)
                fitted[name] = {"beta": beta, "loss": val_loss, "policy": policy}
            best = min(v["loss"] for v in fitted.values())
            admissible = [name for name, v in fitted.items() if v["loss"] <= best + epsilon]
            radius = max(fitted[name]["policy"] for name in admissible) - min(
                fitted[name]["policy"] for name in admissible
            )
            forced = min(admissible, key=lambda name: fitted[name]["loss"])
            true_v = _true_v(x[test_ids], z[test_ids], condition)
            xcf = x[test_ids].copy(); xcf[:, :, :2, PRICE_INDEX] += delta
            true_v_cf = _true_v(xcf, z[test_ids], condition)
            true_effect = float((softmax(true_v_cf.reshape(-1, N_ALTERNATIVES))[:, 0].mean()
                                 - softmax(true_v.reshape(-1, N_ALTERNATIVES))[:, 0].mean()))
            rows.append({
                "condition": condition.name,
                "rep": rep,
                "n": n,
                "tasks": tasks,
                "epsilon": epsilon,
                "delta": delta,
                "best_loss": best,
                "admissible_count": len(admissible),
                "admissible_models": "+".join(admissible),
                "policy_radius": radius,
                "true_policy_effect": true_effect,
                "forced_policy_error": abs(fitted[forced]["policy"] - true_effect),
                "interval_covers": int(min(fitted[name]["policy"] for name in admissible) <= true_effect <= max(fitted[name]["policy"] for name in admissible)),
            })
    fields = list(rows[0])
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} policy-equivalence rows to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--tasks", type=int, default=12)
    ap.add_argument("--epsilon", type=float, default=0.01)
    ap.add_argument("--delta", type=float, default=0.75)
    ap.add_argument("--out", default="results/policy_equivalence_prototype.csv")
    args = ap.parse_args()
    run(**vars(args))
