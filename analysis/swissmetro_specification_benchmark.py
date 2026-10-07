"""Unified specification benchmark on the canonical Swissmetro DCE sample.

This is a public-data model audit, not a claim of randomized fibre
identification.  Every specification uses the same purpose-1/3 sample,
availability flags, GA-adjusted costs, and respondent-level two-fold split.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from scipy.stats import chi2

from observational_equivalence_test import fit_mnl, predict, read_swissmetro, rows_to_arrays


def make_design(x: np.ndarray, name: str) -> np.ndarray:
    n = len(x)
    t = x[:, :, 0] / 100.0
    c = x[:, :, 1] / 100.0
    if name == "base":
        X = np.zeros((n, 3, 4))
        X[:, 0, 0] = 1.0
        X[:, 2, 1] = 1.0
        X[:, :, 2] = t
        X[:, :, 3] = c
        return X
    if name == "quadratic":
        X = np.zeros((n, 3, 6))
        X[:, 0, 0] = 1.0
        X[:, 2, 1] = 1.0
        X[:, :, 2] = t
        X[:, :, 3] = c
        X[:, :, 4] = t * t
        X[:, :, 5] = (c * c) / 10.0
        return X
    if name == "log":
        X = np.zeros((n, 3, 6))
        X[:, 0, 0] = 1.0
        X[:, 2, 1] = 1.0
        X[:, :, 2] = t
        X[:, :, 3] = c
        X[:, :, 4] = np.log1p(t)
        X[:, :, 5] = np.log1p(c)
        return X
    if name == "alt_specific":
        X = np.zeros((n, 3, 8))
        X[:, 0, 0] = 1.0
        X[:, 2, 1] = 1.0
        for j in range(3):
            X[:, j, 2 + j] = t[:, j]
            X[:, j, 5 + j] = c[:, j]
        return X
    raise ValueError(name)


def loglik(X, y, avail, beta) -> float:
    p = predict(X, avail, beta)
    return float(np.log(p[np.arange(len(y)), y] + 1e-300).sum())


def score(X, y, avail, beta) -> tuple[float, float]:
    p = predict(X, avail, beta)
    return (float(np.mean(np.log(p[np.arange(len(y)), y] + 1e-300))),
            float(np.mean(np.argmax(p, axis=1) == y)))


def run(path: str, out: str) -> None:
    rows = read_swissmetro(path, dce_only=True)
    ids, y, x, avail = rows_to_arrays(rows)
    names = ("base", "quadratic", "log", "alt_specific")
    rng = np.random.default_rng(20261007)
    respondent_ids = np.unique(ids).copy()
    rng.shuffle(respondent_ids)
    folds = np.array_split(respondent_ids, 2)
    records = []
    full = {}
    for name in names:
        X = make_design(x, name)
        beta = fit_mnl(X, y, avail)
        ll = loglik(X, y, avail, beta)
        full[name] = (X, beta, ll)
        fold_ll, fold_acc = [], []
        for test_ids in folds:
            test = np.isin(ids, test_ids)
            train = ~test
            b = fit_mnl(X[train], y[train], avail[train])
            ll_test, acc_test = score(X[test], y[test], avail[test], b)
            fold_ll.append(ll_test); fold_acc.append(acc_test)
        records.append({
            "model": name,
            "n_parameters": int(X.shape[2]),
            "n_rows": int(len(y)),
            "respondents": int(len(np.unique(ids))),
            "loglik": ll,
            "aic": -2.0 * ll + 2.0 * X.shape[2],
            "bic": -2.0 * ll + np.log(len(y)) * X.shape[2],
            "in_sample_log_score": float(np.mean(score(X, y, avail, beta)[0:1])),
            "in_sample_accuracy": score(X, y, avail, beta)[1],
            "two_fold_oof_log_score": float(np.mean(fold_ll)),
            "two_fold_oof_accuracy": float(np.mean(fold_acc)),
            "beta": ";".join(f"{v:.9g}" for v in beta),
        })
    base_ll = full["base"][2]
    for rec in records:
        rec["lr_vs_base"] = float(2.0 * (rec["loglik"] - base_ll))
        rec["lr_df_vs_base"] = int(rec["n_parameters"] - 4)
        rec["lr_p_vs_base"] = (1.0 if rec["lr_df_vs_base"] <= 0 else
                                float(chi2.sf(rec["lr_vs_base"], rec["lr_df_vs_base"])))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    for rec in records:
        print(rec["model"], "ll", round(rec["loglik"], 3),
              "oof_log", round(rec["two_fold_oof_log_score"], 5),
              "oof_acc", round(rec["two_fold_oof_accuracy"], 4),
              "LR_p", f"{rec['lr_p_vs_base']:.4g}")
    print(f"wrote {len(records)} specification rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--out", default="results/swissmetro_specification_benchmark.csv")
    args = parser.parse_args()
    run(args.path, args.out)
