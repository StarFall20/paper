"""Fit and summarize the candidate Swissmetro MNL on public data.

This is a compact descriptive run used by the external-validation record. It
keeps respondents intact in the two-fold score and does not claim a randomized
fibre test.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

from observational_equivalence_test import design, fit_mnl, predict, read_swissmetro, rows_to_arrays


def valid_rows(path):
    raw = read_swissmetro(path)
    kept = [r for r in raw if r["choice"] >= 0 and r["avail"][r["choice"]] > 0]
    return raw, kept


def score(X, y, avail, beta):
    p = predict(X, avail, beta)
    idx = np.arange(len(y))
    return float(np.mean(np.log(p[idx, y] + 1e-300))), float(np.mean(p.argmax(axis=1) == y))


def run(path, out):
    raw, rows = valid_rows(path)
    ids, y, x, avail = rows_to_arrays(rows)
    X = design(x)
    beta = fit_mnl(X, y, avail)
    ll, acc = score(X, y, avail, beta)

    rng = np.random.default_rng(20261007)
    respondent_ids = np.unique(ids).copy()
    rng.shuffle(respondent_ids)
    folds = np.array_split(respondent_ids, 2)
    oof_ll, oof_acc = [], []
    for test_ids in folds:
        test = np.isin(ids, test_ids)
        train = ~test
        held_beta = fit_mnl(X[train], y[train], avail[train])
        fold_ll, fold_acc = score(X[test], y[test], avail[test], held_beta)
        oof_ll.append(fold_ll); oof_acc.append(fold_acc)

    shares = np.bincount(y, minlength=3) / len(y)
    fields = {
        "raw_rows": len(raw),
        "retained_rows": len(rows),
        "dropped_missing_choice": len(raw) - len(rows),
        "respondents": len(np.unique(ids)),
        "choice_share_train": shares[0],
        "choice_share_sm": shares[1],
        "choice_share_car": shares[2],
        "beta_train_asc": beta[0],
        "beta_car_asc": beta[1],
        "beta_time": beta[2],
        "beta_cost": beta[3],
        "in_sample_log_score": ll,
        "in_sample_accuracy": acc,
        "two_fold_oof_log_score": float(np.mean(oof_ll)),
        "two_fold_oof_accuracy": float(np.mean(oof_acc)),
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader(); writer.writerow(fields)
    print(fields)
    print(f"wrote model summary to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--out", default="results/swissmetro_model_summary.csv")
    args = parser.parse_args()
    run(args.path, args.out)
