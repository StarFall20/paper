"""External diagnostic audit on the public Swissmetro DCE.

This is an out-of-sample diagnostic check, not a claim that the public data
identify a data-generating mechanism.  Respondents are split into development,
validation, and test blocks.  The validation block compares an additive MNL,
one predeclared observable expansion (alternative-specific time/cost), and a
flexible classifier.  A respondent-level score-overdispersion statistic is
reported as a separate heterogeneity signal.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from observational_equivalence_test import fit_mnl, predict, read_swissmetro, rows_to_arrays
from swissmetro_specification_benchmark import make_design


def respondent_split(ids: np.ndarray, seed: int):
    unique = np.unique(ids).copy()
    np.random.default_rng(seed).shuffle(unique)
    n = len(unique)
    return unique[: int(.60 * n)], unique[int(.60 * n): int(.80 * n)], unique[int(.80 * n):]


def logloss(X, y, avail, beta):
    p = predict(X, avail, beta)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def accuracy(X, y, avail, beta):
    p = predict(X, avail, beta)
    return float(np.mean(np.argmax(p, axis=1) == y))


def score_overdispersion(ids, X, y, avail, beta):
    p = predict(X, avail, beta)
    chosen = X[np.arange(len(y)), y]
    expected = np.einsum("nj,njp->np", p, X)
    score = chosen - expected
    unique = np.unique(ids)
    respondent_scores = np.stack([score[ids == rid].sum(axis=0) for rid in unique])
    information = np.einsum("nj,njp,njq->npq", p, X, X)
    respondent_info = np.stack([information[ids == rid].sum(axis=0) for rid in unique])
    numerator = np.sum(respondent_scores ** 2)
    denominator = np.trace(np.sum(respondent_info, axis=0))
    return float(numerator / np.clip(denominator, 1e-12, None))


def flexible_fit(raw, y, train):
    model = HistGradientBoostingClassifier(
        max_iter=200, max_leaf_nodes=15, learning_rate=.05,
        l2_regularization=1e-3, random_state=20261007,
    )
    model.fit(raw[train], y[train])
    return model


def flexible_metrics(model, raw, y, rows):
    p = np.zeros((len(rows), 3))
    pred = model.predict_proba(raw[rows])
    p[:, model.classes_.astype(int)] = pred
    return (float(-np.log(np.clip(p[np.arange(len(rows)), y[rows]], 1e-12, 1.0)).mean()),
            float(np.mean(np.argmax(p, axis=1) == y[rows])))


def run(path: str, out: str, seed: int = 20261007):
    rows = read_swissmetro(path, dce_only=True)
    ids, y, x, avail = rows_to_arrays(rows)
    train_ids, val_ids, test_ids = respondent_split(ids, seed)
    train = np.isin(ids, train_ids)
    val = np.isin(ids, val_ids)
    test = np.isin(ids, test_ids)
    base = make_design(x, "base")
    structured = make_design(x, "alt_specific")
    raw = np.concatenate([x.reshape(len(x), -1), avail], axis=1)
    base_beta = fit_mnl(base[train], y[train], avail[train])
    structured_beta = fit_mnl(structured[train], y[train], avail[train])
    flexible = flexible_fit(raw, y, train)
    flex_val_loss, flex_val_acc = flexible_metrics(flexible, raw, y, np.flatnonzero(val))
    flex_test_loss, flex_test_acc = flexible_metrics(flexible, raw, y, np.flatnonzero(test))
    record = {
        "dataset": "Swissmetro_DCE_purpose_1_3",
        "rows": len(y), "respondents": len(np.unique(ids)),
        "train_respondents": len(train_ids), "validation_respondents": len(val_ids),
        "test_respondents": len(test_ids),
        "base_validation_logloss": logloss(base[val], y[val], avail[val], base_beta),
        "structured_validation_logloss": logloss(structured[val], y[val], avail[val], structured_beta),
        "flexible_validation_logloss": flex_val_loss,
        "base_test_logloss": logloss(base[test], y[test], avail[test], base_beta),
        "structured_test_logloss": logloss(structured[test], y[test], avail[test], structured_beta),
        "flexible_test_logloss": flex_test_loss,
        "base_validation_accuracy": accuracy(base[val], y[val], avail[val], base_beta),
        "structured_validation_accuracy": accuracy(structured[val], y[val], avail[val], structured_beta),
        "flexible_validation_accuracy": flex_val_acc,
        "base_test_accuracy": accuracy(base[test], y[test], avail[test], base_beta),
        "structured_test_accuracy": accuracy(structured[test], y[test], avail[test], structured_beta),
        "flexible_test_accuracy": flex_test_acc,
        "structured_gain_validation": logloss(base[val], y[val], avail[val], base_beta)
        - logloss(structured[val], y[val], avail[val], structured_beta),
        "flexible_gain_validation": logloss(base[val], y[val], avail[val], base_beta)
        - flex_val_loss,
        "base_score_overdispersion_validation": score_overdispersion(
            ids[val], base[val], y[val], avail[val], base_beta),
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        writer.writeheader(); writer.writerow(record)
    print("rows", len(y), "respondents", len(np.unique(ids)))
    print("validation gains", round(record["structured_gain_validation"], 5),
          round(record["flexible_gain_validation"], 5),
          "score overdispersion", round(record["base_score_overdispersion_validation"], 4))
    print("test logloss", round(record["base_test_logloss"], 5),
          round(record["structured_test_logloss"], 5), round(record["flexible_test_logloss"], 5))
    print("wrote", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--out", default="results/swissmetro_recoverability_audit.csv")
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    run(args.path, args.out, args.seed)
