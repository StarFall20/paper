"""External validation on Kenneth Train's public vehicle stated-choice DCE.

The source archive is the public sample accompanying Train's mixed-logit
software.  This script treats it as an implementation check: it compares
predeclared utility representations on respondent-grouped holdouts and audits
whether the task pool contains repeated complete menus.  It does not claim a
randomized candidate-preserving fibre experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from observational_equivalence_test import fit_mnl, predict


def read_train_vehicle(path: str):
    raw = np.loadtxt(path)
    if raw.ndim != 2 or raw.shape[1] != 11:
        raise ValueError(f"expected 11 columns, got {raw.shape}")
    tasks = []
    for task_id in np.unique(raw[:, 1]).astype(int):
        block = raw[raw[:, 1].astype(int) == task_id]
        if len(block) != 3 or int(block[:, 2].sum()) != 1:
            raise ValueError(f"task {task_id} does not have three alternatives and one choice")
        tasks.append({
            "person": int(block[0, 0]),
            "task": task_id,
            "choice": int(np.argmax(block[:, 2])),
            "attributes": block[:, 3:].astype(float),
        })
    return tasks


def arrays(tasks):
    ids = np.asarray([t["person"] for t in tasks], dtype=int)
    y = np.asarray([t["choice"] for t in tasks], dtype=int)
    attrs = np.stack([t["attributes"] for t in tasks])
    avail = np.ones((len(tasks), 3), dtype=float)
    return ids, y, attrs, avail


def design(attrs: np.ndarray, name: str) -> np.ndarray:
    # The source code negates price and operating cost. Scaling is fixed before
    # estimation and is shared by every specification.
    p = -attrs[:, :, 0] / 10000.0
    oc = -attrs[:, :, 1] / 100.0
    rng = attrs[:, :, 2] / 10.0
    ev, hybrid = attrs[:, :, 3], attrs[:, :, 5]
    hp, medhp = attrs[:, :, 6], attrs[:, :, 7]
    base = [p, oc, rng, ev, hybrid, hp, medhp]
    if name == "linear":
        cols = base
    elif name == "quadratic":
        cols = base + [p * p, oc * oc, rng * rng, p * rng, p * ev, p * hybrid]
    elif name == "type_interactions":
        cols = base + [ev * hp, hybrid * hp, ev * rng, hybrid * rng]
    elif name == "full":
        cols = base + [p * p, oc * oc, rng * rng, p * rng, p * ev,
                       p * hybrid, ev * hp, hybrid * hp, ev * rng,
                       hybrid * rng]
    else:
        raise ValueError(name)
    return np.stack(cols, axis=2)


def effective_rank(X):
    """Rank of alternative utility differences, after removing a reference arm."""
    D = (X[:, :-1, :] - X[:, -1:, :]).reshape(-1, X.shape[2])
    return int(np.linalg.matrix_rank(D, tol=1e-9))


def log_score(X, y, avail, beta):
    probs = predict(X, avail, beta)
    return float(np.mean(np.log(probs[np.arange(len(y)), y] + 1e-300)))


def accuracy(X, y, avail, beta):
    probs = predict(X, avail, beta)
    return float(np.mean(np.argmax(probs, axis=1) == y))


def task_signature(attrs):
    # A complete-menu signature uses all observed alternative attributes and is
    # sorted by row position because alternatives are unlabeled in the archive.
    return tuple(np.round(np.asarray(attrs, dtype=float).ravel(), 8))


def run(path: str, out: str, provenance_out: str, seed: int = 20261008) -> None:
    tasks = read_train_vehicle(path)
    ids, y, attrs, avail = arrays(tasks)
    model_names = ("linear", "quadratic", "type_interactions", "full")
    people = np.unique(ids)
    rng = np.random.default_rng(seed)
    shuffled = people.copy(); rng.shuffle(shuffled)
    folds = np.array_split(shuffled, 5)
    records = []
    fits = {}
    for name in model_names:
        X = design(attrs, name)
        beta = fit_mnl(X, y, avail)
        rank = effective_rank(X)
        ll = float(np.sum(np.log(predict(X, avail, beta)[np.arange(len(y)), y] + 1e-300)))
        fits[name] = (X, beta, ll)
        fold_scores, fold_acc = [], []
        for test_people in folds:
            test = np.isin(ids, test_people)
            train = ~test
            b = fit_mnl(X[train], y[train], avail[train])
            fold_scores.append(log_score(X[test], y[test], avail[test], b))
            fold_acc.append(accuracy(X[test], y[test], avail[test], b))
        records.append({
            "model": name,
            "raw_parameters": int(X.shape[2]),
            "effective_rank": rank,
            "n_parameters": rank,
            "n_tasks": int(len(y)),
            "respondents": int(len(people)),
            "loglik": ll,
            "aic": -2 * ll + 2 * rank,
            "bic": -2 * ll + np.log(len(y)) * rank,
            "in_sample_log_score": log_score(X, y, avail, beta),
            "in_sample_accuracy": accuracy(X, y, avail, beta),
            "five_fold_oof_log_score": float(np.mean(fold_scores)),
            "five_fold_oof_accuracy": float(np.mean(fold_acc)),
            "beta": ";".join(f"{v:.9g}" for v in beta),
        })
    base_ll = fits["linear"][2]
    for rec in records:
        rec["lr_vs_linear"] = 2 * (rec["loglik"] - base_ll)
        rec["lr_df_vs_linear"] = rec["effective_rank"] - records[0]["effective_rank"]
        rec["lr_p_vs_linear"] = (1.0 if rec["lr_df_vs_linear"] <= 0 else
                                  float(chi2.sf(rec["lr_vs_linear"], rec["lr_df_vs_linear"])))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)

    signatures = [task_signature(t["attributes"]) for t in tasks]
    unique_signatures = len(set(signatures))
    people_task_counts = np.asarray([np.sum(ids == person) for person in people])
    checksum = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    provenance = [
        f"source_file={path}",
        f"sha256={checksum}",
        f"rows={sum(len(t['attributes']) for t in tasks)}",
        f"tasks={len(tasks)}",
        f"respondents={len(people)}",
        f"alternatives_per_task=3",
        f"unique_complete_menu_signatures={unique_signatures}",
        f"duplicate_complete_menu_tasks={len(tasks) - unique_signatures}",
        f"min_tasks_per_respondent={int(people_task_counts.min())}",
        f"max_tasks_per_respondent={int(people_task_counts.max())}",
        "assignment_note=public observational stated-choice archive; no candidate-preserving randomization",
        f"holdout_note=5-fold respondent-grouped cross-fitting with fixed seed {seed}",
    ]
    with open(provenance_out, "w") as handle:
        handle.write("\n".join(provenance) + "\n")
    for rec in records:
        print(rec["model"], "oof_log", round(rec["five_fold_oof_log_score"], 6),
              "oof_acc", round(rec["five_fold_oof_accuracy"], 4),
              "LR_p", f"{rec['lr_p_vs_linear']:.4g}")
    print("tasks", len(tasks), "respondents", len(people),
          "unique_menus", unique_signatures)
    print(f"wrote {len(records)} model rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--out", default="results/train_vehicle_external_validation.csv")
    parser.add_argument("--provenance-out", default="data/provenance_train_vehicle_2026-10-08.md")
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    run(args.path, args.out, args.provenance_out, seed=args.seed)
