"""Negative-control benchmark for Behavioral Metamorphic Specification Testing.

The benchmark separates a candidate-basis departure from two process
departures that can also break a candidate-preserving relation: a task-specific
error scale and a side/order effect. A rejection in those controls is
expected. The purpose is to check that BMST is reported as a relation test,
not as an automatic nonlinear-utility classifier.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from exact_paired_task_benchmark import design, fit_mnl, softmax
from randomization_paired_task_test import randomization_pvalue


CONDITIONS = ("null", "nonlinear", "random_taste", "scale_drift", "order_effect")


def generate(seed: int, respondents: int = 300, nuisance_tasks: int = 4,
             condition: str = "null"):
    rng = np.random.default_rng(seed)
    base_beta = np.array([-0.35, -0.18, -1.15, -0.95])
    xs, ys, ids, pair_ids, pair_side = [], [], [], [], []
    for rid in range(respondents):
        random_cost = rng.normal(0.0, 1.0) if condition == "random_taste" else 0.0
        base = np.array([[50.0, 40.0], [70.0, 50.0], [90.0, 60.0]])
        base += rng.normal(0.0, 2.0, size=base.shape)
        tasks = [(0, base), (1, base + np.array([20.0, 50.0])[None, :])]
        tasks += [(-1, rng.uniform(40.0, 120.0, size=(3, 2)))
                  for _ in range(nuisance_tasks)]
        rng.shuffle(tasks)
        local_pair = {}
        for side, xx in tasks:
            utility = np.array([base_beta[0], 0.0, base_beta[1]])
            utility += base_beta[2:] @ (xx.T / 100.0)
            if condition == "nonlinear":
                utility += 6.0 * (xx[:, 0] / 100.0) ** 2
            if condition == "random_taste":
                utility += random_cost * (xx[:, 1] / 100.0)
            if condition == "order_effect" and side == 1:
                utility[0] += 0.75
            scale = 1.75 if condition == "scale_drift" and side == 1 else 1.0
            p = softmax((utility * scale)[None, :])[0]
            row = len(xs)
            xs.append(xx)
            ys.append(int(rng.choice(3, p=p)))
            ids.append(rid)
            pair_ids.append(-1)
            pair_side.append(-1)
            if side >= 0:
                local_pair[side] = row
        pair_ids[local_pair[0]] = rid
        pair_ids[local_pair[1]] = rid
        pair_side[local_pair[0]] = 0
        pair_side[local_pair[1]] = 1
    return (np.asarray(ids), np.asarray(ys), np.asarray(xs),
            np.asarray(pair_ids), np.asarray(pair_side))


def evaluate(data, seed: int, randomization_reps: int = 499):
    ids, y, x, pair_ids, pair_side = data
    unique = np.unique(ids)
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    contrasts, clusters = [], []
    for test_ids in np.array_split(shuffled, 2):
        test_mask = np.isin(ids, test_ids)
        train_mask = ~test_mask
        beta = fit_mnl(design(x[train_mask], None), y[train_mask])
        probs = softmax(np.einsum("njp,p->nj", design(x[test_mask], None), beta))
        rows = np.flatnonzero(test_mask)
        local = {row: i for i, row in enumerate(rows)}
        by_pair = {}
        for row in rows:
            pid = int(pair_ids[row])
            if pid >= 0:
                by_pair.setdefault(pid, {})[int(pair_side[row])] = row
        for pair_rows in by_pair.values():
            if 0 not in pair_rows or 1 not in pair_rows:
                continue
            left, right = pair_rows[0], pair_rows[1]
            observed = np.eye(3)[y[left]] - np.eye(3)[y[right]]
            expected = probs[local[left]] - probs[local[right]]
            contrasts.append(observed - expected)
            clusters.append(int(ids[left]))
    pvalue = randomization_pvalue(
        np.asarray(contrasts), np.asarray(clusters),
        randomization_reps, seed + 17,
    )
    return pvalue, len(contrasts)


def run(reps: int = 100, respondents: int = 300, randomization_reps: int = 499,
        out: str = "results/bmst_negative_controls.csv"):
    rows = []
    for condition in CONDITIONS:
        for rep in range(reps):
            pvalue, pairs = evaluate(
                generate(20261020 + rep, respondents, condition=condition),
                20400000 + rep, randomization_reps,
            )
            rows.append({
                "condition": condition,
                "rep": rep,
                "pairs": pairs,
                "pvalue": pvalue,
                "reject": int(pvalue < 0.05),
            })
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        print(condition, "rejection_rate",
              round(float(np.mean([row["reject"] for row in subset])), 3))
    print(f"wrote {len(rows)} BMST negative-control rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--respondents", type=int, default=300)
    parser.add_argument("--randomization-reps", type=int, default=499)
    parser.add_argument("--out", default="results/bmst_negative_controls.csv")
    run(**vars(parser.parse_args()))
