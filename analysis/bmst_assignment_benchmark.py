"""Between-respondent BMST benchmark with one member of each pair observed.

The recommended supplement assigns each respondent one common orientation of
the focal task pairs. The respondent never sees both members of a pair, so
carryover cannot create the relation violation. A respondent-cluster sign flip
then follows from the randomized orientation under the candidate null.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from exact_paired_task_benchmark import design, fit_mnl, softmax
from randomization_paired_task_test import randomization_pvalue


CONDITIONS = ("additive", "nonlinear", "random_price", "interaction")


def generate(seed: int, respondents: int = 300, nuisance_tasks: int = 4,
             condition: str = "additive"):
    rng = np.random.default_rng(seed)
    beta = np.array([-0.35, -0.18, -1.15, -0.95])
    xs, ys, ids, orientations, focal = [], [], [], [], []
    for rid in range(respondents):
        orientation = int(rng.integers(0, 2))
        random_cost = rng.normal() if condition == "random_price" else 0.0
        tasks = []
        for typ, shift in (
            ("time", np.array([20.0, 0.0])),
            ("cost", np.array([0.0, 60.0])),
            ("joint", np.array([20.0, 50.0])),
        ):
            base = np.array([[50.0, 40.0], [70.0, 50.0], [90.0, 60.0]])
            base += rng.normal(0.0, 2.0, size=base.shape)
            tasks.append((typ, base + orientation * shift[None, :], True))
        for _ in range(nuisance_tasks):
            tasks.append(("nuisance", rng.uniform(40.0, 120.0, size=(3, 2)), False))
        rng.shuffle(tasks)
        for typ, xx, is_focal in tasks:
            utility = np.array([beta[0], 0.0, beta[1]])
            utility += beta[2:] @ (xx.T / 100.0)
            if condition == "nonlinear":
                utility += 6.0 * (xx[:, 0] / 100.0) ** 2
            elif condition == "random_price":
                utility += random_cost * (xx[:, 1] / 100.0)
            elif condition == "interaction":
                utility += 8.0 * (xx[:, 0] / 100.0) * (xx[:, 1] / 100.0)
            p = softmax(utility[None, :])[0]
            xs.append(xx)
            ys.append(int(rng.choice(3, p=p)))
            ids.append(rid)
            orientations.append(orientation if is_focal else 0)
            focal.append(is_focal)
    return (np.asarray(ids), np.asarray(ys), np.asarray(xs),
            np.asarray(orientations), np.asarray(focal))


def evaluate(data, seed: int, randomization_reps: int = 499):
    ids, y, x, orientation, focal = data
    unique = np.unique(ids)
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    cluster_scores, clusters = [], []
    for test_ids in np.array_split(shuffled, 2):
        test_mask = np.isin(ids, test_ids)
        train_mask = ~test_mask
        beta = fit_mnl(design(x[train_mask], None), y[train_mask])
        probs = softmax(np.einsum("njp,p->nj", design(x[test_mask], None), beta))
        rows = np.flatnonzero(test_mask)
        local = {row: i for i, row in enumerate(rows)}
        for rid in np.unique(ids[test_mask]):
            cluster_rows = rows[ids[rows] == rid]
            score = np.zeros(3)
            for row in cluster_rows[focal[cluster_rows]]:
                z = 1.0 if orientation[row] else -1.0
                score += z * (np.eye(3)[y[row]] - probs[local[row]])
            cluster_scores.append(score)
            clusters.append(int(rid))
    pvalue = randomization_pvalue(
        np.asarray(cluster_scores), np.asarray(clusters),
        randomization_reps, seed + 19,
    )
    return pvalue, len(cluster_scores)


def run(reps: int = 100, respondents: int = 300, randomization_reps: int = 499,
        out: str = "results/bmst_assignment_benchmark.csv"):
    rows = []
    for condition in CONDITIONS:
        for rep in range(reps):
            pvalue, clusters = evaluate(
                generate(20261030 + rep, respondents, condition=condition),
                20500000 + rep, randomization_reps,
            )
            rows.append({
                "condition": condition,
                "rep": rep,
                "clusters": clusters,
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
    print(f"wrote {len(rows)} BMST assignment rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--respondents", type=int, default=300)
    parser.add_argument("--randomization-reps", type=int, default=499)
    parser.add_argument("--out", default="results/bmst_assignment_benchmark.csv")
    run(**vars(parser.parse_args()))
