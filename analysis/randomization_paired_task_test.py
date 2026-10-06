"""Design-based paired-task randomization test.

This is an audit of a sharper version of the model-equivalent task idea.  The
observed statistic uses the physical base-versus-shift orientation.  Under the
candidate, the two one-hot outcomes in a pair are exchangeable, so a paired
sign-flip distribution supplies a finite-sample reference conditional on the
observed unordered outcomes.  The test is kept separate from the multiplier
bootstrap benchmark because its validity relies on pair-level exchangeability
and a purpose-built instrument.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from exact_paired_task_benchmark import design, fit_mnl, simulate, softmax


def randomization_pvalue(contrasts, clusters, reps, seed):
    contrasts = np.asarray(contrasts, dtype=float)
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    cluster_index = {rid: i for i, rid in enumerate(unique)}
    mean = contrasts.mean(0)
    observed = float(mean @ mean)
    rng = np.random.default_rng(seed)
    # Cluster-level signs preserve arbitrary within-respondent dependence.
    signs = rng.choice(np.array([-1.0, 1.0]), size=(reps, len(unique)))
    signs = signs[:, [cluster_index[rid] for rid in clusters]]
    draws = (signs @ contrasts) / len(contrasts)
    stats = np.sum(draws * draws, axis=1)
    return float((1.0 + np.sum(stats >= observed)) / (reps + 1.0))


def evaluate(ids, y, x, pair_ids, pair_side, pair_type, term, randomization_reps, seed):
    unique = np.unique(ids)
    rng = np.random.default_rng(seed)
    shuffled = unique.copy(); rng.shuffle(shuffled)
    contrasts = []; clusters = []; types = []
    for test_ids in np.array_split(shuffled, 2):
        test_mask = np.isin(ids, test_ids)
        train_mask = ~test_mask
        beta = fit_mnl(design(x[train_mask], term), y[train_mask])
        probs = softmax(np.einsum("njp,p->nj", design(x[test_mask], term), beta))
        test_rows = np.flatnonzero(test_mask)
        local = {global_row: i for i, global_row in enumerate(test_rows)}
        by_pair = {}
        for global_row in test_rows:
            pid = int(pair_ids[global_row])
            if pid >= 0:
                by_pair.setdefault(pid, {})[int(pair_side[global_row])] = global_row
        for sides in by_pair.values():
            if 0 not in sides or 1 not in sides:
                continue
            left, right = sides[0], sides[1]
            observed = np.eye(3)[y[left]] - np.eye(3)[y[right]]
            pl = probs[local[left]]; pr = probs[local[right]]
            contrasts.append(observed - (pl - pr))
            clusters.append(int(ids[left]))
            types.append(pair_type[left])
    contrasts = np.asarray(contrasts)
    clusters = np.asarray(clusters)
    types = np.asarray(types)
    records = [{"pair_type": "all",
                "pvalue": randomization_pvalue(contrasts, clusters,
                                                randomization_reps, seed + 7),
                "pairs": len(contrasts)}]
    for typ in sorted(set(types.tolist())):
        take = types == typ
        records.append({"pair_type": typ,
                        "pvalue": randomization_pvalue(contrasts[take],
                                                        clusters[take],
                                                        randomization_reps,
                                                        seed + 17 + len(typ)),
                        "pairs": int(take.sum())})
    return records


def run(reps=100, randomization_reps=499, respondents=300,
        out="results/randomization_paired_task_test.csv"):
    rows = []
    conditions = ("additive", "nonlinear", "threshold", "interaction", "random_price",
                  "nonlinear_random_price", "interaction_random_price", "cubic")
    for condition in conditions:
        for rep in range(reps):
            data = simulate(20261007 + rep, respondents=respondents, condition=condition)
            oracle_term = {"nonlinear": "nonlinear", "threshold": "threshold",
                           "interaction": "interaction", "nonlinear_random_price": "nonlinear",
                           "interaction_random_price": "interaction"}.get(condition)
            candidate_terms = [None] if oracle_term is None else [None, oracle_term]
            for term in candidate_terms:
                records = evaluate(*data, term=term,
                                   randomization_reps=randomization_reps,
                                   seed=20600000 + rep)
                for record in records:
                    rows.append({"condition": condition, "rep": rep,
                                 "candidate": "oracle_term" if term else "additive",
                                 **record, "reject": int(record["pvalue"] < 0.05)})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    for condition in conditions:
        for candidate in ("additive", "oracle_term"):
            subset = [r for r in rows if r["condition"] == condition and
                      r["candidate"] == candidate and r["pair_type"] == "all"]
            if subset:
                print(condition, candidate, "rejection_rate", round(float(np.mean([r["reject"] for r in subset])), 3))
    print(f"wrote {len(rows)} randomization-test rows to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--randomization-reps", type=int, default=499)
    ap.add_argument("--respondents", type=int, default=300)
    ap.add_argument("--out", default="results/randomization_paired_task_test.csv")
    run(**vars(ap.parse_args()))
