"""Descriptive counterfactual-path disagreement audit on Swissmetro."""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

from observational_equivalence_test import fit_mnl, predict, read_swissmetro, rows_to_arrays
from swissmetro_specification_benchmark import make_design


def split(ids, seed=20261007):
    unique = np.unique(ids).copy()
    np.random.default_rng(seed).shuffle(unique)
    n = len(unique)
    return unique[: int(.60 * n)], unique[int(.60 * n): int(.80 * n)], unique[int(.80 * n):]


def loss(X, y, avail, beta):
    p = predict(X, avail, beta)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def path_gap(x, avail, base_beta, structured_beta):
    # Multiplicative policy path in observed time/cost space.  Availability is
    # held fixed; each point is a feasible perturbation of the public profiles.
    gaps, share_gaps = [], []
    for time_scale in (0.8, 1.0, 1.2):
        for cost_scale in (0.8, 1.0, 1.2):
            xcf = np.array(x, copy=True)
            xcf[:, :, 0] *= time_scale
            xcf[:, :, 1] *= cost_scale
            b = make_design(xcf, "base")
            s = make_design(xcf, "alt_specific")
            pb = predict(b, avail, base_beta)
            ps = predict(s, avail, structured_beta)
            gaps.append(float(np.mean(np.abs(pb - ps))))
            share_gaps.append(float(np.mean(np.abs(pb.mean(0) - ps.mean(0)))))
    return float(np.mean(gaps)), float(np.mean(share_gaps)), float(np.max(gaps))


def run(path: str, out: str, seed: int = 20261007):
    rows = read_swissmetro(path, dce_only=True)
    ids, y, x, avail = rows_to_arrays(rows)
    tr_ids, va_ids, te_ids = split(ids, seed)
    tr, va, te = np.isin(ids, tr_ids), np.isin(ids, va_ids), np.isin(ids, te_ids)
    base = make_design(x, "base")
    structured = make_design(x, "alt_specific")
    b = fit_mnl(base[tr], y[tr], avail[tr])
    s = fit_mnl(structured[tr], y[tr], avail[tr])
    gap, share_gap, max_gap = path_gap(x[va], avail[va], b, s)
    rec = {
        "dataset": "Swissmetro_DCE_purpose_1_3",
        "rows": len(y), "respondents": len(np.unique(ids)),
        "train_respondents": len(tr_ids), "validation_respondents": len(va_ids),
        "test_respondents": len(te_ids),
        "base_validation_logloss": loss(base[va], y[va], avail[va], b),
        "structured_validation_logloss": loss(structured[va], y[va], avail[va], s),
        "base_test_logloss": loss(base[te], y[te], avail[te], b),
        "structured_test_logloss": loss(structured[te], y[te], avail[te], s),
        "policy_path_l1": gap, "policy_path_share_l1": share_gap,
        "policy_path_max_l1": max_gap,
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rec)); writer.writeheader(); writer.writerow(rec)
    print("path L1", round(gap, 6), "share L1", round(share_gap, 6),
          "test logloss", round(rec["base_test_logloss"], 5), round(rec["structured_test_logloss"], 5))
    print("wrote", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--out", default="results/swissmetro_policy_path_audit.csv")
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    run(args.path, args.out, args.seed)
