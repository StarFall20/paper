"""Counterfactual policy-path disagreement audit.

The audit asks a different question from observed-task prediction: do two
behavioural specifications that fit the development data similarly imply the
same choice probabilities along a declared attribute intervention path?  The
path is fixed before outcomes are used.  This is a diagnostic for
policy-relevant recoverability, not a new structural model.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from recoverability_triage import (  # noqa: E402
    expanded_feature_matrix, feature_matrix, fit_mnl, logloss_from_probs,
    mnl_probs, respondent_split, row_indices,
)
from run_simulation import CONDITIONS, N_ALTERNATIVES, make_data, score


PATH = tuple((q, p) for q in (-0.6, 0.0, 0.6) for p in (-0.6, 0.0, 0.6))


def counterfactual_x(x: np.ndarray, quality_shift: float, price_shift: float) -> np.ndarray:
    z = np.array(x, copy=True)
    z[:, :, :2, 0] += quality_shift
    z[:, :, :2, 4] = np.maximum(z[:, :, :2, 4] + price_shift, 0.5)
    z[:, :, 2, :] = 0.0
    return z


def fit_policy_models(x, z, choices, train_ids, tasks):
    rows = row_indices(train_ids, tasks)
    y = choices[train_ids].reshape(-1)
    base = feature_matrix(x, z, structured=False)
    structured = feature_matrix(x, z, structured=True)
    expanded = expanded_feature_matrix(x, z)
    return (fit_mnl(base[rows], y), fit_mnl(structured[rows], y),
            fit_mnl(expanded[rows], y))


def path_gap(x, z, betas):
    base_beta, structured_beta, expanded_beta = betas
    gaps = []
    shares = []
    for q_shift, p_shift in PATH:
        xcf = counterfactual_x(x, q_shift, p_shift)
        b = feature_matrix(xcf, z, structured=False)
        s = feature_matrix(xcf, z, structured=True)
        e = expanded_feature_matrix(xcf, z)
        pb = mnl_probs(b, base_beta)
        ps = mnl_probs(s, structured_beta)
        pe = mnl_probs(e, expanded_beta)
        gaps.append({
            "structured_l1": float(np.mean(np.abs(pb - ps))),
            "expanded_l1": float(np.mean(np.abs(ps - pe))),
            "base_expanded_l1": float(np.mean(np.abs(pb - pe))),
        })
        shares.append({
            "structured_share_l1": float(np.mean(np.abs(pb.mean(0) - ps.mean(0)))),
            "expanded_share_l1": float(np.mean(np.abs(ps.mean(0) - pe.mean(0)))),
        })
    return {
        "path_structured_l1": float(np.mean([g["structured_l1"] for g in gaps])),
        "path_expanded_l1": float(np.mean([g["expanded_l1"] for g in gaps])),
        "path_base_expanded_l1": float(np.mean([g["base_expanded_l1"] for g in gaps])),
        "path_structured_share_l1": float(np.mean([g["structured_share_l1"] for g in shares])),
        "path_expanded_share_l1": float(np.mean([g["expanded_share_l1"] for g in shares])),
    }


def run(reps: int = 24, calibration_reps: int = 100, n: int = 400,
        tasks: int = 12, out: str = "results/policy_path_disagreement.csv"):
    calibration = []
    for rep in range(calibration_reps):
        seed = 22100000 + rep
        x, z, choices, _ = make_data(seed, n, tasks, CONDITIONS[0])
        tr, va, _ = respondent_split(n, seed + 77)
        betas = fit_policy_models(x, z, choices, tr, tasks)
        calibration.append(path_gap(x[va], z[va], betas))
    # A one-sided calibration is used for this single policy-gap family.
    threshold = float(np.quantile([r["path_structured_l1"] for r in calibration], .99))
    rows = []
    for ci, condition in enumerate(CONDITIONS):
        for rep in range(reps):
            seed = 22200000 + ci * 1000 + rep
            x, z, choices, v = make_data(seed, n, tasks, condition)
            tr, va, te = respondent_split(n, seed + 77)
            # Fit the competing specifications only on development
            # respondents; use validation respondents solely to expose the
            # predeclared counterfactual path.
            betas = fit_policy_models(x, z, choices, tr, tasks)
            gap = path_gap(x[va], z[va], betas)
            base = feature_matrix(x, z, structured=False)
            structured = feature_matrix(x, z, structured=True)
            test_rows = row_indices(te, tasks)
            yte = choices[te].reshape(-1)
            base_metrics = score(base[test_rows], yte, v[te].reshape(-1, N_ALTERNATIVES), betas[0])
            structured_metrics = score(structured[test_rows], yte, v[te].reshape(-1, N_ALTERNATIVES), betas[1])
            rows.append({"condition": condition.name, "replication": rep,
                         **gap, "threshold_path_structured_l1": threshold,
                         "policy_alert": int(gap["path_structured_l1"] > threshold),
                         "base_test_regret": base_metrics[4],
                         "structured_test_regret": structured_metrics[4],
                         "structured_test_logloss": structured_metrics[1],
                         "base_test_logloss": base_metrics[1]})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for condition in [c.name for c in CONDITIONS]:
        sub = [r for r in rows if r["condition"] == condition]
        print(condition, "alert", round(float(np.mean([r["policy_alert"] for r in sub])), 3),
              "path_gap", round(float(np.mean([r["path_structured_l1"] for r in sub])), 5),
              "regret_delta", round(float(np.mean([r["base_test_regret"] - r["structured_test_regret"] for r in sub])), 5))
    print("threshold", threshold, "wrote", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=24)
    parser.add_argument("--calibration-reps", type=int, default=100)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--out", default="results/policy_path_disagreement.csv")
    run(**vars(parser.parse_args()))
