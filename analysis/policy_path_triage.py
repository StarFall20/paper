"""Policy-path disagreement plus score-dispersion model-family triage."""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from policy_path_disagreement import fit_policy_models, path_gap  # noqa: E402
from recoverability_triage import (  # noqa: E402
    cluster_score_overdispersion, fit_and_score_family, respondent_split,
    row_indices, target_action,
)
from run_simulation import CONDITIONS, N_ALTERNATIVES, feature_matrix, make_data


def diagnostics(x, z, choices, ids_fit, ids_val, tasks):
    betas = fit_policy_models(x, z, choices, ids_fit, tasks)
    # Evaluate score dispersion after the declared observable library has
    # absorbed the utility-form terms.  This reduces contamination of the
    # heterogeneity signal by a known nonlinear departure.
    structured = feature_matrix(x, z, structured=True)
    vr = row_indices(ids_val, tasks)
    q = cluster_score_overdispersion(structured[vr], choices[ids_val].reshape(-1),
                                     betas[1], len(ids_val), tasks)
    gap = path_gap(x[ids_val], z[ids_val], betas)
    return {**gap, "cluster_score": q}, betas


def run(reps: int = 24, calibration_reps: int = 100, n: int = 400,
        tasks: int = 12, out: str = "results/policy_path_triage.csv"):
    cal = []
    for rep in range(calibration_reps):
        seed = 23100000 + rep
        x, z, choices, _ = make_data(seed, n, tasks, CONDITIONS[0])
        tr, va, _ = respondent_split(n, seed + 77)
        d, _ = diagnostics(x, z, choices, tr, va, tasks)
        cal.append(d)
    # Two diagnostics are calibrated jointly at a conservative 99% tail.
    d_threshold = float(np.quantile([r["path_structured_l1"] for r in cal], .99))
    q_threshold = float(np.quantile([r["cluster_score"] for r in cal], .975))
    rows = []
    for ci, condition in enumerate(CONDITIONS):
        for rep in range(reps):
            seed = 23200000 + ci * 1000 + rep
            x, z, choices, v = make_data(seed, n, tasks, condition)
            tr, va, te = respondent_split(n, seed + 77)
            # Keep the policy-path diagnostic cross-fitted: development
            # respondents estimate the models and validation respondents
            # expose their counterfactual disagreement.
            d, _ = diagnostics(x, z, choices, tr, va, tasks)
            policy = d["path_structured_l1"] > d_threshold
            hetero = d["cluster_score"] > q_threshold
            if policy and hetero:
                action, reason = "unresolved", "policy_and_cluster"
            elif policy:
                action, reason = "observed_structure", "policy_path"
            elif hetero:
                action, reason = "heterogeneity", "cluster_score"
            else:
                action, reason = "base", "no_signal"
            metrics = fit_and_score_family(action, x, z, choices, v,
                                           np.concatenate([tr, va]), te, tasks, seed + 301)
            rows.append({"condition": condition.name, "replication": rep, **d,
                         "threshold_policy_path": d_threshold,
                         "threshold_cluster_score": q_threshold,
                         "policy_signal": int(policy), "heterogeneity_signal": int(hetero),
                         "action": action, "reason": reason,
                         "target_action": target_action(condition),
                         "test_accuracy": metrics[0], "test_logloss": metrics[1],
                         "test_brier": metrics[2], "test_share_rmse": metrics[3],
                         "test_decision_regret": metrics[4]})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for condition in [c.name for c in CONDITIONS]:
        sub = [r for r in rows if r["condition"] == condition]
        print(condition,
              "action_accuracy", round(float(np.mean([r["action"] == r["target_action"] for r in sub])), 3),
              "policy", round(float(np.mean([r["policy_signal"] for r in sub])), 3),
              "hetero", round(float(np.mean([r["heterogeneity_signal"] for r in sub])), 3),
              "regret", round(float(np.mean([r["test_decision_regret"] for r in sub])), 4))
    print("thresholds", d_threshold, q_threshold, "wrote", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=24)
    parser.add_argument("--calibration-reps", type=int, default=100)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--out", default="results/policy_path_triage.csv")
    run(**vars(parser.parse_args()))
