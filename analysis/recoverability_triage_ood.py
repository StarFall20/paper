"""Out-of-library cubic stress test for the recoverability triage rule."""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from recoverability_triage import (
    calibrate, classify, fit_and_score_family, get_diagnostics, respondent_split,
)
from run_simulation import N_ALTERNATIVES, CONDITIONS, make_data, softmax


def make_cubic_data(seed: int, n: int, tasks: int):
    """Generate an omitted cubic term absent from the structured candidate set."""
    rng = np.random.default_rng(seed)
    # Reuse the original support and observed-covariate construction, then
    # replace the additive choices with a cubic out-of-library utility.
    x, z, _, _ = make_data(seed, n, tasks, CONDITIONS[0])
    base = np.array([0.55, 0.35, 0.45, 0.25, -0.85])
    v = np.einsum("ntjp,p->ntj", x, base)
    v[:, :, 2] += -0.35
    v += 0.70 * x[:, :, :, 0] ** 3 - 0.35 * x[:, :, :, 4] ** 3
    v[:, :, 2] += 0.15 * z[:, None, 0]
    p = softmax(v.reshape(-1, N_ALTERNATIVES)).reshape(n, tasks, N_ALTERNATIVES)
    choices = np.array([rng.choice(N_ALTERNATIVES, p=pp) for pp in p.reshape(-1, N_ALTERNATIVES)])
    return x, z, choices.reshape(n, tasks), v


def run(reps: int = 24, calibration_reps: int = 100, n: int = 400,
        tasks: int = 12, out: str = "results/recoverability_triage_ood.csv"):
    thresholds, _ = calibrate(calibration_reps, n, tasks)
    rows = []
    for rep in range(reps):
        seed = 20900000 + rep
        x, z, choices, v = make_cubic_data(seed, n, tasks)
        tr, va, te = respondent_split(n, seed + 77)
        d = get_diagnostics(x, z, choices, tr, va, tasks, seed + 101)
        action, reason = classify(d, thresholds)
        metrics = fit_and_score_family(action, x, z, choices, v,
                                       np.concatenate([tr, va]), te, tasks, seed + 202)
        rows.append({"condition": "cubic_out_of_library", "replication": rep,
                     **d, "threshold_structured_gain": thresholds.structured_gain,
                     "threshold_flex_gain": thresholds.flex_gain,
                     "threshold_cluster_score": thresholds.cluster_score,
                     "action": action, "reason": reason,
                     "test_accuracy": metrics[0], "test_logloss": metrics[1],
                     "test_brier": metrics[2], "test_share_rmse": metrics[3],
                     "test_decision_regret": metrics[4]})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    actions = {a: sum(r["action"] == a for r in rows) / len(rows)
               for a in ("base", "observed_structure", "heterogeneity", "unresolved")}
    print("thresholds", thresholds, "actions", actions,
          "mean_regret", float(np.mean([r["test_decision_regret"] for r in rows])))
    print("wrote", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=24)
    parser.add_argument("--calibration-reps", type=int, default=100)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--out", default="results/recoverability_triage_ood.csv")
    run(**vars(parser.parse_args()))
