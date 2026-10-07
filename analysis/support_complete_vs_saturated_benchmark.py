"""Compare the support-complete score with a standard saturated fibre LR.

Both procedures use the same finite fibre and the same conditional
randomization law.  This is a novelty audit: it tests whether the proposed
complete contrast parameterization has inferential content beyond adding a
dummy for each supported profile.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from support_complete_fibre_benchmark import generate, score_test
from support_complete_fibre_design import candidate_values, profiles_3x3


def bernoulli_loglik(y, p):
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return float(np.sum(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def saturated_lr_pvalue(points, selected_fibre, selected_profile, y,
                        reps=499, seed=0):
    fibres = candidate_values(points)
    p0 = 1.0 / (1.0 + np.exp(-(-0.35 + 0.18 * (selected_fibre - 2))))
    profile_ids = np.unique(selected_profile)
    index = {int(v): j for j, v in enumerate(profile_ids)}

    def statistic(profile):
        phat = np.zeros(len(profile_ids))
        for v, j in index.items():
            mask = profile == v
            phat[j] = np.mean(y[mask]) if np.any(mask) else 0.5
        fitted = np.asarray([phat[index[int(v)]] for v in profile])
        return 2.0 * (bernoulli_loglik(y, fitted) - bernoulli_loglik(y, p0))

    observed = statistic(selected_profile)
    rng = np.random.default_rng(seed)
    by_fibre = {g: np.flatnonzero(fibres == g) for g in (1, 2, 3)}
    draws = np.empty((reps, len(selected_profile)), dtype=int)
    for g in (1, 2, 3):
        mask = selected_fibre == g
        if np.any(mask):
            draws[:, mask] = rng.choice(by_fibre[g], size=(reps, np.sum(mask)))
    reference = np.asarray([statistic(draws[b]) for b in range(reps)])
    return float((1.0 + np.sum(reference >= observed)) / (reps + 1.0))


def run(reps=200, respondents=400, randomization_reps=499,
        eta=0.45, out="results/support_complete_vs_saturated_benchmark.csv"):
    points = profiles_3x3()
    conditions = ("null", "aligned", "interaction", "hidden_quadratic", "hidden_orthogonal")
    rows = []
    for condition in conditions:
        for rep in range(reps):
            pts, fibre, profile, y = generate(20261007 + rep,
                respondents=respondents, condition=condition, eta=eta)
            score_p = score_test(pts, fibre, profile, y, "support_complete",
                randomization_reps=randomization_reps, seed=20262000 + rep)
            lr_p = saturated_lr_pvalue(pts, fibre, profile, y,
                reps=randomization_reps, seed=20263000 + rep)
            rows.append({"condition": condition, "rep": rep,
                "score_pvalue": score_p, "saturated_lr_pvalue": lr_p,
                "score_reject": int(score_p < 0.05),
                "saturated_lr_reject": int(lr_p < 0.05)})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for condition in conditions:
        sub = [r for r in rows if r["condition"] == condition]
        print(condition, "score", round(np.mean([r["score_reject"] for r in sub]), 3),
              "saturated_LR", round(np.mean([r["saturated_lr_reject"] for r in sub]), 3))
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--respondents", type=int, default=400)
    parser.add_argument("--randomization-reps", type=int, default=499)
    parser.add_argument("--eta", type=float, default=0.45)
    parser.add_argument("--out", default="results/support_complete_vs_saturated_benchmark.csv")
    run(**vars(parser.parse_args()))
