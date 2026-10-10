"""Planning power for the actual eight-task, respondent-randomized pilot.

The finite-support benchmark treats each draw as one independent response.
The field instrument assigns one complete questionnaire version to a
respondent who answers eight tasks, so its planning calculation must preserve
respondent-level dependence and use the respondent as the randomization unit.
This script simulates clustered multinomial choices and applies a
respondent-level permutation Wald statistic to the task-by-choice response
vector. It is a sample-size planning calculation, not evidence from human
respondents.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np


def wilson(successes: int, n: int):
    z = 1.959963984540054
    p = successes / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return mid - half, mid + half


def task_cards():
    # The sharing and price rows are taken from the respondent-facing cards.
    sharing_a = np.array([0, 2, 1, 0, 2, 1, 0, 2], dtype=float)
    sharing_b = np.array([1, 1, 0, 2, 1, 0, 2, 1], dtype=float)
    price_a = np.array([79, 59, 69, 79, 89, 99, 109, 119], dtype=float)
    price_b = np.array([99, 89, 109, 119, 69, 79, 59, 69], dtype=float)
    # Both versions have phi_A=(2,2) and phi_B=(3,1).
    phi_a = np.array([2.0, 2.0])
    phi_b = np.array([3.0, 1.0])
    return sharing_a, sharing_b, price_a, price_b, phi_a, phi_b


def fixed_utilities():
    sharing_a, sharing_b, price_a, price_b, phi_a, phi_b = task_cards()
    # A simple prespecified candidate response surface for planning. The
    # coefficients are deliberately modest so neither product dominates.
    ua = 0.24 * phi_a[0] + 0.16 * phi_a[1] + 0.10 * sharing_a - 0.012 * price_a
    ub = 0.24 * phi_b[0] + 0.16 * phi_b[1] + 0.10 * sharing_b - 0.012 * price_b
    uo = np.full(8, -0.20)
    base = np.column_stack((ua, ub, uo))
    # A hidden raw-level contrast: the squared-level average is 2.0 for A in
    # Version 1 and 1.0 for A in Version 2; B equals 1.5 in both versions.
    # It keeps phi fixed and creates a controlled alternative to sufficiency.
    hidden_v1 = np.column_stack((np.full(8, 2.0), np.full(8, 1.5), np.zeros(8)))
    hidden_v2 = np.column_stack((np.full(8, 1.0), np.full(8, 1.5), np.zeros(8)))
    return base, hidden_v1, hidden_v2


def simulate(seed: int, respondents: int, eta: float):
    rng = np.random.default_rng(seed)
    base, hidden_v1, hidden_v2 = fixed_utilities()
    if respondents % 2:
        raise ValueError("respondents must be even for the fixed 1:1 version allocation")
    version = np.zeros(respondents, dtype=int)
    version[: respondents // 2] = 1
    rng.shuffle(version)
    # Random coefficients and an opt-out intercept induce within-respondent
    # dependence across the eight tasks.
    random_effect = rng.normal(0.0, 0.42, size=(respondents, 3))
    random_effect[:, 2] = rng.normal(0.0, 0.30, size=respondents)
    scale = np.exp(rng.normal(0.0, 0.12, size=respondents))
    utilities = base[None, :, :] + random_effect[:, None, :]
    utilities = utilities + eta * np.where(version[:, None, None] == 1, hidden_v1[None], hidden_v2[None])
    utilities = utilities * scale[:, None, None]
    gumbel = -np.log(-np.log(rng.uniform(size=utilities.shape)))
    choice = np.argmax(utilities + gumbel, axis=2)
    return version, choice


def _scores(choice):
    # Drop the opt-out indicator, which is linearly implied by the two product
    # indicators within each task. The task-by-product vector retains all eight
    # repeated observations while keeping the covariance nonsingular after a
    # small numerical ridge.
    return (choice[:, :, None] == np.array([0, 1])[None, None, :]).astype(float).reshape(len(choice), -1)


def _statistic(scores, treated, covariance=None):
    n = len(scores)
    n1 = int(treated.sum())
    n0 = n - n1
    mean1 = scores[treated].mean(axis=0)
    mean0 = scores[~treated].mean(axis=0)
    diff = mean1 - mean0
    if covariance is None:
        centered = scores - scores.mean(axis=0)
        covariance = centered.T @ centered / max(n - 1, 1)
        covariance = covariance + np.eye(scores.shape[1]) * 1e-7
    return float((n0 * n1 / n) * (diff @ np.linalg.pinv(covariance) @ diff))


def permutation_pvalue(version, choice, randomization_reps=499, seed=0):
    scores = _scores(choice)
    covariance = (scores - scores.mean(axis=0)).T @ (scores - scores.mean(axis=0)) / max(len(scores) - 1, 1)
    covariance = covariance + np.eye(scores.shape[1]) * 1e-7
    observed = _statistic(scores, version == 1, covariance)
    rng = np.random.default_rng(seed)
    n1 = int(np.sum(version == 1))
    exceed = 0
    for _ in range(randomization_reps):
        draw = np.zeros(len(version), dtype=bool)
        draw[rng.choice(len(version), size=n1, replace=False)] = True
        exceed += int(_statistic(scores, draw, covariance) >= observed - 1e-12)
    return (1 + exceed) / (randomization_reps + 1)


def run(reps=200, randomization_reps=499,
        out="results/questionnaire_cluster_power.csv"):
    rows = []
    for respondents in (200, 400, 800):
        for eta in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
            rejects = 0
            for rep in range(reps):
                version, choice = simulate(20261010 + 1000 * respondents + rep, respondents, eta)
                pvalue = permutation_pvalue(version, choice, randomization_reps,
                                            seed=20262010 + rep)
                rejects += int(pvalue <= 0.05)
            low, high = wilson(rejects, reps)
            row = {
                "design": "eight_task_respondent_cluster",
                "randomization_unit": "respondent",
                "tasks_per_respondent": 8,
                "respondents": respondents,
                "eta": eta,
                "replications": reps,
                "randomization_reps": randomization_reps,
                "rejection_rate": rejects / reps,
                "ci_low": low,
                "ci_high": high,
            }
            rows.append(row)
            print(respondents, eta, round(row["rejection_rate"], 3), flush=True)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} planning rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--randomization-reps", type=int, default=499)
    parser.add_argument("--out", default="results/questionnaire_cluster_power.csv")
    run(**vars(parser.parse_args()))
