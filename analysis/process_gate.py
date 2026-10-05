"""Prototype process-aware gate for the innovation audit.

This file is intentionally separate from the locked primary benchmark.  It
adds two mechanisms that a utility-only specification cannot represent:

* attribute non-attendance (a respondent ignores price in every task), and
* state dependence (a respondent receives a repeat-choice bonus after the
  first task).

The diagnostic reports the respondent-level price-score overdispersion used by
``heterogeneity_gate.py`` and a sequence residual that compares observed repeat
choices with the repeat probability implied by an independent-task MNL.  The
prototype is evidence for a triage question, not a calibrated inferential
test.  Any threshold used in the paper must be obtained by a parametric
bootstrap and fixed before test evaluation.
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass

import numpy as np

from run_simulation import (
    N_ALTERNATIVES,
    N_ATTRIBUTES,
    OPT_OUT_INDEX,
    PRICE_INDEX,
    fit_mnl,
    softmax,
)


@dataclass(frozen=True)
class ProcessCondition:
    name: str
    ana: bool = False
    inertia: bool = False
    random_price: bool = False


CONDITIONS = [
    ProcessCondition("additive"),
    ProcessCondition("random_price", random_price=True),
    ProcessCondition("attribute_nonattendance", ana=True),
    ProcessCondition("inertia", inertia=True),
    ProcessCondition("ana_plus_inertia", ana=True, inertia=True),
]


def make_data(seed: int, n: int, tasks: int, condition: ProcessCondition):
    """Generate panel choices with an explicit process mechanism."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, tasks, N_ALTERNATIVES, N_ATTRIBUTES))
    x[:, :, OPT_OUT_INDEX, :] = 0.0
    x[:, :, :, PRICE_INDEX] = np.abs(x[:, :, :, PRICE_INDEX]) + 0.5
    x[:, :, OPT_OUT_INDEX, :] = 0.0
    z = rng.normal(size=(n, 2))
    z[:, 0] = (z[:, 0] > 0).astype(float)
    z[:, 1] = (z[:, 1] > 0).astype(float)
    base = np.array([0.55, 0.35, 0.45, 0.25, -0.85])
    v = np.einsum("ntjp,p->ntj", x, base)
    v[:, :, OPT_OUT_INDEX] += -0.35 + 0.15 * z[:, None, 0]

    if condition.ana:
        attends_price = rng.binomial(1, 0.65, size=n).astype(bool)
        # Removing the price contribution gives a direct process-level ANA
        # mechanism while retaining the same displayed choice tasks.
        v[~attends_price, :, :,] -= base[PRICE_INDEX] * x[~attends_price, :, :, PRICE_INDEX]
    if condition.random_price:
        random_price = rng.normal(0.0, 0.45, size=n)
        v += random_price[:, None, None] * x[:, :, :, PRICE_INDEX]

    choices = np.empty((n, tasks), dtype=int)
    probabilities = np.empty((n, tasks, N_ALTERNATIVES), dtype=float)
    for t in range(tasks):
        utility = v[:, t, :].copy()
        if condition.inertia and t > 0:
            utility += 0.85 * (np.arange(N_ALTERNATIVES)[None, :] == choices[:, t - 1, None])
        p = softmax(utility)
        probabilities[:, t, :] = p
        choices[:, t] = np.array([rng.choice(N_ALTERNATIVES, p=row) for row in p])
    return x, z, choices, v, probabilities


def feature_matrix(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return additive alternative-specific features including opt-out."""
    n, tasks, j, _ = x.shape
    flat_x = x.reshape(n * tasks, j, N_ATTRIBUTES)
    optout = np.broadcast_to(
        (np.arange(j) == OPT_OUT_INDEX).astype(float)[None, :, None],
        (n * tasks, j, 1),
    )
    return np.concatenate([flat_x, optout], axis=2)


def score_overdispersion(x4: np.ndarray, y2: np.ndarray, beta: np.ndarray) -> float:
    """Respondent-level price score overdispersion ratio."""
    linear = np.einsum("ntjp,p->ntj", x4, beta)
    probabilities = softmax(linear.reshape(-1, N_ALTERNATIVES)).reshape(linear.shape)
    price = x4[:, :, :, PRICE_INDEX]
    chosen = price[np.arange(len(y2))[:, None], np.arange(y2.shape[1])[None, :], y2]
    expected = (probabilities * price).sum(axis=2)
    score = (chosen - expected).sum(axis=1)
    information = ((probabilities * price**2).sum(axis=2) - expected**2).sum(axis=1)
    return float(np.sum(score**2) / np.clip(np.sum(information), 1e-12, None))


def fit_and_diagnose(x, z, y, train_ids, test_ids, tasks):
    features = feature_matrix(x, z)
    train_rows = np.concatenate([train_ids * tasks + q for q in range(tasks)])
    train = features[train_rows]
    y_train = np.concatenate([y[train_ids, q] for q in range(tasks)])
    beta = fit_mnl(train, y_train)
    test_rows = (test_ids[:, None] * tasks + np.arange(tasks)[None, :]).reshape(-1)
    test_x = features[test_rows].reshape(len(test_ids), tasks, N_ALTERNATIVES, -1)
    q_price = score_overdispersion(test_x, y[test_ids], beta)

    linear = np.einsum("ntjp,p->ntj", test_x, beta)
    p = softmax(linear.reshape(-1, N_ALTERNATIVES)).reshape(linear.shape)
    previous = y[test_ids, :-1]
    observed_repeat = (y[test_ids, 1:] == previous).mean()
    predicted_repeat = p[:, 1:, :][
        np.arange(len(test_ids))[:, None],
        np.arange(tasks - 1)[None, :],
        previous,
    ].mean()
    return q_price, float(observed_repeat - predicted_repeat), float(observed_repeat), float(predicted_repeat)


def run(reps: int = 30, n: int = 400, tasks: int = 12,
        out: str = "results/process_gate_30rep.csv") -> None:
    rows = []
    for ci, condition in enumerate(CONDITIONS):
        for rep in range(reps):
            seed = 20261004 + ci * 1000 + rep
            x, z, y, _, _ = make_data(seed, n, tasks, condition)
            ids = np.arange(n)
            rng = np.random.default_rng(seed + 77)
            rng.shuffle(ids)
            cut = int(0.8 * n)
            q_price, inertia_gap, observed_repeat, predicted_repeat = fit_and_diagnose(
                x, z, y, ids[:cut], ids[cut:], tasks
            )
            rows.append({
                "condition": condition.name,
                "replication": rep,
                "q_price": q_price,
                "inertia_gap": inertia_gap,
                "observed_repeat": observed_repeat,
                "predicted_repeat": predicted_repeat,
                "true_ana": int(condition.ana),
                "true_inertia": int(condition.inertia),
                "true_random_price": int(condition.random_price),
            })
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote process-gate diagnostics to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--out", default="results/process_gate_30rep.csv")
    args = parser.parse_args()
    run(args.reps, args.n, args.tasks, args.out)
