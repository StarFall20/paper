"""Gate test for a sequential, coverage-directed fibre design.

Wave 1 estimates a residual direction.  Wave 2 selects a candidate-preserving
fibre using only wave-1 data and then uses an independent randomized assignment
for inference.  The gate compares this design with a fixed fibre.  It is a
method-transfer experiment, not a new primary estimand.
"""
from __future__ import annotations

import csv
import math
import os

import numpy as np


def profiles() -> np.ndarray:
    return np.asarray([(a, b) for a in range(3) for b in range(3)], dtype=float)


def features(x: np.ndarray) -> np.ndarray:
    return np.column_stack((x[:, 0] - x[:, 1], x[:, 0] * x[:, 1]))


def fibre_pairs(x: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    phi = x.sum(axis=1).astype(int)
    pairs = {
        1: (np.asarray([0, 1], dtype=float), np.asarray([1, 0], dtype=float)),
        2: (np.asarray([0, 2], dtype=float), np.asarray([2, 0], dtype=float)),
        3: (np.asarray([1, 2], dtype=float), np.asarray([2, 1], dtype=float)),
    }
    assert all(int(a.sum()) == g and int(b.sum()) == g for g, (a, b) in pairs.items())
    return pairs


def logistic(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def wave1_direction(rng: np.random.Generator, x: np.ndarray, u: np.ndarray,
                    eta: float, n: int) -> np.ndarray:
    ids = rng.integers(0, len(x), size=n)
    xx = x[ids]
    rr = features(xx)
    # The null response surface is allowed to vary by candidate summary.  Remove
    # the fibre mean before estimating the raw-coordinate direction.
    z = xx.sum(axis=1).astype(int)
    centered = rr.copy()
    for g in np.unique(z):
        mask = z == g
        centered[mask] -= centered[mask].mean(axis=0)
    p = logistic(eta * (centered @ u))
    y = rng.binomial(1, p)
    gram = centered.T @ centered + 1e-6 * np.eye(2)
    return np.linalg.solve(gram, centered.T @ (y - 0.5))


def choose_fibre(direction: np.ndarray, pairs: dict[int, tuple[np.ndarray, np.ndarray]]) -> int:
    scores = {}
    for g, (a, b) in pairs.items():
        delta = features(a[None, :])[0] - features(b[None, :])[0]
        scores[g] = float(abs(direction @ delta))
    return max(scores, key=scores.get)


def wave2_pvalue(rng: np.random.Generator, pair: tuple[np.ndarray, np.ndarray],
                 u: np.ndarray, eta: float, n: int) -> float:
    # One member of a candidate-preserving pair is assigned independently to
    # each respondent.  The mean signed outcome has a known null variance.
    a, b = pair
    profiles2 = np.vstack((a, b))
    assignment = rng.integers(0, 2, size=n)
    xx = profiles2[assignment]
    centered = features(xx)
    # For a fixed pair, centering by the pair mean gives the local residual.
    centered -= features(profiles2).mean(axis=0)
    p = logistic(eta * (centered @ u))
    y = rng.binomial(1, p)
    signed = (2 * assignment - 1) * (y - 0.5)
    z = abs(float(signed.mean()) / np.sqrt(0.25 / n))
    # Two-sided normal tail without scipy.
    return float(math.erfc(z / math.sqrt(2.0)))


def run(out: str = "results/sequential_fibre_coverage_gate.csv",
        reps: int = 500, seed: int = 20261007) -> None:
    rng = np.random.default_rng(seed)
    x = profiles()
    pairs = fibre_pairs(x)
    conditions = {
        "null": (np.asarray([0.0, 0.0]), 0.0),
        "h1": (np.asarray([1.0, 0.0]), 0.45),
        "h2": (np.asarray([0.0, 1.0]), 0.45),
        "mixed": (np.asarray([1.0, 0.7]), 0.45),
    }
    rows = []
    for condition, (u, eta) in conditions.items():
        adaptive_reject = 0
        static_reject = 0
        chosen = {1: 0, 2: 0, 3: 0}
        for _ in range(reps):
            direction = wave1_direction(rng, x, u, eta, n=240)
            g_adapt = choose_fibre(direction, pairs)
            chosen[g_adapt] += 1
            p_adapt = wave2_pvalue(rng, pairs[g_adapt], u, eta, n=240)
            p_static = wave2_pvalue(rng, pairs[2], u, eta, n=240)
            adaptive_reject += p_adapt < 0.05
            static_reject += p_static < 0.05
        rows.append({
            "condition": condition,
            "adaptive_rejection": adaptive_reject / reps,
            "static_fibre2_rejection": static_reject / reps,
            "adaptive_chosen_fibre1": chosen[1] / reps,
            "adaptive_chosen_fibre2": chosen[2] / reps,
            "adaptive_chosen_fibre3": chosen[3] / reps,
            "wave1_n": 240,
            "wave2_n": 240,
            "replications": reps,
            "seed": seed,
        })
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print("wrote", len(rows), "rows to", out)


if __name__ == "__main__":
    run()
