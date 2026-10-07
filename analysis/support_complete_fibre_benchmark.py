"""Calibrated finite-support benchmark for the fibre contrast audit.

The assignment is uniform within each declared non-singleton fibre.  The
randomization reference uses the assignment-law covariance, fixed before the
outcomes are generated.  This avoids reusing an observed-assignment
covariance for every redraw.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from support_complete_fibre_design import candidate_values, fibre_contrast_basis, profiles_3x3


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _center(values, fibres):
    values = np.asarray(values, dtype=float).copy()
    for g in (1, 2, 3):
        ids = np.flatnonzero(fibres == g)
        values[ids] -= values[ids].mean(axis=0)
    return values


def patterns(points):
    a, b = points[:, 0], points[:, 1]
    fibres = candidate_values(points)
    dictionary = _center(np.column_stack((a - b, a * b)), fibres)
    raw = {
        "aligned": _center(a - b, fibres),
        "interaction": _center(a * b, fibres),
        "hidden_quadratic": _center(a * a + 0.7 * b * b, fibres),
        "null": np.zeros(len(points)),
    }
    baseline = sigmoid(-0.35 + 0.18 * (fibres - 2))
    kg = {g: np.sum(fibres == g) for g in (1, 2, 3)}
    assign = np.zeros(len(points))
    for g in (1, 2, 3):
        assign[fibres == g] = 1.0 / (3.0 * kg[g])
    metric = assign * baseline * (1.0 - baseline)
    seed = np.asarray([0.7, -0.4, 1.2, -1.1, 0.3, 0.9, -0.8, 0.6, -0.2])
    gram = dictionary.T @ (metric[:, None] * dictionary)
    hidden = seed - dictionary @ np.linalg.pinv(gram) @ (dictionary.T @ (metric * seed))
    raw["hidden_orthogonal"] = _center(hidden, fibres)
    for name, value in raw.items():
        norm = float(np.sqrt(np.sum(metric * value * value)))
        if name != "null" and norm > 0:
            raw[name] = value / norm
    return raw


def features(points, mode):
    fibres = candidate_values(points)
    if mode == "dictionary":
        a, b = points[:, 0], points[:, 1]
        return _center(np.column_stack((a - b, a * b)), fibres), fibres
    blocks = []
    for g in (1, 2, 3):
        ids = np.flatnonzero(fibres == g)
        q = fibre_contrast_basis(len(ids))
        block = np.zeros((len(points), q.shape[1]))
        block[ids] = q
        blocks.append(block)
    return np.hstack(blocks), fibres


def generate(seed, respondents=400, condition="null", eta=0.45):
    rng = np.random.default_rng(seed)
    points = profiles_3x3(); fibres = candidate_values(points)
    by_fibre = {g: np.flatnonzero(fibres == g) for g in (1, 2, 3)}
    h = patterns(points)[condition]
    selected_fibre = rng.choice(np.array([1, 2, 3]), size=respondents)
    selected_profile = np.array([rng.choice(by_fibre[int(g)]) for g in selected_fibre])
    baseline = -0.35 + 0.18 * (selected_fibre - 2)
    probabilities = sigmoid(baseline + eta * h[selected_profile])
    y = rng.binomial(1, probabilities)
    return points, selected_fibre, selected_profile, y


def score_test(points, selected_fibre, selected_profile, y, mode,
               randomization_reps=499, seed=0):
    X, fibres = features(points, mode)
    p0 = sigmoid(-0.35 + 0.18 * (selected_fibre - 2))
    residual = y - p0
    observed_rows = X[selected_profile]
    score = observed_rows.T @ residual
    covariance = np.zeros((X.shape[1], X.shape[1]))
    for g in (1, 2, 3):
        rows = X[fibres == g]
        exx = rows.T @ rows / len(rows)
        n_g = np.sum(selected_fibre == g)
        p_g = sigmoid(-0.35 + 0.18 * (g - 2))
        covariance += n_g * p_g * (1.0 - p_g) * exx
    covariance += 1e-8 * np.eye(X.shape[1])
    inverse = np.linalg.pinv(covariance)
    statistic = float(score @ inverse @ score)
    rng = np.random.default_rng(seed)
    draws = np.empty((randomization_reps, len(selected_profile)), dtype=int)
    for g in (1, 2, 3):
        row_ids = np.flatnonzero(fibres == g)
        n_g = np.sum(selected_fibre == g)
        if n_g:
            draws[:, selected_fibre == g] = rng.choice(row_ids, size=(randomization_reps, n_g))
    draw_rows = X[draws]
    draw_scores = np.einsum("bnp,n->bp", draw_rows, residual)
    draw_statistics = np.einsum("bp,pq,bq->b", draw_scores, inverse, draw_scores)
    return float((1.0 + np.sum(draw_statistics >= statistic)) / (randomization_reps + 1.0))


def run(reps=200, respondents=400, randomization_reps=499,
        eta=0.45, out="results/support_complete_fibre_benchmark.csv"):
    rows = []
    conditions = ("null", "aligned", "interaction", "hidden_quadratic", "hidden_orthogonal")
    for condition in conditions:
        for mode in ("dictionary", "support_complete"):
            for rep in range(reps):
                points, fibre, profile, y = generate(20261007 + rep,
                    respondents=respondents, condition=condition, eta=eta)
                pvalue = score_test(points, fibre, profile, y, mode,
                    randomization_reps=randomization_reps, seed=20262000 + rep)
                rows.append({"condition": condition, "statistic": mode,
                    "rep": rep, "respondents": respondents, "eta": eta,
                    "pvalue": pvalue, "reject": int(pvalue < 0.05)})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for condition in conditions:
        for mode in ("dictionary", "support_complete"):
            subset = [r for r in rows if r["condition"] == condition and r["statistic"] == mode]
            print(condition, mode, round(float(np.mean([r["reject"] for r in subset])), 3))
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--respondents", type=int, default=400)
    parser.add_argument("--randomization-reps", type=int, default=499)
    parser.add_argument("--eta", type=float, default=0.45)
    parser.add_argument("--out", default="results/support_complete_fibre_benchmark.csv")
    run(**vars(parser.parse_args()))
