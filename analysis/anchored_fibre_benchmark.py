"""Benchmark the anchored, coarsened utility-fibre design.

The candidate observes only a composite burden s=a1+a2. The observational
sample keeps the decomposition coordinate z=a1-a2 near zero, so an omitted
z^2 term is nearly invisible to an ordinary likelihood-ratio test and to an
out-of-fold residual learner. The fibre supplement moves along s-constant
directions and assigns one member of each pair to each respondent.

This script is a planning benchmark for the revised innovation. It is
deliberately binary: the zero-gap anchor has a direct scale-invariance result.
The multinomial extension is a separate design arm in the manuscript.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np


def sigmoid(x):
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def fit_logit(features, y, steps=900, lr=0.12, l2=1e-4):
    features = np.asarray(features, dtype=float)
    y = np.asarray(y, dtype=float)
    # Standardize non-intercept columns for stable gradient steps, then return
    # coefficients on the original feature scale.
    center = features[:, 1:].mean(0) if features.shape[1] > 1 else np.zeros(0)
    spread = features[:, 1:].std(0) if features.shape[1] > 1 else np.zeros(0)
    spread = np.where(spread < 1e-8, 1.0, spread)
    scaled = features.copy()
    if features.shape[1] > 1:
        scaled[:, 1:] = (features[:, 1:] - center) / spread
    beta = np.zeros(scaled.shape[1])
    for _ in range(steps):
        p = sigmoid(scaled @ beta)
        grad = scaled.T @ (p - y) / len(y) + l2 * beta
        beta -= lr * grad
    if features.shape[1] > 1:
        raw = beta.copy()
        raw[1:] = beta[1:] / spread
        raw[0] = beta[0] - np.sum(beta[1:] * center / spread)
        return raw
    return beta


def loglik(features, y, beta):
    p = sigmoid(features @ beta)
    return float(np.sum(
        y * np.log(np.clip(p, 1e-12, 1.0)) +
        (1.0 - y) * np.log(np.clip(1.0 - p, 1e-12, 1.0))
    ))


def pvalue_signflip(scores, reps, seed):
    scores = np.asarray(scores, dtype=float)
    observed = abs(float(np.mean(scores)))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(reps, len(scores)))
    draws = np.abs((signs @ scores) / len(scores))
    return float((1.0 + np.sum(draws >= observed)) / (reps + 1.0))


def raw_from_s_z(s, z):
    return np.column_stack(((s + z) / 2.0, (s - z) / 2.0))


def complexity_distance(a, b):
    return np.abs(a - b).sum(axis=1)


def task_probability(a, b, condition):
    s_a = a.sum(axis=1); s_b = b.sum(axis=1)
    z_a = a[:, 0] - a[:, 1]; z_b = b[:, 0] - b[:, 1]
    utility_gap = -0.055 * (s_a - s_b)
    if condition == "omitted_decomposition":
        utility_gap += 0.0010 * (z_a ** 2 - z_b ** 2)
    scale = np.ones(len(a))
    if condition == "complexity_only":
        scale += 0.016 * complexity_distance(a, b)
    return sigmoid(utility_gap / scale)


def build(seed, respondents, condition, nuisance_tasks=8):
    rng = np.random.default_rng(seed)
    obs = []
    focal = []
    for rid in range(respondents):
        # The observational support is deliberately close to z=0.
        s_a = rng.uniform(65.0, 105.0, size=nuisance_tasks)
        s_b = rng.uniform(65.0, 105.0, size=nuisance_tasks)
        z_a = rng.normal(0.0, 1.0, size=nuisance_tasks)
        z_b = rng.normal(0.0, 1.0, size=nuisance_tasks)
        a = raw_from_s_z(s_a, z_a)
        b = raw_from_s_z(s_b, z_b)
        p = task_probability(a, b, condition)
        y = rng.binomial(1, p)
        for q in range(nuisance_tasks):
            obs.append((rid, a[q], b[q], int(y[q])))

        orientation = int(rng.integers(0, 2))
        # Each tuple is (type, base A, base B, transformed A, transformed B).
        pairs = []
        # Zero-gap pair: both alternatives have s=90.
        base_a = raw_from_s_z(np.array([90.0]), np.array([30.0]))[0]
        base_b = raw_from_s_z(np.array([90.0]), np.array([-10.0]))[0]
        # Common null move preserves pairwise raw geometry.
        trans_a = base_a + np.array([10.0, -10.0])
        trans_b = base_b + np.array([10.0, -10.0])
        pairs.append(("zero_gp", base_a, base_b, trans_a, trans_b))

        # Nonzero-gap geometry-preserving pair.
        base_a = raw_from_s_z(np.array([120.0]), np.array([20.0]))[0]
        base_b = raw_from_s_z(np.array([80.0]), np.array([0.0]))[0]
        trans_a = base_a + np.array([10.0, -10.0])
        trans_b = base_b + np.array([10.0, -10.0])
        pairs.append(("gap_gp", base_a, base_b, trans_a, trans_b))

        # Nonzero-gap geometry-changing decomposition move. Each alternative
        # keeps its composite burden s, while the pairwise raw distance changes.
        base_a = raw_from_s_z(np.array([120.0]), np.array([20.0]))[0]
        base_b = raw_from_s_z(np.array([80.0]), np.array([0.0]))[0]
        trans_a = base_a + np.array([30.0, -30.0])
        trans_b = base_b + np.array([-30.0, 30.0])
        pairs.append(("gap_gc", base_a, base_b, trans_a, trans_b))

        # Zero-gap geometry-changing pair. It is a scale-invariant complexity
        # control under a symmetric binary comparison rule.
        base_a = raw_from_s_z(np.array([90.0]), np.array([30.0]))[0]
        base_b = raw_from_s_z(np.array([90.0]), np.array([-10.0]))[0]
        trans_a = base_a + np.array([10.0, -10.0])
        trans_b = base_b + np.array([-10.0, 10.0])
        pairs.append(("zero_gc", base_a, base_b, trans_a, trans_b))

        for typ, a0, b0, a1, b1 in pairs:
            a_use, b_use = (a1, b1) if orientation else (a0, b0)
            p = float(task_probability(a_use[None, :], b_use[None, :], condition)[0])
            y = int(rng.binomial(1, p))
            focal.append({"rid": rid, "kind": typ, "orientation": orientation,
                          "a": a_use, "b": b_use, "y": y})
    return obs, focal


def candidate_features(rows, enriched=False):
    sdiff = np.array([r[1].sum() - r[2].sum() for r in rows], dtype=float)
    if not enriched:
        return np.column_stack((np.ones(len(rows)), sdiff))
    zdiff = np.array([(r[1][0] - r[1][1]) ** 2 -
                      (r[2][0] - r[2][1]) ** 2 for r in rows], dtype=float)
    return np.column_stack((np.ones(len(rows)), sdiff, zdiff))


def lr_diagnostic(obs, seed):
    del seed  # retained in the signature so planning runs remain reproducible
    y = np.array([r[3] for r in obs])
    x0 = candidate_features(obs, False)
    x1 = candidate_features(obs, True)
    b0 = fit_logit(x0, y)
    b1 = fit_logit(x1, y)
    stat = 2.0 * (loglik(x1, y, b1) - loglik(x0, y, b0))
    # One enriched direction; 3.84 is the conventional 5% reference. This is
    # a planning comparator, not a claim of exact finite-sample chi-square size.
    return int(stat > 3.84), stat


def residual_diagnostic(obs, seed, permutation_reps=199):
    rng = np.random.default_rng(seed)
    ids = np.array([r[0] for r in obs])
    unique = np.unique(ids); rng.shuffle(unique)
    train_ids = set(unique[:len(unique) // 2])
    train = [r for r in obs if r[0] in train_ids]
    test = [r for r in obs if r[0] not in train_ids]
    beta = fit_logit(candidate_features(train, False),
                     np.array([r[3] for r in train]))
    X = candidate_features(test, False)
    residual = np.array([r[3] for r in test]) - sigmoid(X @ beta)
    # A small random-feature residual learner. It searches nonlinear functions
    # of the decomposition coordinate without adding them to the candidate
    # utility model.
    test_ids = np.array([r[0] for r in test])
    z = np.array([r[1][0] - r[1][1] for r in test])
    H = np.column_stack((z, z ** 2,
                         np.cos(0.5 * z), np.cos(z + 0.7),
                         np.cos(2.0 * z + 1.4)))
    H -= H.mean(0, keepdims=True)
    denom = np.sqrt(np.mean(H ** 2, axis=0) + 1e-12)
    moments = residual[:, None] * H / denom
    # Aggregate by respondent before the sign-flip reference.
    cluster_scores = np.array([moments[test_ids == rid].sum(0)
                               for rid in np.unique(test_ids)])
    observed = np.max(np.abs(cluster_scores.mean(0)))
    signs = rng.choice(np.array([-1.0, 1.0]),
                       size=(permutation_reps, len(cluster_scores)))
    draws = (signs @ cluster_scores) / len(cluster_scores)
    pvalue = (1.0 + np.sum(np.max(np.abs(draws), axis=1) >= observed)) / (permutation_reps + 1.0)
    return int(pvalue < 0.05), float(pvalue)


def fibre_diagnostic(obs, focal, kind, seed, permutation_reps=199):
    rng = np.random.default_rng(seed)
    ids = np.array(sorted({r["rid"] for r in focal}))
    rng.shuffle(ids)
    train_ids = set(ids[:len(ids) // 2])
    train = [r for r in obs if r[0] in train_ids]
    test_ids = set(ids[len(ids) // 2:])
    beta = fit_logit(candidate_features(train, False),
                     np.array([r[3] for r in train]))
    rows = [r for r in focal if r["rid"] in test_ids and r["kind"] == kind]
    scores = []
    for r in rows:
        sdiff = r["a"].sum() - r["b"].sum()
        p = float(sigmoid(np.array([1.0, sdiff]) @ beta))
        sign = 1.0 if r["orientation"] else -1.0
        scores.append(sign * (r["y"] - p))
    pvalue = pvalue_signflip(scores, permutation_reps, seed + len(kind))
    return int(pvalue < 0.05), float(pvalue)


def run(reps=100, respondents=400, permutation_reps=199,
        out="results/anchored_fibre_benchmark.csv"):
    conditions = ("null", "omitted_decomposition", "complexity_only")
    kinds = ("zero_gp", "gap_gp", "zero_gc", "gap_gc")
    rows = []
    for condition in conditions:
        for rep in range(reps):
            obs, focal = build(20800000 + rep, respondents, condition)
            lr_reject, lr_stat = lr_diagnostic(obs, 20810000 + rep)
            ml_reject, ml_p = residual_diagnostic(obs, 20820000 + rep,
                                                  permutation_reps)
            rows.append({"condition": condition, "rep": rep,
                         "diagnostic": "LR_observational",
                         "arm": "observational", "reject": lr_reject,
                         "stat": lr_stat})
            rows.append({"condition": condition, "rep": rep,
                         "diagnostic": "ML_residual_proxy",
                         "arm": "observational", "reject": ml_reject,
                         "stat": ml_p})
            for kind in kinds:
                reject, pvalue = fibre_diagnostic(
                    obs, focal, kind, 20830000 + rep, permutation_reps)
                rows.append({"condition": condition, "rep": rep,
                             "diagnostic": "anchored_fibre",
                             "arm": kind, "reject": reject, "stat": pvalue})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    for condition in conditions:
        for diagnostic in ("LR_observational", "ML_residual_proxy", "anchored_fibre"):
            for arm in (("observational",) if diagnostic != "anchored_fibre" else kinds):
                subset = [r for r in rows if r["condition"] == condition and
                          r["diagnostic"] == diagnostic and r["arm"] == arm]
                if subset:
                    print(condition, diagnostic, arm,
                          "rejection_rate", round(float(np.mean([r["reject"] for r in subset])), 3))
    print(f"wrote {len(rows)} anchored-fibre rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--respondents", type=int, default=400)
    parser.add_argument("--permutation-reps", type=int, default=199)
    parser.add_argument("--out", default="results/anchored_fibre_benchmark.csv")
    run(**vars(parser.parse_args()))
