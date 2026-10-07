"""Minimax falsification design on a candidate-preserving utility fibre.

This module turns the reverse-design idea into one object.  The candidate
utility menu is held fixed within every pair, while the task block is chosen
to maximise the smallest projected local signal over a predeclared library of
plausible omitted directions.  The resulting criterion targets falsification
power, not parameter-estimation efficiency.

The smart-device implementation is a proof-of-concept.  The departure
library is deliberately frozen before simulation outcomes are generated and
contains curvature, threshold, and interaction directions that are not used
to select one hand-picked task.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import smart_device_fibre_candidate_grid as grid
import smart_device_nuisance_balanced_design as balanced


FEATURE_NAMES = (
    "quadratic_total",
    "intelligence_threshold",
    "price_threshold",
    "mode_support_interaction",
    "intelligence_cloud_interaction",
)
BASELINE_TARGET_GAPS = (0.32, 0.40, 0.57, 0.80, 0.98)
# Broad probability-scale coverage is frozen before outcomes. The maximin
# criterion chooses the exact gap within each bin.
MAXIMIN_GAP_BINS = ((0.20, 0.35), (0.35, 0.50), (0.50, 0.65),
                    (0.65, 0.80), (0.80, 1.01))


def logistic(value):
    value = np.clip(np.asarray(value, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-value))


def pair_key(row):
    return (tuple(row["a_plus"]), tuple(row["a_minus"]),
            tuple(row["b_plus"]), tuple(row["b_minus"]))


def departure_features(profile):
    """Return a frozen, interpretable omitted-direction library."""
    mode, intelligence, support, evidence, data, price = profile
    x = np.asarray(profile, dtype=float)
    return np.asarray((
        float(np.sum(x * x)),
        float(int(intelligence == 2)),
        float(int(price == 2)),
        float(int(mode == 2 and support == 2)),
        float(int(intelligence == 2 and data == 2)),
    ), dtype=float)


def odd_departure_vector(row):
    """Odd A--B response for each frozen departure direction."""
    plus = departure_features(row["a_plus"]) - departure_features(row["b_plus"])
    minus = departure_features(row["a_minus"]) - departure_features(row["b_minus"])
    return 0.5 * (plus - minus)


def prepare_pool():
    """Build a nuisance-balanced pool and standardised local signals."""
    all_candidates = grid.enumerate_candidates()
    pool = balanced.balanced_pool(all_candidates)
    if len(pool) < len(MAXIMIN_GAP_BINS):
        raise RuntimeError("candidate fibre pool is smaller than task block")

    raw = np.asarray([odd_departure_vector(row) for row in pool], dtype=float)
    gaps = np.asarray([float(row["gap"]) for row in pool], dtype=float)
    probabilities = logistic(gaps)
    # Local score signals use the square-root Fisher weight.  Column scaling
    # makes the minimax criterion invariant to the units of the frozen basis.
    weighted = np.sqrt(probabilities * (1.0 - probabilities))[:, None] * raw
    scales = np.sqrt(np.mean(weighted * weighted, axis=0))
    scales[scales < 1e-12] = 1.0
    standardised = weighted / scales
    return pool, raw, standardised, scales


def bin_pools(pool):
    pools = []
    for lower, upper in MAXIMIN_GAP_BINS:
        rows = [
            i for i, row in enumerate(pool)
            if lower <= float(row["gap"]) < upper
        ]
        if not rows:
            raise RuntimeError(f"no candidate fibre pair in gap bin {lower}:{upper}")
        pools.append(rows)
    return pools


def objective(indices, design_matrix):
    matrix = design_matrix[np.asarray(indices, dtype=int)]
    information = matrix.T @ matrix
    eigenvalues = np.linalg.eigvalsh(information)
    return float(eigenvalues[0]), float(np.trace(information)), eigenvalues


def choose_maximin(pool, design_matrix):
    """Enumerate the small five-bin design space and choose E-falsification."""
    pools = bin_pools(pool)
    best = None
    checked = 0
    for combo in itertools.product(*pools):
        checked += 1
        if len(set(combo)) != len(combo):
            continue
        minimum, trace, eigenvalues = objective(combo, design_matrix)
        # Trace is a deterministic tie-breaker after the worst direction.
        score = (minimum, trace, tuple(-int(i) for i in combo))
        if best is None or score > best["score"]:
            best = {"indices": tuple(combo), "score": score,
                    "minimum": minimum, "trace": trace,
                    "eigenvalues": eigenvalues}
    if best is None:
        raise RuntimeError("no distinct task block satisfies the gap bins")
    best["checked_combinations"] = checked
    return best


def choose_balanced_baseline(pool):
    selected = balanced.choose_balanced(pool, targets=BASELINE_TARGET_GAPS)
    index_by_key = {pair_key(row): i for i, row in enumerate(pool)}
    indices = tuple(index_by_key[pair_key(row)] for row in selected)
    return indices


def design_rows(pool, indices, design_name, target_gaps, score=None):
    rows = []
    for task, index in enumerate(indices, 1):
        row = pool[index]
        sig = balanced.nuisance_signature(row["a_plus"], row["b_plus"])
        out = OrderedDict(
            design=design_name,
            task=task,
            pool_index=index,
            target_gap=target_gaps[task - 1],
            candidate_gap=float(row["gap"]),
            candidate_A=float(row["u_a"]),
            candidate_B=float(row["u_b"]),
            distance=float(row["distance"]),
            hamming=int(row["hamming"]),
            nuisance_signature=str(sig),
        )
        signals = odd_departure_vector(row)
        for name, value in zip(FEATURE_NAMES, signals):
            out[f"odd_{name}"] = float(value)
        if score is not None:
            out["min_eigenvalue"] = float(score["minimum"])
            out["information_trace"] = float(score["trace"])
        rows.append(out)
    return rows


def design_matrix_for(indices, standardised):
    return standardised[np.asarray(indices, dtype=int)]


def sign_flip_pvalue(z, variances, permutation_reps, rng):
    """Omnibus task-vector sign-flip test with respondent-level signs."""
    n = z.shape[0]
    mean = np.mean(z, axis=0)
    observed = float(n * np.sum((mean * mean) / variances))
    signs = rng.choice(np.asarray((-1.0, 1.0)),
                       size=(permutation_reps, n))
    permuted = (signs @ z) / n
    statistics = n * np.sum((permuted * permuted) / variances[None, :], axis=1)
    return float((1.0 + np.sum(statistics >= observed)) /
                 (permutation_reps + 1.0))


def simulate_one(indices, pool, raw, condition, direction, respondents,
                 permutation_reps, seed, eta=0.75):
    rng = np.random.default_rng(seed)
    rows = [pool[int(i)] for i in indices]
    gaps = np.asarray([float(row["gap"]) for row in rows], dtype=float)
    p0 = logistic(gaps)
    direction = np.asarray(direction, dtype=float)
    raw_matrix = raw[np.asarray(indices, dtype=int)]
    odd = raw_matrix @ direction
    orientation = rng.choice(np.asarray((-1.0, 1.0)), size=respondents)
    if condition == "null":
        odd = np.zeros_like(odd)
    probabilities = logistic(gaps[None, :] +
                             eta * orientation[:, None] * odd[None, :])
    choices = rng.binomial(1, probabilities)
    # Align each respondent's observed member back to the plus orientation.
    z = orientation[:, None] * (choices - p0[None, :])
    variances = p0 * (1.0 - p0)
    pvalue = sign_flip_pvalue(
        z, variances, permutation_reps, np.random.default_rng(seed + 131))
    return int(pvalue < 0.05), pvalue


def benchmark_designs(pool, raw, maximin, baseline, reps=50,
                      respondents=600, permutation_reps=199,
                      out="results/fibre_maximin_falsification_benchmark.csv"):
    rng = np.random.default_rng(20261007)
    directions = [("feature_" + name, np.eye(len(FEATURE_NAMES))[j])
                  for j, name in enumerate(FEATURE_NAMES)]
    for j in range(10):
        direction = rng.normal(size=len(FEATURE_NAMES))
        direction /= np.linalg.norm(direction)
        directions.append((f"random_{j}", direction))

    design_indices = {"balanced_baseline": baseline,
                      "maximin_falsification": maximin["indices"]}
    rows = []
    rep_seed = 61000000
    for design_name, indices in design_indices.items():
        for condition, direction_name, direction in [
            ("null", "null", np.zeros(len(FEATURE_NAMES)))
        ] + [("departure", name, vector) for name, vector in directions]:
            for rep in range(reps):
                reject, pvalue = simulate_one(
                    indices, pool, raw, condition, direction, respondents,
                    permutation_reps, rep_seed, eta=0.75)
                rows.append({"design": design_name, "condition": condition,
                             "direction": direction_name, "rep": rep,
                             "reject": reject, "pvalue": pvalue,
                             "respondents": respondents,
                             "permutation_reps": permutation_reps})
                rep_seed += 1
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_design(out, rows):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        fields = list(OrderedDict.fromkeys(
            field for row in rows for field in row.keys()))
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(out, rows):
    groups = OrderedDict()
    for row in rows:
        key = (row["design"], row["condition"], row["direction"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (design_name, condition, direction), subset in groups.items():
        rejection = float(np.mean([r["reject"] for r in subset]))
        pvalues = np.asarray([r["pvalue"] for r in subset], dtype=float)
        summary.append({"design": design_name, "condition": condition,
                        "direction": direction, "replications": len(subset),
                        "rejection_rate": rejection,
                        "median_pvalue": float(np.median(pvalues))})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        fields = list(summary[0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    return summary


def run(reps=50, respondents=600, permutation_reps=199,
        design_out="results/fibre_maximin_falsification_design.csv",
        benchmark_out="results/fibre_maximin_falsification_benchmark.csv",
        summary_out="results/fibre_maximin_falsification_summary.csv"):
    pool, raw, standardised, scales = prepare_pool()
    best = choose_maximin(pool, standardised)
    baseline = choose_balanced_baseline(pool)
    design_rows_all = design_rows(pool, baseline, "balanced_baseline",
                                  BASELINE_TARGET_GAPS)
    design_rows_all += design_rows(pool, best["indices"],
                                   "maximin_falsification",
                                   tuple((lo + hi) / 2.0
                                         for lo, hi in MAXIMIN_GAP_BINS),
                                   score=best)
    write_design(design_out, design_rows_all)
    rows = benchmark_designs(pool, raw, best, baseline, reps=reps,
                             respondents=respondents,
                             permutation_reps=permutation_reps,
                             out=benchmark_out)
    summary = write_summary(summary_out, rows)
    print("balanced_pool", len(pool))
    print("checked_combinations", best["checked_combinations"])
    print("maximin_indices", list(best["indices"]))
    print("baseline_indices", list(baseline))
    print("maximin_eigenvalues", [round(float(x), 6)
                                  for x in best["eigenvalues"]])
    print("departure_scales", [round(float(x), 6) for x in scales])
    for row in summary:
        if row["condition"] == "null" or row["direction"].startswith("random_"):
            print(row)
    print("wrote", design_out, benchmark_out, summary_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--respondents", type=int, default=600)
    parser.add_argument("--permutation-reps", type=int, default=199)
    parser.add_argument("--design-out", default="results/fibre_maximin_falsification_design.csv")
    parser.add_argument("--benchmark-out", default="results/fibre_maximin_falsification_benchmark.csv")
    parser.add_argument("--summary-out", default="results/fibre_maximin_falsification_summary.csv")
    run(**vars(parser.parse_args()))
