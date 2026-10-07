"""Nuisance-orthogonal fibre quotient completeness in the device testbed.

The candidate basis preserves the full candidate menu differences.  The
departure operator is projected away from declared presentation and
geometry-scale tangents before rank and power are evaluated.  This turns the
earlier fibre audit into one testability object with an explicit nuisance
boundary.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
from collections import defaultdict

import numpy as np


BETA0 = np.asarray((0.22, 0.18, 0.12, -0.60), dtype=float)
FEATURE_NAMES = (
    "quadratic_total",
    "care_midpoint_interaction",
    "intelligence_cloud_interaction",
)
GAP_BINS = ((0.20, 0.35), (0.35, 0.50), (0.50, 0.65),
            (0.65, 0.80), (0.80, 1.01))


def sigmoid(x):
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def candidate_basis(profile):
    mode, intelligence, support, evidence, data, price = profile
    return np.asarray((mode + support, intelligence + evidence, data, price),
                      dtype=float)


def candidate_utility(profile):
    return float(candidate_basis(profile) @ BETA0)


def departure_features(profile):
    mode, intelligence, support, evidence, data, price = profile
    values = np.asarray(profile, dtype=float)
    return np.asarray((
        np.sum(values * values),
        float(mode == 1 and support == 1),
        float(intelligence == 2 and data == 2),
    ), dtype=float)


def motion_signature(a, b):
    delta = np.asarray(a, dtype=int) - np.asarray(b, dtype=int)
    return (int(np.count_nonzero(delta)),
            int(np.sum(delta > 0)),
            int(np.sum(delta < 0)))


def squared_distance(a, b):
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(delta @ delta)


def enumerate_pool():
    groups = defaultdict(list)
    for profile in itertools.product(range(3), repeat=6):
        groups[tuple(candidate_basis(profile))].append(profile)
    pairs = []
    for members in groups.values():
        pairs.extend(itertools.combinations(members, 2))

    rows = []
    for a_pair in pairs:
        for b_pair in pairs:
            a_plus, a_minus = a_pair
            b_plus, b_minus = b_pair
            gap = candidate_utility(a_plus) - candidate_utility(b_plus)
            if not 0.20 <= gap <= 1.00:
                continue
            if motion_signature(a_plus, b_plus) != \
                    motion_signature(a_minus, b_minus):
                continue
            odd = 0.5 * (
                (departure_features(a_plus) - departure_features(b_plus)) -
                (departure_features(a_minus) - departure_features(b_minus))
            )
            if np.max(np.abs(odd)) < 1e-12:
                continue
            rows.append({
                "a_plus": a_plus,
                "a_minus": a_minus,
                "b_plus": b_plus,
                "b_minus": b_minus,
                "gap": float(gap),
                "odd": odd,
                "distance_plus": squared_distance(a_plus, b_plus),
                "distance_minus": squared_distance(a_minus, b_minus),
                "motion_signature": motion_signature(a_plus, b_plus),
            })
    return rows


def prepare_pool():
    rows = enumerate_pool()
    gaps = np.asarray([row["gap"] for row in rows], dtype=float)
    p = sigmoid(gaps)
    variance = p * (1.0 - p)
    departure = np.asarray([row["odd"] for row in rows], dtype=float)
    geometry = np.asarray([
        p[i] * (1.0 - p[i]) * gaps[i] *
        (row["distance_minus"] - row["distance_plus"])
        for i, row in enumerate(rows)
    ])
    # The constant column represents a signed framing/order tangent.  The
    # geometry column represents a declared error-scale tangent.
    nuisance = np.column_stack((np.ones(len(rows)), geometry))
    return rows, gaps, variance, departure, nuisance


def bins(gaps):
    groups = []
    for lo, hi in GAP_BINS:
        ids = np.where((gaps >= lo) & (gaps < hi))[0]
        if len(ids) == 0:
            raise RuntimeError(f"empty candidate-gap bin {lo}:{hi}")
        groups.append(ids)
    return groups


def orthogonal_information(indices, departure, nuisance, variance):
    indices = np.asarray(indices, dtype=int)
    inv_variance = 1.0 / np.maximum(variance[indices], 1e-10)
    nmat = nuisance[indices]
    dmat = departure[indices]
    gram = nmat.T @ (inv_variance[:, None] * nmat)
    projection = nmat @ np.linalg.pinv(gram, rcond=1e-10) @ \
        nmat.T @ np.diag(inv_variance)
    residualizer = np.eye(len(indices)) - projection
    information = dmat.T @ (inv_variance[:, None] * residualizer) @ dmat
    information = 0.5 * (information + information.T)
    eigenvalues = np.linalg.eigvalsh(information)
    return residualizer, information, eigenvalues


def raw_information(indices, departure, variance):
    indices = np.asarray(indices, dtype=int)
    information = departure[indices].T @ \
        (variance[indices, None] * departure[indices])
    information = 0.5 * (information + information.T)
    return information, np.linalg.eigvalsh(information)


def select_design(rows, gaps, variance, departure, nuisance,
                  restarts=500, seed=20261007):
    gap_bins = bins(gaps)
    # The primary design deliberately allows a declared geometry nuisance to
    # vary.  A design that makes the nuisance tangent identically zero would
    # reduce NO-FQC to the unprojected fibre audit and would not test the
    # point of the extension.
    orth_gap_bins = [
        group[np.abs(nuisance[group, 1]) > 0.10] for group in gap_bins
    ]
    if any(len(group) == 0 for group in orth_gap_bins):
        raise RuntimeError("a gap bin has no nonzero nuisance support")
    rng = np.random.default_rng(seed)

    def score(indices):
        _residualizer, _information, eigenvalues = orthogonal_information(
            indices, departure, nuisance, variance)
        return float(eigenvalues[0]), float(np.sum(eigenvalues)), eigenvalues

    best = None
    for _ in range(restarts):
        indices = [int(rng.choice(group)) for group in orth_gap_bins]
        current = score(indices)
        for _sweep in range(3):
            improved = False
            for position, group in enumerate(orth_gap_bins):
                for _trial in range(50):
                    proposal = list(indices)
                    proposal[position] = int(rng.choice(group))
                    if len(set(proposal)) < len(proposal):
                        continue
                    proposed = score(proposal)
                    if (proposed[0], proposed[1]) > (current[0], current[1]):
                        indices, current = proposal, proposed
                        improved = True
            if not improved:
                break
        candidate = (current, tuple(indices))
        if best is None or (candidate[0][0], candidate[0][1]) > \
                (best[0][0], best[0][1]):
            best = candidate

    naive = None
    for _ in range(restarts):
        indices = [int(rng.choice(group)) for group in orth_gap_bins]
        information, eigenvalues = raw_information(indices, departure, variance)
        candidate = (float(eigenvalues[0]), float(np.sum(eigenvalues)),
                     tuple(indices), eigenvalues)
        if naive is None or (candidate[0], candidate[1]) > \
                (naive[0], naive[1]):
            naive = candidate

    orth_indices = best[1]
    residualizer, information, eigenvalues = orthogonal_information(
        orth_indices, departure, nuisance, variance)
    target = departure[np.asarray(orth_indices), 0]
    residual_target = residualizer @ target
    inv_variance = 1.0 / variance[np.asarray(orth_indices)]
    orth_weights = inv_variance * residual_target
    orth_weights /= np.sqrt(max(residual_target @
                                (inv_variance * residual_target), 1e-12))

    naive_indices = naive[2]
    naive_target = departure[np.asarray(naive_indices), 0]
    naive_weights = variance[np.asarray(naive_indices)] * naive_target
    naive_weights /= np.sqrt(max(naive_target @
                                 (variance[np.asarray(naive_indices)] *
                                  naive_target), 1e-12))
    return {
        "orth_indices": orth_indices,
        "orth_weights": orth_weights,
        "orth_information": information,
        "orth_eigenvalues": eigenvalues,
        "naive_indices": naive_indices,
        "naive_weights": naive_weights,
        "naive_eigenvalues": naive[3],
    }


def task_probabilities(row, condition, eta=0.0, scale_strength=0.0,
                       framing=0.0):
    plus_a = departure_features(row["a_plus"])
    plus_b = departure_features(row["b_plus"])
    minus_a = departure_features(row["a_minus"])
    minus_b = departure_features(row["b_minus"])
    h_plus = float(plus_a[0] - plus_b[0])
    h_minus = float(minus_a[0] - minus_b[0])
    gap = row["gap"]
    gap_plus = gap + eta * h_plus
    gap_minus = gap + eta * h_minus
    if condition in {"scale", "combined"}:
        gap_plus /= 1.0 + scale_strength * row["distance_plus"]
        gap_minus /= 1.0 + scale_strength * row["distance_minus"]
    elif condition == "unlisted_complexity":
        gap_plus /= 1.0 + scale_strength * abs(h_plus)
        gap_minus /= 1.0 + scale_strength * abs(h_minus)
    if condition == "framing":
        gap_plus += framing
        gap_minus -= framing
    elif condition == "geometry_framing":
        delta = row["distance_minus"] - row["distance_plus"]
        gap_plus += framing * delta / 2.0
        gap_minus -= framing * delta / 2.0
    return float(sigmoid(gap_plus)), float(sigmoid(gap_minus))


def randomization_pvalue(scores, reps, seed):
    observed = abs(float(np.mean(scores)))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray((-1.0, 1.0)),
                       size=(reps, len(scores)))
    draws = np.abs((signs @ scores) / len(scores))
    return float((1.0 + np.sum(draws >= observed)) / (reps + 1.0))


def evaluate(rows, design, condition, respondents, permutation_reps, seed,
             departure_strength=0.10):
    rng = np.random.default_rng(seed)
    eta = departure_strength if condition in {"omitted", "combined"} else 0.0
    scale_strength = 0.50 if condition in {"scale", "combined"} else 0.20 \
        if condition == "unlisted_complexity" else 0.0
    framing = 0.20 if condition == "framing" else \
        0.50 if condition == "geometry_framing" else 0.0
    orth_indices = design["orth_indices"]
    naive_indices = design["naive_indices"]
    orth_weights = design["orth_weights"]
    naive_weights = design["naive_weights"]
    orth_p0 = np.asarray([sigmoid(rows[i]["gap"]) for i in orth_indices])
    naive_p0 = np.asarray([sigmoid(rows[i]["gap"]) for i in naive_indices])
    orth_plus = [task_probabilities(rows[i], condition, eta, scale_strength,
                                    framing) for i in orth_indices]
    naive_plus = [task_probabilities(rows[i], condition, eta, scale_strength,
                                      framing) for i in naive_indices]
    orth_scores = np.zeros(respondents)
    naive_scores = np.zeros(respondents)
    for respondent in range(respondents):
        orientation = 1 if rng.integers(0, 2) else -1
        orth_probs = np.asarray([x[0] if orientation > 0 else x[1]
                                 for x in orth_plus])
        naive_probs = np.asarray([x[0] if orientation > 0 else x[1]
                                  for x in naive_plus])
        orth_y = rng.binomial(1, orth_probs)
        naive_y = rng.binomial(1, naive_probs)
        sign = float(orientation)
        orth_scores[respondent] = sign * (orth_weights @
                                          (orth_y - orth_p0))
        naive_scores[respondent] = sign * (naive_weights @
                                            (naive_y - naive_p0))
    orth_p = randomization_pvalue(orth_scores, permutation_reps, seed + 11)
    naive_p = randomization_pvalue(naive_scores, permutation_reps, seed + 23)
    return int(orth_p < 0.05), int(naive_p < 0.05), orth_p, naive_p


def write_design(path, rows, design):
    records = []
    for name, indices in (("nuisance_orthogonal", design["orth_indices"]),
                          ("raw_signal", design["naive_indices"])):
        for order, index in enumerate(indices, 1):
            row = rows[index]
            records.append({
                "design": name,
                "task": order,
                "pool_index": index,
                "candidate_gap": row["gap"],
                "distance_plus": row["distance_plus"],
                "distance_minus": row["distance_minus"],
                "geometry_delta": row["distance_minus"] -
                row["distance_plus"],
                "motion_signature": row["motion_signature"],
                **{f"odd_{name}": float(value) for name, value in
                   zip(FEATURE_NAMES, row["odd"])},
            })
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def run(reps=200, respondents=600, permutation_reps=499,
        design_out="results/no_fqc_structural_design.csv",
        benchmark_out="results/no_fqc_structural_benchmark_200rep.csv",
        departure_strength=0.10):
    rows, gaps, variance, departure, nuisance = prepare_pool()
    design = select_design(rows, gaps, variance, departure, nuisance)
    write_design(design_out, rows, design)
    print("pool_size", len(rows))
    print("orthogonal_indices", design["orth_indices"])
    print("raw_signal_indices", design["naive_indices"])
    print("orthogonal_eigenvalues",
          np.round(design["orth_eigenvalues"], 6))
    print("raw_signal_eigenvalues",
          np.round(design["naive_eigenvalues"], 6))
    conditions = ("null", "omitted", "scale", "combined", "framing",
                  "unlisted_complexity", "geometry_framing")
    records = []
    for condition in conditions:
        for rep in range(reps):
            orth, naive, p_orth, p_naive = evaluate(
                rows, design, condition, respondents, permutation_reps,
                82000000 + rep,
                departure_strength=departure_strength)
            records.extend((
                {"condition": condition, "rep": rep,
                 "diagnostic": "nuisance_orthogonal", "reject": orth,
                 "pvalue": p_orth},
                {"condition": condition, "rep": rep,
                 "diagnostic": "raw_signal", "reject": naive,
                 "pvalue": p_naive},
            ))
    os.makedirs(os.path.dirname(benchmark_out) or ".", exist_ok=True)
    with open(benchmark_out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    for condition in conditions:
        for diagnostic in ("nuisance_orthogonal", "raw_signal"):
            subset = [x for x in records if x["condition"] == condition and
                      x["diagnostic"] == diagnostic]
            print(condition, diagnostic, "rejection_rate",
                  round(float(np.mean([x["reject"] for x in subset])), 3))
    print("wrote", len(records), "rows to", benchmark_out)
    print("departure_strength", departure_strength)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--respondents", type=int, default=600)
    parser.add_argument("--permutation-reps", type=int, default=499)
    parser.add_argument("--design-out",
                        default="results/no_fqc_structural_design.csv")
    parser.add_argument("--benchmark-out",
                        default="results/no_fqc_structural_benchmark_200rep.csv")
    parser.add_argument("--departure-strength", type=float, default=0.10)
    args = parser.parse_args()
    run(**vars(args))
