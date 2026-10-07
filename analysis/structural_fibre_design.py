"""Coefficient-robust candidate-fibre construction and calibration audit.

The earlier fibre generator equated a scalar utility value at one coefficient
vector.  This module instead equates a coarsened candidate sufficient-statistic
vector.  The equality then holds for every coefficient vector in the
candidate model, so coefficient estimation error cannot create a plus/minus
gap by itself.

The task block is selected with the same maximin falsification criterion as
the coefficient-calibrated proof of concept, using a randomized exchange
search because the structural fibre is larger.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from collections import OrderedDict, defaultdict

import numpy as np

HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import smart_device_fibre_design as base


BETA0 = np.asarray((0.22, 0.18, 0.12, -0.60), dtype=float)
BASIS_NAMES = ("care_intensity", "clinical_quality", "data_level", "price_level")
FEATURE_NAMES = (
    "quadratic_total", "care_midpoint_interaction",
    "intelligence_cloud_interaction",
)
GAP_BINS = ((0.20, 0.35), (0.35, 0.50), (0.50, 0.65),
            (0.65, 0.80), (0.80, 1.01))


def candidate_basis(profile):
    mode, intelligence, support, evidence, data, price = profile
    # A deliberately coarsened candidate basis.  It represents care intensity
    # and clinical quality as composite indices; raw decomposition is left
    # available for falsification.
    return np.asarray((mode + support, intelligence + evidence, data, price),
                      dtype=float)


def candidate_utility(profile, beta=BETA0):
    return float(candidate_basis(profile) @ np.asarray(beta, dtype=float))


def departure_features(profile):
    mode, intelligence, support, evidence, data, price = profile
    x = np.asarray(profile, dtype=float)
    return np.asarray((
        np.sum(x * x),
        # The end-point mode/support interaction is structurally constant on
        # the care-intensity fibre. The midpoint interaction is observable and
        # provides a meaningful decomposition departure instead.
        float(int(mode == 1 and support == 1)),
        float(int(intelligence == 2 and data == 2)),
    ), dtype=float)


def omitted_interaction(profile):
    return base.omitted_interaction(profile)


def raw_signature(a, b):
    delta = np.asarray(a, dtype=int) - np.asarray(b, dtype=int)
    return (int(np.sum(delta * delta)), int(np.count_nonzero(delta)),
            int(np.sum(np.abs(delta))), int(np.sum(delta > 0)),
            int(np.sum(delta < 0)))


def enumerate_pairs():
    groups = defaultdict(list)
    profiles = list(itertools.product(range(3), repeat=6))
    for profile in profiles:
        groups[tuple(candidate_basis(profile))].append(
            (profile, omitted_interaction(profile)))
    pairs = []
    for stat, members in groups.items():
        for (plus, h_plus), (minus, h_minus) in itertools.combinations(members, 2):
            if abs(h_plus - h_minus) < 0.2:
                continue
            pairs.append({"basis": stat, "plus": plus, "minus": minus,
                          "h_plus": h_plus, "h_minus": h_minus})
    return pairs


def enumerate_reflections(beta=BETA0):
    pairs = enumerate_pairs()
    rows = []
    for a_pair in pairs:
        for b_pair in pairs:
            a_plus, a_minus = a_pair["plus"], a_pair["minus"]
            b_plus, b_minus = b_pair["plus"], b_pair["minus"]
            gap = candidate_utility(a_plus, beta) - candidate_utility(b_plus, beta)
            if not 0.20 <= gap <= 1.0:
                continue
            odd_a = a_pair["h_plus"] - a_pair["h_minus"]
            odd_b = b_pair["h_plus"] - b_pair["h_minus"]
            if abs(odd_a + odd_b) > 1e-12 or abs(odd_a - odd_b) < 0.5:
                continue
            if sum((x - y) ** 2 for x, y in zip(a_plus, b_plus)) != \
                    sum((x - y) ** 2 for x, y in zip(a_minus, b_minus)):
                continue
            if raw_signature(a_plus, b_plus) != raw_signature(a_minus, b_minus):
                continue
            rows.append({
                "a_plus": a_plus, "a_minus": a_minus,
                "b_plus": b_plus, "b_minus": b_minus,
                "basis_A": a_pair["basis"], "basis_B": b_pair["basis"],
                "gap": float(gap), "odd_interaction": float((odd_a - odd_b) / 2.0),
                "distance": int(sum((x - y) ** 2 for x, y in zip(a_plus, b_plus))),
                "hamming": int(sum(x != y for x, y in zip(a_plus, b_plus))),
                "signature": raw_signature(a_plus, b_plus),
            })
    unique = []
    seen = set()
    for row in rows:
        key = (row["a_plus"], row["a_minus"], row["b_plus"], row["b_minus"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def pair_odd_features(row):
    plus = departure_features(row["a_plus"]) - departure_features(row["b_plus"])
    minus = departure_features(row["a_minus"]) - departure_features(row["b_minus"])
    return 0.5 * (plus - minus)


def prepare_pool():
    pool = enumerate_reflections()
    raw = np.asarray([pair_odd_features(row) for row in pool], dtype=float)
    gap = np.asarray([row["gap"] for row in pool], dtype=float)
    p = 1.0 / (1.0 + np.exp(-gap))
    weighted = np.sqrt(p * (1.0 - p))[:, None] * raw
    scales = np.sqrt(np.mean(weighted * weighted, axis=0))
    scales[scales < 1e-12] = 1.0
    return pool, raw, weighted / scales, scales


def objective(indices, matrix):
    info = matrix[np.asarray(indices, dtype=int)].T @ matrix[np.asarray(indices, dtype=int)]
    eig = np.linalg.eigvalsh(info)
    return float(eig[0]), float(np.trace(info)), eig


def bins(pool):
    result = []
    for lo, hi in GAP_BINS:
        ids = [i for i, row in enumerate(pool) if lo <= row["gap"] < hi]
        if not ids:
            raise RuntimeError(f"empty structural fibre gap bin {lo}:{hi}")
        result.append(ids)
    return result


def random_exchange_maximin(pool, matrix, restarts=400, sweeps=8, seed=20261007):
    """Scalable exchange search over a structural fibre."""
    rng = np.random.default_rng(seed)
    pools = bins(pool)
    best = None
    for _ in range(restarts):
        current = [int(rng.choice(group)) for group in pools]
        score = objective(current, matrix)
        for _sweep in range(sweeps):
            improved = False
            for position in rng.permutation(len(pools)):
                candidates = pools[position]
                order = rng.permutation(len(candidates))
                local_best = score
                local_index = current[position]
                for candidate_position in order[: min(80, len(order))]:
                    candidate = candidates[int(candidate_position)]
                    trial = list(current)
                    trial[position] = candidate
                    if len(set(trial)) < len(trial):
                        continue
                    trial_score = objective(trial, matrix)
                    if (trial_score[0], trial_score[1]) > (local_best[0], local_best[1]):
                        local_best = trial_score
                        local_index = candidate
                if local_index != current[position]:
                    current[position] = local_index
                    score = local_best
                    improved = True
            if not improved:
                break
        candidate = {"indices": tuple(current), "minimum": score[0],
                     "trace": score[1], "eigenvalues": score[2]}
        if best is None or (candidate["minimum"], candidate["trace"]) > \
                (best["minimum"], best["trace"]):
            best = candidate
    best["restarts"] = restarts
    return best


def choose_random_design(pool, seed=20261009):
    rng = np.random.default_rng(seed)
    return tuple(int(rng.choice(group)) for group in bins(pool))


def sign_flip_pvalue(z, variances, permutation_reps, rng):
    n = z.shape[0]
    mean = np.mean(z, axis=0)
    observed = float(n * np.sum(mean * mean / variances))
    signs = rng.choice(np.asarray((-1.0, 1.0)), size=(permutation_reps, n))
    perm = (signs @ z) / n
    statistic = n * np.sum(perm * perm / variances[None, :], axis=1)
    return float((1.0 + np.sum(statistic >= observed)) /
                 (permutation_reps + 1.0))


def simulate_null(pool, indices, beta_sd, respondents, permutation_reps, seed):
    rng = np.random.default_rng(seed)
    beta = BETA0 + rng.normal(0.0, beta_sd, len(BETA0))
    rows = [pool[i] for i in indices]
    gaps = np.asarray([candidate_utility(row["a_plus"], beta) -
                       candidate_utility(row["b_plus"], beta) for row in rows])
    p0 = 1.0 / (1.0 + np.exp(-gaps))
    orientation = rng.choice(np.asarray((-1.0, 1.0)), size=respondents)
    y = rng.binomial(1, p0[None, :], size=(respondents, len(indices)))
    z = orientation[:, None] * (y - p0[None, :])
    pvalue = sign_flip_pvalue(z, p0 * (1 - p0), permutation_reps,
                              np.random.default_rng(seed + 991))
    return int(pvalue < 0.05), pvalue


def simulate_departure(pool, raw, indices, direction, beta_sd, respondents,
                       permutation_reps, seed, eta=0.30):
    rng = np.random.default_rng(seed)
    beta = BETA0 + rng.normal(0.0, beta_sd, len(BETA0))
    rows = [pool[i] for i in indices]
    gaps = np.asarray([candidate_utility(row["a_plus"], beta) -
                       candidate_utility(row["b_plus"], beta) for row in rows])
    p0 = 1.0 / (1.0 + np.exp(-gaps))
    odd = raw[np.asarray(indices, dtype=int)] @ np.asarray(direction, dtype=float)
    orientation = rng.choice(np.asarray((-1.0, 1.0)), size=respondents)
    probabilities = 1.0 / (1.0 + np.exp(-(
        gaps[None, :] + eta * orientation[:, None] * odd[None, :])))
    y = rng.binomial(1, probabilities)
    z = orientation[:, None] * (y - p0[None, :])
    pvalue = sign_flip_pvalue(z, p0 * (1 - p0), permutation_reps,
                              np.random.default_rng(seed + 1991))
    return int(pvalue < 0.05), pvalue


def write_design(out, pool, designs):
    rows = []
    for design_name, indices, best in designs:
      for task, index in enumerate(indices, 1):
        row = pool[index]
        minimum = "" if best is None else best["minimum"]
        trace = "" if best is None else best["trace"]
        entry = OrderedDict(design=design_name, task=task,
                            pool_index=index, candidate_gap=row["gap"],
                            candidate_basis_A=str(tuple(int(x) for x in row["basis_A"])),
                            candidate_basis_B=str(tuple(int(x) for x in row["basis_B"])),
                            distance=row["distance"], hamming=row["hamming"],
                            raw_signature=str(row["signature"]),
                            candidate_gap_mismatch_at_beta0=0.0,
                            min_eigenvalue=minimum,
                            information_trace=trace)
        for name, value in zip(FEATURE_NAMES, pair_odd_features(row)):
            entry[f"odd_{name}"] = float(value)
        rows.append(entry)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]),
                                lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def run(reps=100, respondents=600, permutation_reps=199,
        out="results/structural_fibre_design.csv",
        benchmark_out="results/structural_fibre_calibration_benchmark.csv",
        power_out="results/structural_fibre_power_benchmark.csv"):
    pool, raw, matrix, scales = prepare_pool()
    best = random_exchange_maximin(pool, matrix)
    random_indices = choose_random_design(pool)
    write_design(out, pool, [("structural_maximin", best["indices"], best),
                             ("structural_random", random_indices, None)])
    rows = []
    for design_name, indices in (("structural_maximin", best["indices"]),
                                 ("structural_random", random_indices)):
      for beta_sd in (0.0, 0.02, 0.05, 0.10):
        for rep in range(reps):
            reject, pvalue = simulate_null(
                pool, indices, beta_sd, respondents, permutation_reps,
                71000000 + rep + int(beta_sd * 10000) +
                (0 if design_name == "structural_maximin" else 1000000))
            rows.append({"design": design_name, "beta_sd": beta_sd,
                         "rep": rep, "reject": reject, "pvalue": pvalue,
                         "respondents": respondents,
                         "permutation_reps": permutation_reps})
    os.makedirs(os.path.dirname(benchmark_out) or ".", exist_ok=True)
    with open(benchmark_out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]),
                                lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    rng = np.random.default_rng(20261008)
    directions = [("feature_" + name, np.eye(len(FEATURE_NAMES))[j])
                  for j, name in enumerate(FEATURE_NAMES)]
    for j in range(10):
        direction = rng.normal(size=len(FEATURE_NAMES))
        direction /= np.linalg.norm(direction)
        directions.append((f"random_{j}", direction))
    power_rows = []
    for design_name, indices in (("structural_maximin", best["indices"]),
                                 ("structural_random", random_indices)):
      for beta_sd in (0.0, 0.05):
        for direction_name, direction in directions:
            for rep in range(reps):
                reject, pvalue = simulate_departure(
                    pool, raw, indices, direction, beta_sd, respondents,
                    permutation_reps, 72000000 + rep + int(beta_sd * 10000) +
                    (0 if design_name == "structural_maximin" else 1000000))
                power_rows.append({"design": design_name,
                                   "direction": direction_name,
                                   "beta_sd": beta_sd, "rep": rep,
                                   "reject": reject, "pvalue": pvalue,
                                   "respondents": respondents,
                                   "permutation_reps": permutation_reps})
    os.makedirs(os.path.dirname(power_out) or ".", exist_ok=True)
    with open(power_out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(power_rows[0]),
                                lineterminator="\n")
        writer.writeheader(); writer.writerows(power_rows)
    print("basis_groups", len(set(tuple(row["basis_A"]) for row in pool)))
    print("structural_pool", len(pool))
    print("maximin_indices", list(best["indices"]))
    print("selected_gaps", [round(pool[i]["gap"], 3) for i in best["indices"]])
    print("eigenvalues", [round(float(x), 6) for x in best["eigenvalues"]])
    print("departure_scales", [round(float(x), 6) for x in scales])
    for design_name in ("structural_maximin", "structural_random"):
      for beta_sd in (0.0, 0.02, 0.05, 0.10):
        subset = [row for row in rows if row["design"] == design_name
                  and row["beta_sd"] == beta_sd]
        print(design_name, "beta_sd", beta_sd, "null_rejection",
              round(float(np.mean([row["reject"] for row in subset])), 3))
      for beta_sd in (0.0, 0.05):
        subset = [row for row in power_rows if row["design"] == design_name
                  and row["beta_sd"] == beta_sd]
        feature = [row for row in subset if row["direction"].startswith("feature_")]
        print(design_name, "beta_sd", beta_sd, "feature_power_min",
              round(min(float(np.mean([r["reject"] for r in feature
                                       if r["direction"] == name]))
                        for name in ["feature_" + x for x in FEATURE_NAMES]), 3))
    print("wrote", out, benchmark_out, power_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--respondents", type=int, default=600)
    parser.add_argument("--permutation-reps", type=int, default=199)
    parser.add_argument("--out", default="results/structural_fibre_design.csv")
    parser.add_argument("--benchmark-out", default="results/structural_fibre_calibration_benchmark.csv")
    parser.add_argument("--power-out", default="results/structural_fibre_power_benchmark.csv")
    run(**vars(parser.parse_args()))
