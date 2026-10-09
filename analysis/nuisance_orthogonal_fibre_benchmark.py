"""Planning benchmark for nuisance-orthogonal utility-fibre contrasts.

The candidate observes only the composite burden s=a1+a2. Candidate-preserving
task pairs change the raw decomposition z while keeping the candidate gap
fixed. A design contrast is constructed by projecting the omitted-utility
signal away from declared nuisance tangents (a constant framing/order tangent
and a geometry-dependent scale tangent). This is a proof-of-concept for the
round-eight innovation audit, not a final estimator.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os

import numpy as np


def sigmoid(x):
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def raw_from_s_z(s, z):
    return np.column_stack(((s + z) / 2.0, (s - z) / 2.0))


def distance(a, b):
    # An L1 distance is constant for the present fixed-sum construction:
    # changing the decomposition while preserving s leaves the absolute
    # coordinate gap unchanged.  Squared Euclidean distance supplies a
    # non-degenerate geometry index for the scale nuisance tangent and keeps
    # the contrast sensitive to tradeoff structure.
    return np.sum((a - b) ** 2, axis=-1)


def arm_grid():
    arms = []
    for u in (-45.0, -15.0, 15.0, 45.0, 75.0):
        for g in (0.20, 0.40, 0.60, 0.80, 1.00):
            s_a = 90.0 + u / 2.0
            s_b = 90.0 - u / 2.0
            z_a0, z_b0 = 0.25, 2.00
            z_a1, z_b1 = z_a0 + g, z_b0 - g
            a0 = raw_from_s_z(np.array([s_a]), np.array([z_a0]))[0]
            b0 = raw_from_s_z(np.array([s_b]), np.array([z_b0]))[0]
            a1 = raw_from_s_z(np.array([s_a]), np.array([z_a1]))[0]
            b1 = raw_from_s_z(np.array([s_b]), np.array([z_b1]))[0]
            h0 = z_a0 ** 2 - z_b0 ** 2
            h1 = z_a1 ** 2 - z_b1 ** 2
            d0 = float(distance(a0[None, :], b0[None, :])[0])
            d1 = float(distance(a1[None, :], b1[None, :])[0])
            arms.append({
                "u": u, "g": g, "a0": a0, "b0": b0, "a1": a1, "b1": b1,
                "h0": h0, "h1": h1, "d0": d0, "d1": d1,
            })
    return arms


def arm_tangents(arms, beta=0.030):
    signal = []
    scale = []
    for arm in arms:
        u = arm["u"]
        h_delta = arm["h1"] - arm["h0"]
        p0 = sigmoid(beta * u)
        deriv = p0 * (1.0 - p0)
        # Local derivative of the pair probability contrast under an omitted
        # decomposition term eta*(z_A^2-z_B^2).
        signal.append(deriv * h_delta)
        # Local derivative under scale = 1 + lambda * raw distance.
        scale.append(-deriv * beta * u * (arm["d1"] - arm["d0"]))
    signal = np.asarray(signal)
    scale = np.asarray(scale)
    # A constant tangent covers a framing/order shift shared across focal arms.
    nuisance = np.column_stack((np.ones(len(arms)), scale))
    variance = np.asarray([p0 * (1.0 - p0) for p0 in
                           sigmoid(beta * np.asarray([a["u"] for a in arms]))])
    return signal, nuisance, variance


def project_signal(signal, nuisance, variance):
    # The inverse Bernoulli variance gives a locally efficient metric. The
    # projection removes the declared nuisance tangent before the signal is
    # converted to an estimating contrast.
    inv_var = 1.0 / np.maximum(np.asarray(variance), 1e-8)
    gram = nuisance.T @ (inv_var[:, None] * nuisance)
    pinv = np.linalg.pinv(gram, rcond=1e-10)
    projection = nuisance @ pinv @ nuisance.T @ np.diag(inv_var)
    residual = signal - projection @ signal
    weights = inv_var * residual
    weights = weights / max(np.linalg.norm(weights), 1e-12)
    return residual, weights


def select_design(arms, size=5):
    signal, nuisance, variance = arm_tangents(arms)
    best = None
    naive_candidates = []
    for idx in itertools.combinations(range(len(arms)), size):
        idx = np.asarray(idx, dtype=int)
        residual, weights = project_signal(signal[idx], nuisance[idx], variance[idx])
        score = float(np.sqrt(np.maximum(residual @
                                         (weights / max(np.linalg.norm(weights), 1e-12)), 0.0)))
        naive_signal = signal[idx]
        naive_score = float(np.linalg.norm(naive_signal))
        naive_candidates.append((naive_score, idx))
        candidate = (score, idx, weights)
        if best is None or candidate[0] > best[0]:
            best = candidate
    naive_idx = max(naive_candidates, key=lambda x: x[0])[1]
    _, orth_idx, orth_weights = best
    naive_weights = signal[naive_idx]
    naive_weights = naive_weights / max(np.linalg.norm(naive_weights), 1e-12)
    return {
        "orth_idx": orth_idx,
        "orth_weights": orth_weights,
        "naive_idx": naive_idx,
        "naive_weights": naive_weights,
        "signal": signal,
        "nuisance": nuisance,
        "variance": variance,
        "orth_residual_norm": float(np.linalg.norm(
            project_signal(signal[orth_idx], nuisance[orth_idx], variance[orth_idx])[0])),
        "naive_signal_norm": float(np.linalg.norm(signal[naive_idx])),
    }


def probabilities(arms, idx, condition, eta=0.0, scale_strength=0.0,
                  framing=0.0, nonlinear_scale=False, complexity_scale=False,
                  complexity_interaction=False,
                  geometry_framing=False,
                  orientation=None):
    out0, out1 = [], []
    for i in idx:
        arm = arms[int(i)]
        gap0 = 0.030 * arm["u"] + eta * arm["h0"] + framing
        gap1 = 0.030 * arm["u"] + eta * arm["h1"] + framing
        if scale_strength:
            if complexity_scale:
                # A deliberately unlisted tradeoff-complexity nuisance: the
                # attenuation depends on the decomposition contrast itself,
                # not on the declared geometric distance tangent.
                d0, d1 = abs(arm["h0"]), abs(arm["h1"])
                if complexity_interaction:
                    gap_factor = 1.0 + 0.45 * arm["u"] / 75.0
                    d0, d1 = gap_factor * d0, gap_factor * d1
            else:
                d0 = arm["d0"] ** 2 if nonlinear_scale else arm["d0"]
                d1 = arm["d1"] ** 2 if nonlinear_scale else arm["d1"]
            gap0 /= 1.0 + scale_strength * d0
            gap1 /= 1.0 + scale_strength * d1
        if geometry_framing:
            # Out-of-library interaction: presentation framing changes with
            # the pair's geometry.  It is distinct from a shared intercept
            # and from the declared multiplicative scale tangent.
            delta_d = arm["d1"] - arm["d0"]
            gap0 -= framing * delta_d / 2.0
            gap1 += framing * delta_d / 2.0
        out0.append(float(sigmoid(gap0)))
        out1.append(float(sigmoid(gap1)))
    return np.asarray(out0), np.asarray(out1)


def randomization_pvalue(cluster_scores, reps, rng):
    observed = abs(float(np.mean(cluster_scores)))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(reps, len(cluster_scores)))
    draws = np.abs((signs @ cluster_scores) / len(cluster_scores))
    return float((1.0 + np.sum(draws >= observed)) / (reps + 1.0))


def evaluate(arms, design, condition, respondents, permutation_reps, seed):
    rng = np.random.default_rng(seed)
    idx = design["orth_idx"]
    weights = design["orth_weights"]
    # Comparators use the same arms and respondent budget. The naive contrast
    # maximizes the raw omitted signal and ignores nuisance projection.
    naive_idx = design["naive_idx"]
    naive_weights = design["naive_weights"]
    nonlinear_scale = False
    complexity_scale = False
    complexity_interaction = False
    geometry_framing = False
    if condition == "omitted":
        eta, scale_strength, framing = 0.10, 0.0, 0.0
    elif condition == "scale":
        eta, scale_strength, framing = 0.0, 0.005, 0.0
    elif condition == "combined":
        eta, scale_strength, framing = 0.10, 0.005, 0.0
    elif condition == "framing":
        eta, scale_strength, framing = 0.0, 0.0, 0.10
    elif condition == "nonlinear_scale":
        eta, scale_strength, framing, nonlinear_scale = 0.0, 0.0008, 0.0, True
    elif condition == "unlisted_complexity":
        eta, scale_strength, framing, complexity_scale = 0.0, 0.20, 0.0, True
    elif condition == "out_of_library_complexity":
        eta, scale_strength, framing = 0.0, 0.20, 0.0
        complexity_scale, complexity_interaction = True, True
    elif condition == "geometry_framing":
        eta, scale_strength, framing = 0.0, 0.0, 0.50
        geometry_framing = True
    else:
        eta, scale_strength, framing = 0.0, 0.0, 0.0
    p0, p1 = probabilities(arms, idx, condition, eta, scale_strength, framing,
                            nonlinear_scale, complexity_scale,
                            complexity_interaction, geometry_framing)
    q0, q1 = probabilities(arms, naive_idx, condition, eta, scale_strength,
                            framing, nonlinear_scale, complexity_scale,
                            complexity_interaction, geometry_framing)
    cluster_orth = np.zeros(respondents)
    cluster_naive = np.zeros(respondents)
    for rid in range(respondents):
        orientation = int(rng.integers(0, 2))
        y_orth = rng.binomial(1, p1 if orientation else p0)
        y_naive = rng.binomial(1, q1 if orientation else q0)
        p_orth = (p1 if orientation else p0)
        p_naive = (q1 if orientation else q0)
        sign = 1.0 if orientation else -1.0
        cluster_orth[rid] = sign * float(weights @ (y_orth - sigmoid(0.030 * np.asarray(
            [arms[int(i)]["u"] for i in idx]))))
        cluster_naive[rid] = sign * float(naive_weights @ (y_naive - sigmoid(0.030 * np.asarray(
            [arms[int(i)]["u"] for i in naive_idx]))))
    p_orth = randomization_pvalue(cluster_orth, permutation_reps,
                                   np.random.default_rng(seed + 101))
    p_naive = randomization_pvalue(cluster_naive, permutation_reps,
                                    np.random.default_rng(seed + 202))
    return int(p_orth < 0.05), int(p_naive < 0.05), p_orth, p_naive


def run(reps=100, respondents=600, permutation_reps=199,
        out="results/nuisance_orthogonal_fibre_benchmark.csv"):
    arms = arm_grid()
    design = select_design(arms)
    print("orthogonal indices", design["orth_idx"].tolist())
    print("naive indices", design["naive_idx"].tolist())
    print("orthogonal residual norm", round(design["orth_residual_norm"], 6))
    orth_nuisance = design["nuisance"][design["orth_idx"]].T @ design["orth_weights"]
    print("max nuisance projection", round(float(np.max(np.abs(orth_nuisance))), 10))
    conditions = ("null", "omitted", "scale", "framing", "nonlinear_scale",
                  "unlisted_complexity", "out_of_library_complexity",
                  "geometry_framing", "combined")
    rows = []
    for condition in conditions:
        for rep in range(reps):
            orth, naive, p_orth, p_naive = evaluate(
                arms, design, condition, respondents, permutation_reps,
                61000000 + rep)
            rows.append({
                "condition": condition, "rep": rep,
                "diagnostic": "nuisance_orthogonal",
                "reject": orth, "pvalue": p_orth,
            })
            rows.append({
                "condition": condition, "rep": rep,
                "diagnostic": "naive_signal_max",
                "reject": naive, "pvalue": p_naive,
            })
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for condition in conditions:
        for diagnostic in ("nuisance_orthogonal", "naive_signal_max"):
            subset = [r for r in rows if r["condition"] == condition and
                      r["diagnostic"] == diagnostic]
            print(condition, diagnostic, "rejection_rate",
                  round(float(np.mean([r["reject"] for r in subset])), 3))
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--respondents", type=int, default=600)
    parser.add_argument("--permutation-reps", type=int, default=199)
    parser.add_argument("--out", default="results/nuisance_orthogonal_fibre_benchmark.csv")
    run(**vars(parser.parse_args()))
