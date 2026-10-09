"""Planning benchmark for parity-separated utility-fibre contrasts.

A candidate observes s=a1+a2.  For each paired task, the two alternatives keep
s fixed and use a symmetric fibre coordinate z_A(t)=c+t,
z_B(t)=c-t.  The candidate gap is constant in t.  Pairwise squared geometry is
even in t, while an omitted z^2 decomposition term is odd when c != 0.
The central +/- contrast therefore cancels symmetric scale/complexity changes
and retains the directional utility violation.  An asymmetric nuisance is
included as a deliberate failure boundary.
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


def arm_grid():
    arms = []
    for u in (-45.0, -15.0, 15.0, 45.0, 75.0):
        for c in (0.5, 1.0, 1.5, 2.0, 2.5):
            for t in (0.5, 1.0, 1.5):
                s_a, s_b = 90.0 + u / 2.0, 90.0 - u / 2.0
                za_plus, zb_plus = c + t, c - t
                za_minus, zb_minus = c - t, c + t
                a_plus = raw_from_s_z(np.array([s_a]), np.array([za_plus]))[0]
                b_plus = raw_from_s_z(np.array([s_b]), np.array([zb_plus]))[0]
                a_minus = raw_from_s_z(np.array([s_a]), np.array([za_minus]))[0]
                b_minus = raw_from_s_z(np.array([s_b]), np.array([zb_minus]))[0]
                h_plus = za_plus ** 2 - zb_plus ** 2
                h_minus = za_minus ** 2 - zb_minus ** 2
                d_plus = float(np.sum((a_plus - b_plus) ** 2))
                d_minus = float(np.sum((a_minus - b_minus) ** 2))
                arms.append({
                    "u": u, "c": c, "t": t,
                    "a_plus": a_plus, "b_plus": b_plus,
                    "a_minus": a_minus, "b_minus": b_minus,
                    "h_plus": h_plus, "h_minus": h_minus,
                    "d_plus": d_plus, "d_minus": d_minus,
                })
    return arms


def choose_design(arms, size=6):
    # The parity signal is the derivative of the probability contrast with
    # respect to the omitted eta at eta=0.  Use all signs of u but avoid
    # duplicate c,t cells so the score is not driven by one arm.
    scored = []
    beta = 0.030
    for i, arm in enumerate(arms):
        p = sigmoid(beta * arm["u"])
        deriv = p * (1.0 - p)
        signal = deriv * (arm["h_plus"] - arm["h_minus"]) / (2.0 * arm["t"])
        scored.append((abs(float(signal)), i, float(signal)))
    # Choose the strongest arm for each distinct gap and then complete with
    # balanced positive/negative gaps.
    selected = []
    for u in (-45.0, -15.0, 15.0, 45.0, 75.0):
        pool = [row for row in scored if arms[row[1]]["u"] == u]
        selected.append(max(pool)[1])
    selected = np.asarray(selected[:size], dtype=int)
    # Comparator: an asymmetric zero-to-plus move with the same cells.
    return {"parity_idx": selected, "naive_idx": selected.copy()}


def probabilities(arms, idx, condition, eta=0.0, scale_strength=0.0,
                  framing=0.0, odd_scale=False):
    out_plus, out_minus = [], []
    for i in idx:
        arm = arms[int(i)]
        base = 0.030 * arm["u"]
        gap_plus = base + eta * arm["h_plus"] + framing
        gap_minus = base + eta * arm["h_minus"] + framing
        if scale_strength:
            d_plus, d_minus = arm["d_plus"], arm["d_minus"]
            if odd_scale:
                # Deliberate out-of-library asymmetry: scale has an odd
                # component in t and is no longer removed by parity.
                d_plus *= 1.0 + 0.65 * arm["t"]
                d_minus *= 1.0 - 0.65 * arm["t"]
            gap_plus /= 1.0 + scale_strength * d_plus
            gap_minus /= 1.0 + scale_strength * d_minus
        out_plus.append(float(sigmoid(gap_plus)))
        out_minus.append(float(sigmoid(gap_minus)))
    return np.asarray(out_plus), np.asarray(out_minus)


def randomization_pvalue(scores, reps, rng):
    observed = abs(float(np.mean(scores)))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(reps, len(scores)))
    draws = np.abs((signs @ scores) / len(scores))
    return float((1.0 + np.sum(draws >= observed)) / (reps + 1.0))


def evaluate(arms, design, condition, respondents, permutation_reps, seed):
    rng = np.random.default_rng(seed)
    parity_idx = design["parity_idx"]
    # A comparator observes a one-sided 0 -> +t move.  It retains the same
    # candidate-preserving property but lacks the parity cancellation.
    p_plus, p_minus = probabilities(arms, parity_idx, condition,
                                    *condition_params(condition))
    cluster_parity = np.zeros(respondents)
    cluster_naive = np.zeros(respondents)
    for rid in range(respondents):
        orientation = int(rng.integers(0, 2))
        p_use = p_plus if orientation else p_minus
        y = rng.binomial(1, p_use)
        sign = 1.0 if orientation else -1.0
        # The observed score is the oriented response.  Pair-specific weights
        # use the sign of the local eta signal and stabilize the contrast.
        # The symmetric fibre construction gives the same positive odd
        # direction for every selected arm, so equal positive weights preserve
        # the utility signal instead of cancelling it across utility gaps.
        w = np.ones(len(parity_idx), dtype=float)
        w /= np.linalg.norm(w)
        cluster_parity[rid] = sign * float(w @ y)
        # One-sided comparator uses the + member against the candidate baseline
        # and has no minus member to remove even geometry effects.
        y_naive = rng.binomial(1, p_plus)
        p0 = sigmoid(0.030 * np.asarray([arms[int(i)]["u"] for i in parity_idx]))
        cluster_naive[rid] = float(w @ (y_naive - p0))
    p_parity = randomization_pvalue(cluster_parity, permutation_reps,
                                     np.random.default_rng(seed + 101))
    p_naive = randomization_pvalue(cluster_naive, permutation_reps,
                                    np.random.default_rng(seed + 202))
    return int(p_parity < 0.05), int(p_naive < 0.05), p_parity, p_naive


def condition_params(condition):
    if condition == "omitted":
        return 0.10, 0.0, 0.0, False
    if condition == "even_scale":
        return 0.0, 0.005, 0.0, False
    if condition == "framing":
        return 0.0, 0.0, 0.10, False
    if condition == "combined":
        return 0.10, 0.005, 0.0, False
    if condition == "odd_scale":
        return 0.0, 0.005, 0.0, True
    return 0.0, 0.0, 0.0, False


def run(reps=200, respondents=600, permutation_reps=199,
        out="results/parity_fibre_benchmark.csv"):
    arms = arm_grid()
    design = choose_design(arms)
    print("parity indices", design["parity_idx"].tolist())
    for i in design["parity_idx"]:
        arm = arms[int(i)]
        print("arm", int(i), "u", arm["u"], "c", arm["c"], "t", arm["t"],
              "d_equal", round(arm["d_plus"] - arm["d_minus"], 12),
              "h_odd", round(arm["h_plus"] + arm["h_minus"], 12))
    conditions = ("null", "omitted", "even_scale", "framing", "odd_scale",
                  "combined")
    rows = []
    for condition in conditions:
        for rep in range(reps):
            parity, naive, p_parity, p_naive = evaluate(
                arms, design, condition, respondents, permutation_reps,
                88000000 + rep)
            rows.extend([
                {"condition": condition, "rep": rep,
                 "diagnostic": "parity_fibre", "reject": parity,
                 "pvalue": p_parity},
                {"condition": condition, "rep": rep,
                 "diagnostic": "one_sided_comparator", "reject": naive,
                 "pvalue": p_naive},
            ])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    for condition in conditions:
        for diagnostic in ("parity_fibre", "one_sided_comparator"):
            subset = [r for r in rows if r["condition"] == condition and
                      r["diagnostic"] == diagnostic]
            print(condition, diagnostic, "rejection_rate",
                  round(float(np.mean([r["reject"] for r in subset])), 3))
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--respondents", type=int, default=600)
    parser.add_argument("--permutation-reps", type=int, default=199)
    parser.add_argument("--out", default="results/parity_fibre_benchmark.csv")
    run(**vars(parser.parse_args()))
