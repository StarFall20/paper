"""Calibration-error audit for candidate-preserving smart-device fibres.

The task profiles are exact under the supplied additive coefficients. This
script perturbs the true coefficients while keeping the design fixed, measuring
how much calibration error alone can create a parity rejection under a null
with no omitted interaction.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

import smart_device_fibre_design as design


def softmax(values):
    values = np.asarray(values, dtype=float)
    values -= np.max(values)
    e = np.exp(values)
    return e / e.sum()


def randomization_pvalue(scores, reps, rng):
    observed = abs(float(np.mean(scores)))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(reps, len(scores)))
    draws = np.abs((signs @ scores) / len(scores))
    return float((1.0 + np.sum(draws >= observed)) / (reps + 1.0))


def profile_utility(profile, coeff):
    return float(sum(coeff[k][level] for k, level in enumerate(profile)))


def simulate(noise_sd, respondents, permutation_reps, seed):
    rng = np.random.default_rng(seed)
    coeff = tuple(tuple(np.asarray(row) + rng.normal(0.0, noise_sd, 3)
                        for row in design.COEFFICIENTS))
    ap, am, bp, bm = design.SELECTED
    p_plus = softmax([profile_utility(ap, coeff),
                      profile_utility(bp, coeff), design.OPTOUT_UTILITY])
    p_minus = softmax([profile_utility(am, coeff),
                       profile_utility(bm, coeff), design.OPTOUT_UTILITY])
    scores = np.zeros(respondents)
    for rid in range(respondents):
        orientation = 1 if rng.integers(0, 2) else -1
        probs = p_plus if orientation > 0 else p_minus
        choice = int(rng.choice(3, p=probs))
        contrast = 1.0 if choice == 0 else (-1.0 if choice == 1 else 0.0)
        scores[rid] = orientation * contrast
    pvalue = randomization_pvalue(
        scores, permutation_reps, np.random.default_rng(seed + 701))
    # Design-level candidate gap mismatch under the perturbed truth.
    mismatch = abs((profile_utility(ap, coeff) - profile_utility(bp, coeff))
                   - (profile_utility(am, coeff) - profile_utility(bm, coeff)))
    return int(pvalue < 0.05), pvalue, mismatch


def run(reps=100, respondents=400, permutation_reps=199,
        out="results/smart_device_calibration_robustness.csv"):
    rows = []
    for noise_sd in (0.0, 0.005, 0.01, 0.02, 0.05):
        for rep in range(reps):
            reject, pvalue, mismatch = simulate(
                noise_sd, respondents, permutation_reps, 95000000 + rep)
            rows.append({"noise_sd": noise_sd, "rep": rep,
                         "reject": reject, "pvalue": pvalue,
                         "gap_mismatch": mismatch})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    for noise_sd in (0.0, 0.005, 0.01, 0.02, 0.05):
        subset = [r for r in rows if r["noise_sd"] == noise_sd]
        print(noise_sd, "rejection_rate",
              round(float(np.mean([r["reject"] for r in subset])), 3),
              "mean_gap_mismatch",
              round(float(np.mean([r["gap_mismatch"] for r in subset])), 4))
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--respondents", type=int, default=400)
    parser.add_argument("--permutation-reps", type=int, default=199)
    parser.add_argument("--out", default="results/smart_device_calibration_robustness.csv")
    run(**vars(parser.parse_args()))
