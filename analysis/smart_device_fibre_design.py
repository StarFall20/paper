"""Application audit for a parity fibre in the smart-device attributes.

The selected plus/minus tasks are built from the supplied manuscript's six
attributes and additive MNL coefficients.  Every alternative keeps its
candidate utility, the A--B gap, the opt-out utility, and the squared raw-level
A--B distance fixed.  The manuscript's intelligence-by-cloud interaction is
odd across the reflection.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os

import numpy as np

ATTRIBUTES = (
    "use_management", "intelligence", "support", "evidence", "data", "price"
)
LABELS = {
    "use_management": ("Hospital-based", "Hospital setup + periodic visits",
                        "Home use + remote follow-up"),
    "intelligence": ("Basic preset control", "Status monitoring",
                      "Personalized intelligent adjustment"),
    "support": ("Basic guidance", "Periodic follow-up", "Continuous support + alerts"),
    "evidence": ("Basic validation", "Single-center evidence", "Multicenter evidence"),
    "data": ("Local storage", "Institutional encrypted storage",
             "Authorized cloud sharing"),
    "price": ("RMB 500", "RMB 1,500", "RMB 3,000"),
}
# Utilities used in the supplied manuscript's additive candidate.
COEFFICIENTS = (
    (0.0, 0.40, 0.85),
    (0.0, 0.20, 0.50),
    (0.0, 0.30, 0.70),
    (0.0, 0.35, 0.85),
    (0.0, 0.25, 0.12),
    (-0.24, -0.72, -1.44),
)
OPTOUT_UTILITY = -0.42

# A compact candidate-preserving reflection found by exhaustive enumeration.
# Each row is (A_plus, A_minus, B_plus, B_minus), using level indices 0..2.
SELECTED = (
    (1, 0, 1, 2, 2, 2),
    (1, 2, 1, 1, 2, 2),
    (0, 2, 1, 1, 2, 2),
    (2, 0, 1, 0, 2, 2),
)


def candidate_utility(profile):
    return float(sum(COEFFICIENTS[k][level]
                     for k, level in enumerate(profile)))


def omitted_interaction(profile):
    # Terms already present in the supplied DGP: home use x continuous support
    # and personalized intelligence x authorized cloud sharing.
    mode, intelligence, support, _evidence, data, _price = profile
    return float(0.90 * (mode == 2 and support == 2)
                 + 0.55 * (intelligence == 2 and data == 2))


def squared_distance(a, b):
    return float(sum((x - y) ** 2 for x, y in zip(a, b)))


def audit_row():
    ap, am, bp, bm = SELECTED
    profiles = {"A_plus": ap, "A_minus": am, "B_plus": bp, "B_minus": bm}
    utilities = {key: candidate_utility(value) for key, value in profiles.items()}
    omitted = {key: omitted_interaction(value) for key, value in profiles.items()}
    d_plus = squared_distance(ap, bp)
    d_minus = squared_distance(am, bm)
    row = {
        "candidate_A_plus": utilities["A_plus"],
        "candidate_A_minus": utilities["A_minus"],
        "candidate_B_plus": utilities["B_plus"],
        "candidate_B_minus": utilities["B_minus"],
        "candidate_gap_plus": utilities["A_plus"] - utilities["B_plus"],
        "candidate_gap_minus": utilities["A_minus"] - utilities["B_minus"],
        "optout_utility": OPTOUT_UTILITY,
        "distance_plus": d_plus,
        "distance_minus": d_minus,
        "omitted_A_plus": omitted["A_plus"],
        "omitted_A_minus": omitted["A_minus"],
        "omitted_B_plus": omitted["B_plus"],
        "omitted_B_minus": omitted["B_minus"],
        "omitted_gap_plus": omitted["A_plus"] - omitted["B_plus"],
        "omitted_gap_minus": omitted["A_minus"] - omitted["B_minus"],
        "full_menu_plus": str((round(utilities["A_plus"], 6),
                                round(utilities["B_plus"], 6), OPTOUT_UTILITY)),
        "full_menu_minus": str((round(utilities["A_minus"], 6),
                                 round(utilities["B_minus"], 6), OPTOUT_UTILITY)),
    }
    assert abs(row["candidate_A_plus"] - row["candidate_A_minus"]) < 1e-12
    assert abs(row["candidate_B_plus"] - row["candidate_B_minus"]) < 1e-12
    assert abs(row["candidate_gap_plus"] - row["candidate_gap_minus"]) < 1e-12
    assert abs(d_plus - d_minus) < 1e-12
    assert abs(row["omitted_gap_plus"] + row["omitted_gap_minus"]) < 1e-12
    return row


def softmax(values):
    values = np.asarray(values, dtype=float)
    values = values - np.max(values)
    ex = np.exp(values)
    return ex / ex.sum()


def randomization_pvalue(scores, reps, rng):
    observed = abs(float(np.mean(scores)))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(reps, len(scores)))
    draws = np.abs((signs @ scores) / len(scores))
    return float((1.0 + np.sum(draws >= observed)) / (reps + 1.0))


def simulate(condition, respondents, permutation_reps, seed):
    row = audit_row()
    eta = 1.0 if condition == "omitted" else 0.0
    ap, am, bp, bm = SELECTED
    p_plus = softmax([candidate_utility(ap) + eta * omitted_interaction(ap),
                      candidate_utility(bp) + eta * omitted_interaction(bp),
                      OPTOUT_UTILITY])
    p_minus = softmax([candidate_utility(am) + eta * omitted_interaction(am),
                       candidate_utility(bm) + eta * omitted_interaction(bm),
                       OPTOUT_UTILITY])
    rng = np.random.default_rng(seed)
    scores = np.zeros(respondents)
    for rid in range(respondents):
        orientation = 1 if rng.integers(0, 2) else -1
        probs = p_plus if orientation > 0 else p_minus
        choice = int(rng.choice(3, p=probs))
        # A-B contrast, with opt-out coded as zero. Orientation aligns the
        # reflected response before the cluster randomization test.
        contrast = 1.0 if choice == 0 else (-1.0 if choice == 1 else 0.0)
        scores[rid] = orientation * contrast
    pvalue = randomization_pvalue(
        scores, permutation_reps, np.random.default_rng(seed + 991))
    return int(pvalue < 0.05), pvalue


def run(reps=200, respondents=400, permutation_reps=199,
        out="results/smart_device_fibre_benchmark.csv"):
    audit = audit_row()
    print("audit", audit)
    rows = []
    for condition in ("null", "omitted"):
        for rep in range(reps):
            reject, pvalue = simulate(condition, respondents, permutation_reps,
                                      94000000 + rep)
            rows.append({"condition": condition, "rep": rep,
                         "reject": reject, "pvalue": pvalue})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    for condition in ("null", "omitted"):
        subset = [r for r in rows if r["condition"] == condition]
        print(condition, "rejection_rate",
              round(float(np.mean([r["reject"] for r in subset])), 3))
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--respondents", type=int, default=400)
    parser.add_argument("--permutation-reps", type=int, default=199)
    parser.add_argument("--out", default="results/smart_device_fibre_benchmark.csv")
    run(**vars(parser.parse_args()))
