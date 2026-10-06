"""Audit the proposed maximin layer for candidate-preserving task pairs.

The paired-task method is already a complete test object: a transformation
must preserve the candidate utility differences while making plausible
departures visible.  This script asks a narrower design question before the
responses are generated: does selecting shifts by a maximin score improve on
the fixed three-shift instrument used in the benchmark?

The score is deterministic and outcome-free.  For each feasible common shift,
it computes the Euclidean distance between the base and shifted choice
probability vectors under four predeclared probe utilities.  The maximin score
is the weakest normalized separation across probes.  A constrained design is
required to have three distinct shifts, pairwise Manhattan distance at least
30, and total movement at most 150.  An unconstrained design is reported as a
failure boundary because it is allowed to repeat the most sensitive shift.

The output is an audit, not a new empirical result.  If the constrained design
does not improve the fixed design, the optimization should remain a sensitivity
analysis and should not be presented as the paper's primary novelty.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os

import numpy as np


BASE = np.array([[50.0, 40.0], [70.0, 50.0], [90.0, 60.0]])
FIXED = ((20, 0), (0, 60), (20, 50))
PROBE_NAMES = ("nonlinear", "threshold", "interaction", "cubic")


def softmax(u):
    u = u - np.max(u, axis=-1, keepdims=True)
    e = np.exp(u)
    return e / np.sum(e, axis=-1, keepdims=True)


def probe_utility(x, probe):
    """Return the three-alternative utility under one predeclared probe."""
    beta = np.array([-0.35, -0.18])
    u = np.array([-1.15, 0.0, -0.95]) + x @ beta / 100.0
    time = x[:, 0] / 100.0
    cost = x[:, 1] / 100.0
    if probe == "nonlinear":
        u = u + 6.0 * time**2
    elif probe == "threshold":
        u = u - 10.0 * np.maximum(cost - 0.85, 0.0)
    elif probe == "interaction":
        u = u + 8.0 * time * cost
    elif probe == "cubic":
        # Deliberately outside the fitted probe library used in the main
        # benchmark.  The coefficient keeps the probabilities interior.
        u = u + 4.0 * time**3
    else:
        raise ValueError(f"unknown probe: {probe}")
    return u


def separation(shift, probe):
    base = softmax(probe_utility(BASE, probe)[None, :])[0]
    moved = softmax(probe_utility(BASE + np.asarray(shift)[None, :], probe)[None, :])[0]
    return float(np.linalg.norm(moved - base))


def candidate_grid(time_max, cost_max, step):
    return [(time, cost) for time in range(0, time_max + 1, step)
            for cost in range(0, cost_max + 1, step)
            if (time, cost) != (0, 0)]


def scores(shifts):
    values = {shift: {probe: separation(shift, probe) for probe in PROBE_NAMES}
              for shift in shifts}
    maxima = {probe: max(values[shift][probe] for shift in shifts) for probe in PROBE_NAMES}
    normalized = {shift: {probe: values[shift][probe] / maxima[probe]
                          for probe in PROBE_NAMES} for shift in shifts}
    return values, normalized


def design_score(shifts, normalized):
    # The design is judged by the weakest probe after aggregating its selected
    # shifts.  This rewards coverage without allowing one probe to dominate.
    return float(min(sum(normalized[shift][probe] for shift in shifts)
                     for probe in PROBE_NAMES))


def select_maximin(grid, normalized, min_distance, movement_budget):
    best = None
    for shifts in itertools.combinations(grid, 3):
        if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) < min_distance
               for a, b in itertools.combinations(shifts, 2)):
            continue
        if sum(t + c for t, c in shifts) > movement_budget:
            continue
        score = design_score(shifts, normalized)
        key = (score, -sum(t + c for t, c in shifts), shifts)
        if best is None or key > best[0]:
            best = (key, shifts)
    if best is None:
        raise RuntimeError("no feasible three-shift design")
    return best[1]


def select_unconstrained(grid, normalized):
    # This intentionally allows duplicates.  It exposes the common collapse
    # of generic maximin criteria onto one large shift.
    best = None
    for shifts in itertools.combinations_with_replacement(grid, 3):
        score = design_score(shifts, normalized)
        key = (score, -sum(t + c for t, c in shifts), shifts)
        if best is None or key > best[0]:
            best = (key, shifts)
    return best[1]


def run(time_max=40, cost_max=60, step=5, min_distance=30,
        movement_budget=150, out="results/metamorphic_design_score.csv"):
    grid = candidate_grid(time_max, cost_max, step)
    _, normalized = scores(grid)
    constrained = select_maximin(grid, normalized, min_distance, movement_budget)
    unconstrained = select_unconstrained(grid, normalized)
    designs = {"fixed": FIXED, "constrained_maximin": constrained,
               "unconstrained_maximin": unconstrained}

    rows = []
    for name, shifts in designs.items():
        score = design_score(shifts, normalized)
        for rank, shift in enumerate(shifts, 1):
            for probe in PROBE_NAMES:
                rows.append({"design": name, "shift_rank": rank,
                             "time_shift": shift[0], "cost_shift": shift[1],
                             "probe": probe,
                             "normalized_separation": normalized[shift][probe],
                             "design_min_score": score})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print("fixed", FIXED, "min_score", round(design_score(FIXED, normalized), 6))
    print("constrained_maximin", constrained,
          "min_score", round(design_score(constrained, normalized), 6))
    print("unconstrained_maximin", unconstrained,
          "min_score", round(design_score(unconstrained, normalized), 6))
    print(f"wrote {len(rows)} metamorphic design-score rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-max", type=int, default=40)
    parser.add_argument("--cost-max", type=int, default=60)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--min-distance", type=int, default=30)
    parser.add_argument("--movement-budget", type=int, default=150)
    parser.add_argument("--out", default="results/metamorphic_design_score.csv")
    run(**vars(parser.parse_args()))
