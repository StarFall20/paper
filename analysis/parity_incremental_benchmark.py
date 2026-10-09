"""Incremental benchmark: parity fibre versus observational LR/residual checks.

The observational support stays near the candidate fibre's centre, where the
omitted decomposition term is weakly identified.  The parity supplement moves
along a reflected candidate-preserving trajectory.  This benchmark is a
planning comparator, not a final empirical analysis.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import anchored_fibre_benchmark as anchored
import parity_fibre_benchmark as parity


def sigmoid(x):
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def task_probability(a, b, condition):
    sa, sb = a.sum(axis=1), b.sum(axis=1)
    za, zb = a[:, 0] - a[:, 1], b[:, 0] - b[:, 1]
    gap = 0.030 * (sa - sb)
    if condition in ("omitted", "combined"):
        gap = gap + 0.10 * (za ** 2 - zb ** 2)
    if condition in ("even_scale", "combined"):
        d = np.sum((a - b) ** 2, axis=1)
        gap = gap / (1.0 + 0.002 * d)
    return sigmoid(gap)


def build_observational(seed, respondents, condition, tasks=8):
    rng = np.random.default_rng(seed)
    rows = []
    for rid in range(respondents):
        sa = rng.uniform(65.0, 105.0, size=tasks)
        sb = rng.uniform(65.0, 105.0, size=tasks)
        # Narrow support deliberately hides the decomposition direction.
        za = rng.normal(0.0, 0.25, size=tasks)
        zb = rng.normal(0.0, 0.25, size=tasks)
        a = np.column_stack(((sa + za) / 2.0, (sa - za) / 2.0))
        b = np.column_stack(((sb + zb) / 2.0, (sb - zb) / 2.0))
        p = task_probability(a, b, condition)
        y = rng.binomial(1, p)
        for q in range(tasks):
            rows.append((rid, a[q], b[q], int(y[q])))
    return rows


def run(reps=200, respondents=600, permutation_reps=199,
        out="results/parity_incremental_benchmark.csv"):
    conditions = ("null", "omitted", "even_scale", "combined")
    arms = parity.arm_grid()
    design = parity.choose_design(arms)
    rows = []
    for condition in conditions:
        for rep in range(reps):
            obs = build_observational(12000000 + rep, respondents, condition)
            lr_reject, lr_stat = anchored.lr_diagnostic(obs, 12100000 + rep)
            ml_reject, ml_p = anchored.residual_diagnostic(
                obs, 12200000 + rep, permutation_reps)
            parity_condition = {
                "null": "null", "omitted": "omitted",
                "even_scale": "even_scale", "combined": "combined",
            }[condition]
            fibre, fibre_naive, p_fibre, p_naive = parity.evaluate(
                arms, design, parity_condition, respondents, permutation_reps,
                12300000 + rep)
            rows.extend([
                {"condition": condition, "rep": rep,
                 "diagnostic": "observational_LR", "reject": lr_reject,
                 "stat": lr_stat},
                {"condition": condition, "rep": rep,
                 "diagnostic": "observational_residual", "reject": ml_reject,
                 "stat": ml_p},
                {"condition": condition, "rep": rep,
                 "diagnostic": "parity_fibre", "reject": fibre,
                 "stat": p_fibre},
                {"condition": condition, "rep": rep,
                 "diagnostic": "one_sided_fibre", "reject": fibre_naive,
                 "stat": p_naive},
            ])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    for condition in conditions:
        for diagnostic in ("observational_LR", "observational_residual",
                           "parity_fibre", "one_sided_fibre"):
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
    parser.add_argument("--out", default="results/parity_incremental_benchmark.csv")
    run(**vars(parser.parse_args()))
