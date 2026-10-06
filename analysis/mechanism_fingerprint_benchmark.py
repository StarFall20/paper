"""Prototype mechanism-fingerprint audit for the paired-task method.

The candidate-preserving fibre relation detects utility-form departures.  This
extension adds two model-implied placebo relations: alternative relabelling
and a repeated task placed later in a sequence.  The three contrasts are
estimated after freezing an additive candidate on nuisance tasks from
development respondents.  A cluster sign-flip reference is used for each
axis, and the reported signature is only a mechanism-family diagnostic.

The benchmark is deliberately a gate.  If single-mechanism conditions do not
activate separate axes, the extension is dropped instead of being presented as
local identification.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from exact_paired_task_benchmark import design, fit_mnl, softmax


AXES = ("fibre", "presentation", "sequence")
CONDITIONS = (
    "additive", "nonlinear", "random_price", "position_bias", "inertia",
    "nonlinear_position", "position_inertia", "nonlinear_inertia",
)


def draw_choice(probability, rng):
    return int(rng.choice(len(probability), p=probability))


def base_profile(rng):
    return np.array([[50.0, 40.0], [70.0, 50.0], [90.0, 60.0]]) + rng.normal(0.0, 2.0, (3, 2))


def utility(x, condition, random_cost, alternative_ids=None):
    beta = np.array([-1.15, -0.95])
    asc = np.array([-0.35, 0.0, -0.18])
    if alternative_ids is None:
        alternative_ids = np.arange(3)
    out = asc[np.asarray(alternative_ids)] + x @ beta / 100.0
    time = x[:, 0] / 100.0
    cost = x[:, 1] / 100.0
    if condition in ("nonlinear", "nonlinear_position", "nonlinear_inertia"):
        out = out + 6.0 * time**2
    if condition == "random_price":
        out = out + random_cost * cost
    return out


def simulate(seed, respondents=300, nuisance_tasks=8, condition="additive"):
    rng = np.random.default_rng(seed)
    xs, ys, ids, pair_ids, pair_axis, pair_side = [], [], [], [], [], []
    pair_counter = 0
    for rid in range(respondents):
        random_cost = rng.normal(0.0, 1.0) if condition == "random_price" else 0.0

        # Nuisance tasks identify the candidate without exposing it to the
        # three focal relations used to evaluate the signature.
        for _ in range(nuisance_tasks):
            x = rng.uniform(40.0, 120.0, (3, 2))
            p = softmax(utility(x, condition, random_cost)[None, :])[0]
            xs.append(x); ys.append(draw_choice(p, rng)); ids.append(rid)
            pair_ids.append(-1); pair_axis.append("nuisance"); pair_side.append(-1)

        # 1. Candidate-preserving common shift (forced-choice block).
        x = base_profile(rng)
        for side, xx in enumerate((x, x + np.array([35.0, 45.0])[None, :])):
            p = softmax(utility(xx, condition, random_cost)[None, :])[0]
            xs.append(xx); ys.append(draw_choice(p, rng)); ids.append(rid)
            pair_ids.append(pair_counter); pair_axis.append("fibre"); pair_side.append(side)
        pair_counter += 1

        # 2. Same task under a counterbalanced alternative permutation.  The
        # stored outcome is mapped back to alternative identity below.
        x = base_profile(rng)
        orders = (np.array([0, 1, 2]), np.array([1, 0, 2]))
        for side, order in enumerate(orders):
            displayed = x[order]
            # Alternative-specific constants follow alternative identity when
            # the displayed columns are permuted; only the presentation
            # mechanism is allowed to depend on position.
            u = utility(displayed, condition, random_cost, alternative_ids=order)
            if condition in ("position_bias", "nonlinear_position", "position_inertia"):
                u = u + 0.75 * (np.arange(3) == 0)
            chosen_position = draw_choice(softmax(u[None, :])[0], rng)
            chosen_identity = int(order[chosen_position])
            xs.append(x); ys.append(chosen_identity); ids.append(rid)
            pair_ids.append(pair_counter); pair_axis.append("presentation"); pair_side.append(side)
        pair_counter += 1

        # 3. A repeated task at two sequence positions, separated by a
        # neutral filler.  The filler is deliberately absent from the
        # candidate fit and from the stored focal contrast.  Its identity is
        # fixed in advance so that a sequence effect has one declared target.
        x = base_profile(rng)
        first_p = softmax(utility(x, condition, random_cost)[None, :])[0]
        first_choice = draw_choice(first_p, rng)
        neutral_choice = 0
        second_u = utility(x, condition, random_cost)
        if condition in ("inertia", "nonlinear_inertia", "position_inertia"):
            # The second presentation follows the neutral filler.  The
            # candidate's static probability relation is unchanged, while
            # the process condition gives the filler identity a 1.20 utility
            # bonus in the repeated task.
            second_u = second_u + 1.20 * (np.arange(3) == neutral_choice)
        second_choice = draw_choice(softmax(second_u[None, :])[0], rng)
        for side, choice in enumerate((first_choice, second_choice)):
            xs.append(x); ys.append(choice); ids.append(rid)
            pair_ids.append(pair_counter); pair_axis.append("sequence"); pair_side.append(side)
        pair_counter += 1

    return (np.asarray(ids), np.asarray(ys), np.asarray(xs),
            np.asarray(pair_ids), np.asarray(pair_axis, dtype=object),
            np.asarray(pair_side))


def max_t_pvalues(contrast_map, cluster_map, reps, seed):
    """Return maxT-adjusted p-values using one sign per respondent."""
    axes = tuple(contrast_map)
    unique = np.unique(cluster_map[axes[0]])
    index = {rid: i for i, rid in enumerate(unique)}
    observed = {}
    aligned = {}
    for axis in axes:
        c = np.asarray(contrast_map[axis], dtype=float)
        g = np.asarray(cluster_map[axis])
        if not np.array_equal(np.sort(np.unique(g)), np.sort(unique)):
            raise ValueError("all fingerprint axes must use the same respondent clusters")
        order = np.asarray([index[rid] for rid in g])
        aligned[axis] = (c, order)
        observed[axis] = float(np.sum(np.mean(c, axis=0) ** 2))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(reps, len(unique)))
    max_stats = np.zeros(reps)
    for axis in axes:
        c, order = aligned[axis]
        draws = (signs[:, order, None] * c[None, :, :]).mean(axis=1)
        max_stats = np.maximum(max_stats, np.sum(draws * draws, axis=1))
    return {axis: float((1.0 + np.sum(max_stats >= observed[axis])) / (reps + 1.0))
            for axis in axes}


def evaluate(ids, y, x, pair_ids, pair_axis, pair_side, randomization_reps, seed):
    respondents = np.unique(ids)
    rng = np.random.default_rng(seed)
    shuffled = respondents.copy(); rng.shuffle(shuffled)
    contrasts = {axis: [] for axis in AXES}
    clusters = {axis: [] for axis in AXES}
    for test_ids in np.array_split(shuffled, 2):
        test_mask = np.isin(ids, test_ids)
        train_mask = ~test_mask
        # Nuisance tasks only: focal relation rows are reserved for testing.
        train_mask &= pair_ids < 0
        beta = fit_mnl(design(x[train_mask]), y[train_mask])
        test_rows = np.flatnonzero(np.isin(ids, test_ids))
        local = {row: i for i, row in enumerate(test_rows)}
        probs = softmax(np.einsum("njp,p->nj", design(x[test_rows]), beta))
        by_pair = {}
        for row in test_rows:
            pid = int(pair_ids[row])
            if pid >= 0:
                by_pair.setdefault(pid, {})[int(pair_side[row])] = row
        for sides in by_pair.values():
            if 0 not in sides or 1 not in sides:
                continue
            left, right = sides[0], sides[1]
            axis = str(pair_axis[left])
            observed = np.eye(3)[y[left]] - np.eye(3)[y[right]]
            expected = probs[local[left]] - probs[local[right]]
            contrasts[axis].append(observed - expected)
            clusters[axis].append(int(ids[left]))

    adjusted = max_t_pvalues(contrasts, clusters, randomization_reps, seed + 101)
    records = {}
    for axis in AXES:
        c = np.asarray(contrasts[axis], dtype=float)
        pvalue = adjusted[axis]
        records[axis] = {"pvalue": pvalue, "reject": int(pvalue < 0.05), "pairs": len(c)}
    return records


def signature(records):
    active = [axis for axis in AXES if records[axis]["reject"]]
    return "none" if not active else "_and_".join(active) if len(active) > 1 else f"{active[0]}_only"


def run(reps=50, randomization_reps=199, respondents=300, rep_start=0,
        out="results/mechanism_fingerprint_benchmark.csv"):
    rows = []
    for condition in CONDITIONS:
        for rep in range(rep_start, rep_start + reps):
            data = simulate(20261101 + rep, respondents=respondents, condition=condition)
            records = evaluate(*data, randomization_reps=randomization_reps,
                                seed=20700000 + rep)
            rows.append({"condition": condition, "rep": rep,
                         "fibre_pvalue": records["fibre"]["pvalue"],
                         "presentation_pvalue": records["presentation"]["pvalue"],
                         "sequence_pvalue": records["sequence"]["pvalue"],
                         "fibre_reject": records["fibre"]["reject"],
                         "presentation_reject": records["presentation"]["reject"],
                         "sequence_reject": records["sequence"]["reject"],
                         "signature": signature(records)})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        print(condition, "fibre", round(float(np.mean([row["fibre_reject"] for row in subset])), 3),
              "presentation", round(float(np.mean([row["presentation_reject"] for row in subset])), 3),
              "sequence", round(float(np.mean([row["sequence_reject"] for row in subset])), 3))
    print(f"wrote {len(rows)} mechanism-fingerprint rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--randomization-reps", type=int, default=199)
    parser.add_argument("--respondents", type=int, default=300)
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--out", default="results/mechanism_fingerprint_benchmark.csv")
    run(**vars(parser.parse_args()))
