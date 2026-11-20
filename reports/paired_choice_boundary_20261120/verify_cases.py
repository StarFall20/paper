"""Verify all simulated corrected moments with a separate probability LP."""

import argparse
import csv
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, eye, hstack, vstack
from scipy.stats import binom


def run(records, plan):
    rows = list(csv.DictReader(records.open(encoding="utf-8", newline="")))
    all_cells = Counter()
    span_errors = []
    all_ordinary = Counter()
    outcomes = {}
    cached = {}
    for row in rows:
        m, q, trial, family = map(int, (row["respondents"], row["slopes"], row["draw"], row["family"]))
        rho = float(row["rho"])
        k = m, q, trial, family, rho
        assert k not in outcomes
        rng = np.random.default_rng(np.random.SeedSequence([plan["seed"], family, m, trial]))
        labels = rng.integers(0, 2, size=2 * m)
        drivers = rng.standard_normal((2, m, q, 2))
        a = (np.sqrt(1 + rho) + np.sqrt(1 - rho)) / 2
        b = (np.sqrt(1 + rho) - np.sqrt(1 - rho)) / 2
        x = np.empty((2, 2 * m, q + 1))
        x[:, :, 0] = 1
        x[:, ::2, 1:] = a * drivers[..., 0] + b * drivers[..., 1]
        x[:, 1::2, 1:] = b * drivers[..., 0] + a * drivers[..., 1]

        human, auxiliary = x
        count = 2 * m
        if count not in cached:
            identity = eye(count, format="csc")
            floor = csc_matrix(np.ones((count, 1)))
            inequalities = vstack((hstack((-identity, floor)), hstack((identity, floor))), format="csc")
            cached[count] = (inequalities, np.r_[np.zeros(count), np.ones(count)])
        inequalities, limit = cached[count]
        target = (human.T @ (labels - 0.5) + 0.5 * auxiliary.sum(axis=0)) / count
        original = linprog(
            np.r_[np.zeros(count), -1],
            A_ub=inequalities, b_ub=limit, A_eq=np.column_stack((auxiliary.T / count, np.zeros(q + 1))),
            b_eq=target, bounds=[(None, None)] * count + [(None, 0.5)], method="highs-ipm",
        )
        assert original.success, (k, original.message)
        gap = float(original.x[-1])
        original_score_error = float(np.max(np.abs(auxiliary.T @ original.x[:-1] / count - target)))
        assert original_score_error < 1e-7
        state = "exterior" if gap < -5e-8 else "interior" if gap > 5e-8 else "near_boundary"
        span_errors.append(abs(0.5 * (1 - float(row["minimum_radius"])) - gap))
        assert row["state"] == state
        outcomes[k] = state
        all_cells[family, m, q, rho, state] += 1
        all_ordinary[row["ordinary_state"] or "not_requested"] += 1
    assert len(rows) == 2280 and all_ordinary["interior"] == 1800
    assert max(span_errors) < 2e-6
    changes = []
    for m in plan["primary"]["labelled_respondents"]:
        q = round(m * plan["primary"]["slope_count_per_labelled_person"])
        counts = Counter((outcomes[m, q, j, 0, 0.0], outcomes[m, q, j, 0, 0.99]) for j in range(200))
        changes.append({"respondents": m, "slopes": q, "nominal_ratio": 0.585,
                        "ordinary_separation_probability": float(binom.cdf(q, 2 * m - 1, 0.5)),
                        "paired_states_0_to_099": {"/".join(pair): n for pair, n in sorted(counts.items())}})
    return {
        "verified_corrected_cases": len(outcomes),
        "maximum_original_probability_margin_difference": max(span_errors),
        "ordinary_control_states": dict(all_ordinary),
        "cells": [{"family": f, "respondents": m, "slopes": q, "rho": rho, "state": state, "count": n}
                  for (f, m, q, rho, state), n in sorted(all_cells.items())],
        "paired_cases": changes,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--plan", type=Path, default=Path(__file__).with_name("plan.json"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = run(a.records, json.loads(a.plan.read_text(encoding="utf-8")))
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cells"}), flush=True)
