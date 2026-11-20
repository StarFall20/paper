"""Simulate two linked tasks and classify fit existence from the original moments."""

import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, eye, hstack, vstack

from compute_boundary import block_root


@lru_cache(maxsize=None)
def box_constraints(rows):
    identity = eye(rows, format="csc")
    column = csc_matrix(-np.ones((rows, 1)))
    return vstack((hstack((identity, column)), hstack((-identity, column))), format="csc")


def design_pair(seed, family, count, draw, slopes, rho):
    generator = np.random.default_rng(np.random.SeedSequence([seed, family, count, draw]))
    labels = generator.integers(0, 2, size=2 * count) * 2 - 1
    independent = generator.standard_normal((2, count, slopes, 2))
    features = (independent @ block_root(rho)).swapaxes(-1, -2).reshape(2, 2 * count, slopes)
    return np.column_stack((np.ones(2 * count), features[0])), labels, np.column_stack(
        (np.ones(2 * count), features[1])
    )


def minimum_box_radius(x, moment):
    rows, cols = x.shape
    np.linalg.cholesky(x.T @ x)
    result = linprog(
        np.r_[np.zeros(rows), 1.0],
        A_ub=box_constraints(rows), b_ub=np.zeros(2 * rows),
        A_eq=np.column_stack((x.T, np.zeros(cols))), b_eq=moment,
        bounds=[(None, None)] * rows + [(0, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Active full-rank design LP failed: {result.message}")
    coefficients = result.x[:-1]
    radius = float(result.x[-1])
    direction = result.eqlin.marginals
    residual = float(np.max(np.abs(x.T @ coefficients - moment)))
    direction_norm = float(np.sum(np.abs(x @ direction)))
    signed_direction_cost = float(direction_norm - moment @ direction)
    dual_error = float(abs(moment @ direction - radius))
    assert residual < 1e-6 and np.max(np.abs(coefficients)) < radius + 1e-6
    assert direction_norm <= 1 + 1e-6 and dual_error < 1e-6
    if radius > 1 + 1e-7:
        assert signed_direction_cost < 0
        state = "exterior"
    elif radius < 1 - 1e-7:
        state = "interior"
    else:
        state = "near_boundary"
    return state, radius, residual, signed_direction_cost, dual_error


def run(plan, out):
    configs = [(0, ratio, plan["primary"]) for ratio in [plan["primary"]["slope_count_per_labelled_person"]]]
    configs.extend((index, ratio, plan["stress_controls"])
                   for index, ratio in enumerate(plan["stress_controls"]["slope_count_per_labelled_person"], 1))
    out.mkdir(parents=True, exist_ok=True)
    fields = ["family", "rho", "respondents", "tasks", "slopes", "ratio", "draw", "state", "minimum_radius",
              "primal_error", "dual_direction_cost", "duality_error", "ordinary_state", "ordinary_radius"]
    with (out / "cohort_results.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for family, ratio, cfg in configs:
            draws = cfg["cohorts_per_size"] if family == 0 else cfg["cohorts_per_size_ratio"]
            for count in cfg["labelled_respondents"]:
                slopes = round(ratio * count)
                for draw in range(draws):
                    for rho in cfg["correlations"]:
                        human, labels, auxiliary = design_pair(plan["seed"], family, count, draw, slopes, rho)
                        target = human.T @ labels
                        state, radius, residual, cost, gap = minimum_box_radius(auxiliary, target)
                        ordinary_state, ordinary_radius = "", ""
                        if cfg["ordinary_control_in_each_fit"]:
                            ordinary_state, ordinary_radius, *_ = minimum_box_radius(human, target)
                            assert ordinary_state != "exterior"
                        writer.writerow(dict(
                            family=family, rho=rho, respondents=count, tasks=2 * count, slopes=slopes, ratio=ratio,
                            draw=draw, state=state, minimum_radius=radius, primal_error=residual,
                            dual_direction_cost=cost, duality_error=gap, ordinary_state=ordinary_state,
                            ordinary_radius=ordinary_radius,
                        ))
                output.flush()
                print(f"{count} respondents, nominal ratio {ratio:.3f}, {draws} coupled cohorts", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("plan.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.plan.read_text(encoding="utf-8")), args.out)
