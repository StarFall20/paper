"""Search for frequency perturbations invisible to the score and Hessian at zero."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits

from run import HERE, OUTSIDE, frozen_forecast, frequency_scenarios, moment, read_data, sha


def redistribution(x, g, base, direction, tv):
    n, dim = x.shape
    upper = np.triu_indices(dim)
    second = (x[:, :, None] * x[:, None, :])[:, upper[0], upper[1]] / 4
    features = np.column_stack((np.ones(n), x * (.5 - g)[:, None], second))
    scale = np.maximum(np.max(abs(features), axis=0), 1e-12)
    row_constraints = sparse.csr_matrix((features / scale).T)
    keep = sparse.hstack((row_constraints, sparse.csr_matrix(row_constraints.shape)), format="csr")
    eye = sparse.eye(n, format="csr")
    total = sparse.csr_matrix(np.r_[np.zeros(n), np.ones(n)][None, :])
    balances = sparse.vstack((sparse.hstack((eye, -eye)), sparse.hstack((-eye, -eye)), total),
                             format="csr")
    evaluation = np.maximum(x @ direction, 0) - g * (x @ direction)
    fit = linprog(np.r_[evaluation, np.zeros(n)], A_eq=keep, b_eq=row_constraints @ base,
                  A_ub=balances, b_ub=np.r_[base, -base, 2 * tv],
                  bounds=list(zip(base * .25, base * 4)) + [(0, None)] * n, method="highs-ds",
                  options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})
    assert fit.success, fit.message
    proposed = fit.x[:n]
    actual_tv = float(abs(proposed - base).sum() / 2)
    moment_error = float(np.max(abs(features.T @ (proposed - base))))
    assert moment_error < 1e-8 and actual_tv <= tv + 1e-8 and min(proposed) > 0
    assert abs(sum(proposed) - 1) < 1e-10
    return proposed, actual_tv, moment_error


def verification(args):
    sys.path[:0] = [str(args.archive.resolve() / "results/official_ppi_runtime/reference/analysis"),
                    str(args.archive.resolve() / "analysis")]
    from public_conjoint_optimizer_audit import NumericOnly
    from power_tuned_choice_audit import WeightedLogit, support

    plan = json.loads((HERE / "plan.json").read_text())
    results = json.loads(args.results.read_text())
    _, x, y, ai, revision = read_data(args.data, plan, NumericOnly)
    assert revision == results["source_commit"]
    train, aux, labels = (slice(*plan[k]) for k in ("training_rows", "auxiliary_library", "label_pool"))
    block = plan["source_block_size"]
    blocks = np.arange(labels.start, labels.stop).reshape(-1, block)
    rng = np.random.default_rng(plan["allocation_seed"])
    orders = [blocks[rng.permutation(len(blocks))].ravel() for _ in range(plan["allocation_count"])]
    forecasts = {}
    for name, response in zip(plan["frozen_forecasts"], [y[train], ai], strict=True):
        g, _ = frozen_forecast(x[train], response, x, plan["ridge_penalty_multiplier"])
        forecasts[name] = g, frequency_scenarios(g[aux], plan)

    audited = []
    for witness in results["witnesses"]:
        name = witness["forecast"]
        lam = witness["weight"]
        idx = orders[witness["allocation"]][:witness["budget"]]
        assert sha(idx.astype("<i8").tobytes()) == witness["allocation_sha256"]
        g, counts = forecasts[name]
        base_scenario, = [s for s, r in witness["comparisons"].items()
                          if r["alternate_probability_status"] == "interior"]
        exterior, = [r for r in witness["comparisons"].values()
                     if r["alternate_probability_status"] == "outside"]
        direction = np.asarray(exterior["direction"])
        base = counts[base_scenario] / counts[base_scenario].sum()
        design = np.vstack((x[idx], x[aux]))
        labelled_mass = np.full(len(idx), (1 - lam) / len(idx))
        start_human, start_delta = moment(x[idx], y[idx], g[idx], x[aux], g[aux], base)
        start = WeightedLogit(design, np.r_[labelled_mass, lam * base], start_human + lam * start_delta)
        at_zero = np.zeros(design.shape[1])
        start_value, start_gradient = start.value(at_zero)
        start_hessian = start.hessian(at_zero)
        for tv in [0.2, 0.6]:
            changed, actual_tv, moment_error = redistribution(x[aux], g[aux], base, direction, tv)
            human, delta = moment(x[idx], y[idx], g[idx], x[aux], g[aux], changed)
            weights = np.r_[labelled_mass, lam * changed]
            end = WeightedLogit(design, weights, human + lam * delta)
            end_value, end_gradient = end.value(at_zero)
            errors = [abs(start_value - end_value), float(max(abs(start_gradient - end_gradient))),
                      float(np.max(abs(start_hessian - end.hessian(at_zero))))]
            assert max(errors) < 1e-8
            recession = float(weights @ np.maximum(design @ direction, 0) - end.moment @ direction)
            end_state = support(end)
            assert (recession >= -1e-8) or end_state["status"] == OUTSIDE
            audited.append(dict(forecast=name, budget=witness["budget"], weight=lam,
                                allocation=witness["allocation"], base_scenario=base_scenario,
                                total_variation_bound=tv, total_variation=actual_tv,
                                relative_base_count_range=[float(min(changed/base)), float(max(changed/base))],
                                predicted_weights_sha256=sha(changed.astype("<f8").tobytes()),
                                derivative_errors=errors, constrained_moment_error=moment_error,
                                original_direction_recession=recession, final_state=end_state["status"],
                                separating_ray=end_state.get("direction")))
    result = {"witness_allocations": len(results["witnesses"]),
              "audited_scenarios": len(audited),
              "exterior_scenarios": sum(r["final_state"] == OUTSIDE for r in audited),
              "max_derivative_error": max((max(r["derivative_errors"]) for r in audited), default=0),
              "rows": audited}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"verified {result['audited_scenarios']} matched-derivative candidates, "
          f"{result['exterior_scenarios']} exterior")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    with threadpool_limits(limits=1):
        verification(parser.parse_args())
