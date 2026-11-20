"""Independently audit how much frequency transfer a fixed correction can tolerate."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits

from run import HERE, INSIDE, OUTSIDE, frozen_forecast, frequency_scenarios, moment, read_data, sha


def frontier(x, weights, rates, point, change):
    """Return the last feasible point along an affine frequency path."""
    n = len(x)
    equality = sparse.hstack((sparse.csr_matrix(x.T), sparse.csr_matrix(-change[:, None])))
    caps = sparse.hstack((sparse.eye(n, format="csr"), sparse.csr_matrix(-rates[:, None])))
    fit = linprog(np.r_[np.zeros(n), -1.], A_ub=caps, b_ub=weights, A_eq=equality,
                  b_eq=point, bounds=[(0., None)] * n + [(0., 1.)], method="highs-ds")
    assert fit.success, fit.message
    rho = float(fit.x[-1])
    assert 0 < rho < 1
    assert max(abs(equality @ fit.x - point)) < 1e-8
    assert max(caps @ fit.x - weights) < 1e-8
    direction = fit.eqlin.marginals
    direction /= np.linalg.norm(direction, ord=1)
    scores = np.maximum(x @ direction, 0.)
    intercept = float(weights @ scores - point @ direction)
    slope = float(rates @ scores - change @ direction)
    assert intercept > 0 and slope < 0 and abs(intercept + rho * slope) < 1e-8
    return rho, intercept, slope


def audit(args):
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

    paths = []
    for witness in results["witnesses"]:
        name = witness["forecast"]
        lam = witness["weight"]
        idx = orders[witness["allocation"]][:witness["budget"]]
        assert sha(idx.astype("<i8").tobytes()) == witness["allocation_sha256"]
        g, counts = forecasts[name]
        finite, = [s for s, r in witness["comparisons"].items()
                   if r["alternate_probability_status"] == "interior"]
        exterior, = [s for s, r in witness["comparisons"].items()
                     if r["alternate_probability_status"] == "outside"]
        f0 = counts[finite] / counts[finite].sum()
        f1 = counts[exterior] / counts[exterior].sum()
        human, delta = moment(x[idx], y[idx], g[idx], x[aux], g[aux], f0)
        design = np.vstack((x[idx], x[aux]))
        weights = np.r_[np.full(len(idx), (1 - lam) / len(idx)), lam * f0]
        rates = np.r_[np.zeros(len(idx)), lam * (f1 - f0)]
        point = human + lam * delta
        change = lam * x[aux].T @ ((f1 - f0) * g[aux])
        rho, intercept, slope = frontier(design, weights, rates, point, change)
        assert np.linalg.matrix_rank(x[aux]) == x.shape[1]
        below = rho / 2
        above = (1 + rho) / 2
        states = [support(WeightedLogit(design, weights + v * rates, point + v * change))["status"]
                  for v in (below, above)]
        assert states == [INSIDE, OUTSIDE]
        paths.append(dict(forecast=name, budget=witness["budget"], weight=lam,
                          allocation=witness["allocation"], from_scenario=finite, to_scenario=exterior,
                          maximum_feasible_transfer=rho,
                          total_variation_from_finite_at_frontier=float(rho * abs(f1 - f0).sum() / 2),
                          recession_intercept=intercept, recession_slope=slope,
                          interior_probe=below, exterior_probe=above))
    args.out.write_text(json.dumps(dict(path_count=len(paths), paths=paths), indent=2, allow_nan=False) + "\n")
    print(f"verified {len(paths)} paths")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    with threadpool_limits(limits=1):
        audit(parser.parse_args())
