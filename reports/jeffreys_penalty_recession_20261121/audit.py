"""Certify descent after adding a Jeffreys log-information penalty."""

import argparse
import hashlib
import itertools
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits


def independent_basis(contrasts, costs):
    """Choose a full-rank basis in increasing order of directional cost."""
    scaled = contrasts / np.maximum(np.max(abs(contrasts), axis=0), 1e-30)
    chosen = []
    for i in np.argsort(costs, kind="stable"):
        trial = chosen + [i]
        if np.linalg.matrix_rank(scaled[trial], tol=1e-10) > len(chosen):
            chosen.append(i)
        if len(chosen) == contrasts.shape[1]:
            break
    assert len(chosen) == contrasts.shape[1]
    assert np.linalg.svd(scaled[chosen], compute_uv=False)[-1] > 1e-9
    return np.array(chosen)


def basis_check(contrasts, rates):
    chosen = independent_basis(contrasts, rates)
    brute = min(sum(rates[list(rows)])
                for rows in itertools.combinations(range(len(contrasts)), contrasts.shape[1])
                if np.linalg.matrix_rank(contrasts[list(rows)]) == contrasts.shape[1])
    assert abs(sum(rates[chosen]) - brute) < 1e-11
    return float(brute)


def toy():
    x = np.array([[1., -1.], [1., 0.], [1., 1.]])
    w = np.array([1 / 16, 7 / 8, 1 / 16])
    target = np.array([1 / 2, 3 / 32])
    d = np.array([0., 1.])
    b = basis_check(x, abs(x @ d))
    r = float(w @ np.maximum(x @ d, 0.) - target @ d)
    a = 1 / 64
    assert Fraction(r) == Fraction(-1, 32) and b == 1 and Fraction(r + a * b) == Fraction(-1, 64)
    loss = []
    for t in [0., 20., 200., 2000.]:
        u = t * (x @ d)
        log_variance = u - 2 * np.logaddexp(0., u)
        terms = []
        for pair in itertools.combinations(range(len(x)), 2):
            pair = list(pair)
            terms.append(2 * np.log(abs(np.linalg.det(x[pair]))) +
                         sum(np.log(w[pair]) + log_variance[pair]))
        log_det = logsumexp(terms)
        loss.append(float(w @ np.logaddexp(0., u) - t * (target @ d) - a * log_det))
    assert loss[-1] < loss[-2] < loss[0]
    assert abs((loss[-1] - loss[-2]) / 1800 - (r + a * b)) < 1e-12
    return {"labels": 32, "human_sum_half_logdet_normalized_coefficient": a,
            "unpenalized_directional_slope": r, "basis_cost": b, "penalized_directional_slope": r + a * b,
            "objective_at_0_20_200_2000": loss,
            "global_strict_coercivity_threshold": 1 / 32}


def multinomial_check():
    """Check the pairwise Hessian and determinant rate on a small full-rank example."""
    profiles = np.array([[[0., 0.], [1., 0.], [0., 1.]], [[0., 0.], [-1., -1.], [1., 2.]]])
    w = np.array([.4, .6])
    d = np.array([.3, .9])
    focal = profiles @ d
    gaps = focal.max(axis=1)[:, None] - focal
    j, k = np.triu_indices(3, k=1)
    contrasts = (profiles[:, j] - profiles[:, k]).reshape(-1, 2)
    costs = (gaps[:, j] + gaps[:, k]).ravel()
    g = basis_check(contrasts, costs)
    at = 25.
    u = at * focal
    p = np.exp(u - logsumexp(u, axis=1)[:, None])
    mean = np.einsum("ij,ijd->id", p, profiles)
    centred = profiles - mean[:, None, :]
    cov = np.einsum("i,ij,ijd,ije->de", w, p, centred, centred)
    pair_weights = (w[:, None] * p[:, j] * p[:, k]).ravel()
    gram = contrasts.T @ (pair_weights[:, None] * contrasts)
    assert np.max(abs(gram - cov)) < 1e-12
    values = []
    for t in [100., 1000.]:
        u = t * focal
        l = u - logsumexp(u, axis=1)[:, None]
        row_logs = (np.log(w)[:, None] + l[:, j] + l[:, k]).ravel()
        log_terms = []
        for rows in itertools.combinations(range(len(contrasts)), 2):
            rows = list(rows)
            determinant = np.linalg.det(contrasts[rows])
            if abs(determinant) > 1e-12:
                log_terms.append(2 * np.log(abs(determinant)) + sum(row_logs[rows]))
        values.append(-logsumexp(log_terms))
    estimated = float((values[1] - values[0]) / 900.)
    assert abs(g - estimated) < 1e-10
    return {"minimum_basis_cost": g, "direct_logdet_rate": estimated,
            "pairwise_information_max_difference": float(np.max(abs(gram - cov)))}


def common_support_grid():
    """Check the rational constructions for all three outcomes on common support."""
    results = []
    for highest_even_moment in range(17):
        magnitudes = list(range(1, highest_even_moment + 3))
        p0 = [Fraction(1, len(magnitudes))] * len(magnitudes)
        signed = [Fraction(1, math.prod(u * u - v * v for v in magnitudes if v != u)) for u in magnitudes]
        step = min(p / (2 * abs(v)) for p, v in zip(p0, signed, strict=True))
        auxiliary = [[p + factor * step * v for p, v in zip(p0, signed, strict=True)] for factor in (-1, 0, 1)]
        human = [p / 2 for p in p0]
        human[-1] += Fraction(1, 2)
        moments = lambda probs, power: sum(p * u ** power for p, u in zip(probs, magnitudes, strict=True))
        human_absolute = moments(human, 1)
        target_slope = (human_absolute + moments(p0, 1)) / 4
        positive_choice = Fraction(1, 2) + target_slope / human_absolute
        assert Fraction(1, 2) < positive_choice < 1
        assert (positive_choice - Fraction(1, 2)) * human_absolute == target_slope
        auxiliary_counts = math.lcm(*(p.denominator * 2 for family in auxiliary for p in family))
        labelled_counts = math.lcm(*((p * t / 2).denominator for p in human
                                   for t in (positive_choice, 1 - positive_choice)))
        costs = []
        for family in auxiliary:
            assert sum(family) == 1 and min(family) > 0
            assert all(moments(family, 2*r) == moments(p0, 2*r)
                       for r in range(highest_even_moment + 1))
            assert all((p * auxiliary_counts / 2).denominator == 1 for p in family)
            costs.append((human_absolute + moments(family, 1)) / 4 - target_slope)
        assert min(costs) < 0 < max(costs) and costs[1] == 0
        results.append({"matched_order": 2 * highest_even_moment + 1, "paired_magnitudes": len(magnitudes),
                        "common_total_auxiliary_rows": auxiliary_counts, "labelled_rows": labelled_counts,
                        "positive_slope_recession_costs": [str(c) for c in costs]})
    return results


def read_witnesses(archive, data):
    check = archive / "results/public_vaccine_common_menu_frequency"
    sys.path[:0] = [str(check), str(archive / "results/official_ppi_runtime/reference/analysis")]
    from public_conjoint_optimizer_audit import NumericOnly
    from run import frozen_forecast, frequency_scenarios, moment, read_data, sha

    plan = json.loads((check / "plan.json").read_text())
    result_path = check / "results/results.json"
    result = json.loads(result_path.read_text())
    _, x, y, ai, source = read_data(data, plan, NumericOnly)
    assert source == result["source_commit"]
    train, aux, labels = (slice(*plan[field]) for field in ("training_rows", "auxiliary_library", "label_pool"))
    rng = np.random.default_rng(plan["allocation_seed"])
    blocks = np.arange(labels.start, labels.stop).reshape(-1, plan["source_block_size"])
    orders = [blocks[rng.permutation(len(blocks))].ravel() for _ in range(plan["allocation_count"])]
    forecasts = {}
    for name, response in zip(plan["frozen_forecasts"], [y[train], ai], strict=True):
        g, _ = frozen_forecast(x[train], response, x, plan["ridge_penalty_multiplier"])
        forecasts[name] = g, frequency_scenarios(g[aux], plan)

    rows = []
    for case in result["witnesses"]:
        weight, budget, allocation = (case[k] for k in ("weight", "budget", "allocation"))
        idx = orders[allocation][:budget]
        assert sha(idx.astype("<i8").tobytes()) == case["allocation_sha256"]
        predictor = case["forecast"]
        g, counts = forecasts[predictor]
        (design, witness), = [(key, val) for key, val in case["comparisons"].items()
                              if val["alternate_probability_status"] == "outside"]
        f = counts[design] / counts[design].sum()
        human, delta = moment(x[idx], y[idx], g[idx], x[aux], g[aux], f)
        target = human + weight * delta
        features = np.vstack((x[idx], x[aux]))
        w = np.r_[np.full(budget, (1 - weight) / budget), weight * f]
        keep = w > 0
        d = np.asarray(witness["direction"])
        assert abs(np.linalg.norm(d, ord=1) - 1.) < 1e-9
        directional = features @ d
        cost = float(w @ np.maximum(directional, 0) - target @ d)
        assert cost < -1e-8 and abs(cost - witness["direct_recession"]) < 1e-10
        basis = independent_basis(features[keep], abs(directional[keep]))
        per_row_bound = float(abs(directional[keep][basis]).sum())
        assert per_row_bound > 1e-8
        assert np.linalg.slogdet(features[keep][basis])[0] != 0
        coefficient = 1 / (2 * budget)
        bound = float(cost + coefficient * per_row_bound)
        # The displayed basis gives an upper bound for the penalty along this
        # direction without a determinant of a nearly singular matrix.
        theta = 20000. * d
        eta = features @ theta
        log_v = eta - 2. * np.logaddexp(0., eta)
        det_basis = np.linalg.slogdet(features[keep][basis])[1]
        log_lower_bound_det = float(2 * det_basis + sum(np.log(w[keep][basis]) + log_v[keep][basis]))
        unpenalized_probe = float(w @ np.logaddexp(0., eta) - target @ theta)
        probe_upper = unpenalized_probe - coefficient * log_lower_bound_det
        assert abs(probe_upper / 20000. - bound) < .005
        rows.append({"predictor": predictor, "budget": budget, "weight": weight, "allocation": allocation,
                     "frequency_scenario": design, "direct_recession": cost, "basis_cost_upper_bound": per_row_bound,
                     "active_basis_indices": basis.tolist(), "penalty": coefficient,
                     "penalized_directional_slope_upper_bound": bound,
                     "certified_unbounded_after_penalty": bound < -1e-8,
                     "directional_penalty_threshold_lower_bound": -cost / per_row_bound,
                     "loss_upper_bound_at_20000": probe_upper})
    return source, hashlib.sha256(result_path.read_bytes()).hexdigest(), rows


def main(args):
    archive = args.archive.resolve()
    toy_result = toy()
    multi = multinomial_check()
    source, source_result_sha, rows = read_witnesses(archive, args.data.resolve())
    assert len(rows) == 19
    retained = [r for r in rows if r["certified_unbounded_after_penalty"]]
    result = {"status": "passed", "exact_toy": toy_result, "multinomial_identity": multi,
              "exact_common_support_constructions": common_support_grid(),
              "public_source_commit": source, "prior_results_sha256": source_result_sha, "public_rays": len(rows),
              "public_rays_still_certify_unbounded": len(retained), "rows": rows}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{len(retained)} of {len(rows)} published-data rays still certify unboundedness.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    with threadpool_limits(limits=1):
        main(parser.parse_args())
