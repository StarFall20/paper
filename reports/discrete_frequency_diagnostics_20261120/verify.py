"""Compare three equal-support choice designs at one fixed correction weight."""

import argparse
from collections import Counter
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from math import factorial, gcd, lcm, log, prod
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, linprog
from scipy.special import expit

HERE = Path(__file__).resolve().parent
HALF = F(1, 2)


def records(levels, counts, labelled_counts=None):
    rows = []
    n = sum(counts)
    for z in (-1, 1):
        probability = HALF + F(z, 4)
        outcomes = None if labelled_counts is None else [1] * int(probability * n) + [0] * int((1 - probability) * n)
        rows.extend((0, z, None if outcomes is None else outcomes[j]) for j in range(n))
    for index, (u, count) in enumerate(zip(levels, counts)):
        for sign in (-1, 1):
            if labelled_counts is None:
                rows.extend((sign * u, 0, None) for _ in range(count))
            else:
                in_positive = labelled_counts[index]
                ones = in_positive if sign == 1 else count - in_positive
                rows.extend((sign * u, 0, int(j < ones)) for j in range(count))
    assert len(rows) == 4 * n
    return rows


def moment(rows, response):
    return tuple(sum(F(response(row)) * feature(row[0]) for row in rows) / len(rows)
                 for feature in (lambda x: 1, lambda x: x))


def forecast(row):
    return HALF + F(row[1], 4)


def log_partition_derivatives(order):
    # The first log-partition derivative solves p' = p(1-p), p(0)=1/2.
    p = [HALF]
    for degree in range(order - 1):
        square = sum(p[i] * p[degree - i] for i in range(degree + 1))
        p.append((p[degree] - square) / (degree + 1))
    return [p[k - 1] * factorial(k - 1) for k in range(1, order + 1)]


def exact_diagnostics(lab, aux, weight, order):
    label = moment(lab, lambda row: row[2] - weight * forecast(row))
    predicted = moment(aux, forecast)
    target = tuple(a + weight * b for a, b in zip(label, predicted))
    gradient = (HALF - target[0], -target[1])
    derivatives = {1: gradient}
    for degree, derivative in enumerate(log_partition_derivatives(order), 1):
        if degree > 1:
            derivatives[degree] = tuple(
                derivative * ((1 - weight) * F(sum(x**k for x, _, _ in lab), len(lab))
                              + weight * F(sum(x**k for x, _, _ in aux), len(aux)))
                for k in range(degree + 1)
            )
    scores = tuple(
        tuple((v, row[0] * v) for row in rows for v in [scalar(row)])
        for rows, scalar in (
            (lab, lambda row: HALF - row[2] + weight * (forecast(row) - HALF)),
            (aux, lambda row: weight * (HALF - forecast(row))),
        )
    )
    return target, derivatives, scores


def recession(rows_l, rows_u, target, weight, sign):
    support = sum(
        group_weight * F(sum(max(0, sign * row[0]) for row in rows), len(rows))
        for rows, group_weight in ((rows_l, 1 - weight), (rows_u, weight))
    )
    return support - sign * target[1]


def probability_check(lab, aux, target, weight):
    n = len(lab) + len(aux)
    features = np.array([[1, row[0]] for row in lab + aux], float)
    weights = np.r_[np.full(len(lab), float(1 - weight) / len(lab)),
                    np.full(len(aux), float(weight) / len(aux))]
    eq = np.column_stack([(features * weights[:, None]).T, np.zeros(2)])
    margin_column = np.ones((n, 1))
    lower = np.hstack([-np.eye(n), margin_column])
    upper = np.hstack([np.eye(n), margin_column])
    result = linprog(
        np.r_[np.zeros(n), -1], A_eq=eq, b_eq=np.array(target, float),
        A_ub=np.vstack([lower, upper]), b_ub=np.r_[np.zeros(n), np.ones(n)],
        bounds=[(0, 1)] * n + [(0, .5)], method="highs",
    )
    if result.status == 2:
        return "exterior", None, None
    assert result.success, result.message
    margin = float(result.x[-1])
    assert np.max(abs(eq @ result.x - np.array(target, float))) < 1e-10
    if margin < 1e-10:
        return "boundary", margin, None
    b = brentq(lambda beta: float((features.T @ (weights * expit(features[:, 1] * beta)))[1]) - float(target[1]),
               0, 20, xtol=1e-13)
    score = features.T @ (weights * expit(features[:, 1] * b)) - np.array(target, float)
    assert max(abs(score)) < 1e-12
    return "interior", margin, b


def verify():
    plan_path = HERE / "plan.json"
    plan = json.loads(plan_path.read_text())
    levels = plan["levels"]
    weight = F(plan["weight"])
    baseline = plan["auxiliary_pair_baseline"]
    differences = [1 / F(prod(u*u - v*v for v in levels if u != v)) for u in levels]
    multiplier = lcm(*(x.denominator for x in differences))
    direction = [int(x * multiplier) for x in differences]
    assert gcd(*direction) == 1 and direction == [-10, 15, -6, 1]
    orders = range(0, plan["matching_order"] + 1, 2)
    assert all(sum(d * u**k for d, u in zip(direction, levels)) == 0 for k in orders)
    base_sums = [sum(c * u**k for c, u in zip(baseline, levels)) for k in orders]
    counts = []
    n = sum(baseline)
    for a, b, c in product(range(1, n - 2), repeat=3):
        candidate = [a, b, c, n - a - b - c]
        if candidate[-1] > 0 and all(sum(col * u**k for col, u in zip(candidate, levels)) == target
                                     for k, target in zip(orders, base_sums)):
            counts.append(candidate)
    expected = [[c + s * d for c, d in zip(baseline, direction)] for s in (1, 0, -1)]
    assert set(map(tuple, counts)) == set(map(tuple, expected))
    lab = records(levels, plan["label_pair_counts"], plan["label_positive_side_one_counts"])
    outcomes = Counter(lab)
    assert all(outcomes[(x, z, y)] > 0 for x, z, _ in lab for y in (0, 1))
    support = {(x, z) for x, z, _ in lab}
    base_results, results, moment_six = None, [], []
    for index, counts in enumerate(expected):
        aux = records(levels, counts)
        assert {(x, z) for x, z, _ in aux} == support
        diagnostics = exact_diagnostics(lab, aux, weight, plan["matching_order"] + 1)
        target, derivatives, scores = diagnostics
        assert target == (HALF, F(2, 5))
        common = target, {k: v for k, v in derivatives.items() if k <= 5}, scores
        if base_results is None:
            base_results = common
        assert common == base_results
        moment_six.append(derivatives[6][-1])
        plus = recession(lab, aux, target, weight, 1)
        minus = recession(lab, aux, target, weight, -1)
        assert plus == F(1 - index, 48) and minus > 0
        lower, _, _ = probability_check(lab, aux, target, F(0))
        assert lower == "interior"
        state, margin, beta = probability_check(lab, aux, target, weight)
        assert state == ["interior", "boundary", "exterior"][index]
        human_gap = recession(lab, aux, target, F(0), 1)
        unit_gap = recession(lab, aux, target, F(1), 1)
        boundary = human_gap / (human_gap - unit_gap)
        assert boundary == [F(18, 31), HALF, F(18, 41)][index]
        results.append(dict(pair_counts=counts, positive_slope_recession=str(plus), negative_slope_recession=str(minus),
                            largest_feasible_weight=str(boundary), state=state, probability_margin=margin,
                            finite_slope=beta, auxiliary_rows=len(aux)))
    assert len(set(moment_six)) == 3
    loss_limit = ((1 - weight) * F(sum(x == 0 for x, _, _ in lab), len(lab))
                  + weight * F(sum(x == 0 for x, _, _ in records(levels, baseline)), 4 * n))
    assert loss_limit == F(433, 720)
    target, derivatives, _ = base_results
    return dict(status="passed", plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                labelled_rows=len(lab), common_profiles=len(support), common_target=list(map(str, target)),
                common_derivatives={k: list(map(str, v)) for k, v in derivatives.items()},
                pure_sixth_order_derivative=list(map(str, moment_six)), complete_integer_solutions=3,
                shared_objective_value=log(2), bounded_unattained_infimum_factor_log2=str(loss_limit),
                matches_all_score_rows=True, cases=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.write_text(json.dumps(verify(), indent=2) + "\n")
