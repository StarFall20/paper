"""Verify sharp correction-power bounds using the original choice rows."""

import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, linprog


def moment_interval(pair_count, total, cap):
    filled = min(pair_count, math.floor(total / cap**2))
    leftover = max(0.0, total - filled * cap**2)
    concentrated = [cap] * filled
    if filled < pair_count:
        concentrated += [math.sqrt(leftover)] + [0.0] * (pair_count - filled - 1)
    equal = [math.sqrt(total / pair_count)] * pair_count
    return concentrated, equal


def labelled_rows():
    groups = [(0.0, -1, 2), (0.0, 1, 6), (-0.5, 0, 1), (0.5, 0, 7)]
    return np.array([(1, x, 0.5 + z / 4, int(i < positives))
                     for x, z, positives in groups for i in range(8)])


def auxiliary_rows(nodes):
    pairs = [(1, sign * node, 0.5) for node in nodes for sign in (-1, 1)]
    fixed = [(1, 0.0, 0.5 + z / 4) for z in (-1, 1) for _ in nodes]
    return np.array(fixed + pairs)


def corrected_problem(labelled, auxiliary, power):
    label, aux = labelled[:, :2], auxiliary[:, :2]
    # Each feature row scales one independently constrained binary probability.
    coefficients = np.column_stack((
        (1 - power) * label.T / len(label),
        power * aux.T / len(aux),
    ))
    moment = (
        np.mean(label * (labelled[:, 3] - power * labelled[:, 2])[:, None], axis=0)
        + power * np.mean(aux * auxiliary[:, 2, None], axis=0)
    )
    return coefficients, moment


def independent_support_state(coefficients, moment):
    columns = coefficients.shape[1]
    a_ub = np.column_stack((np.vstack((np.eye(columns), -np.eye(columns))), np.ones(2 * columns)))
    a_eq = np.column_stack((coefficients, np.zeros(coefficients.shape[0])))
    optimum = linprog(
        np.r_[np.zeros(columns), -1], A_ub=a_ub, b_ub=np.r_[np.ones(columns), np.zeros(columns)],
        A_eq=a_eq, b_eq=moment, bounds=[(0, 1)] * columns + [(0, 0.5)], method="highs",
    )
    if optimum.status == 2:
        return "exterior"
    if not optimum.success:
        raise RuntimeError(optimum.message)
    return "interior" if optimum.x[-1] > 1e-8 else "boundary"


def diagnostics_at_zero(labelled, auxiliary, power):
    coefficients, moment = corrected_problem(labelled, auxiliary, power)
    label, aux = labelled[:, :2], auxiliary[:, :2]
    label_score = label * (0.5 * (1 - power) - labelled[:, 3] + power * labelled[:, 2])[:, None]
    aux_score = aux * (power * (0.5 - auxiliary[:, 2]))[:, None]
    hessian = ((1 - power) * label.T @ label / len(label) + power * aux.T @ aux / len(aux)) / 4
    return np.r_[math.log(2), coefficients @ np.full(coefficients.shape[1], 0.5) - moment,
                 hessian.ravel(), np.cov(label_score.T).ravel(), np.cov(aux_score.T).ravel()]


def direct_witness_check(labelled, nodes, power, expected=None):
    auxiliary = auxiliary_rows(nodes)
    coefficients, moment = corrected_problem(labelled, auxiliary, power)
    rays = np.array([[0., 1.], [0., -1.]])
    recession = np.maximum(rays @ coefficients, 0).sum(axis=1) - rays @ moment
    analytic = ((1 - power) / 2 + power * np.mean(nodes) - 3 / 8) / 4
    assert abs(recession[0] - analytic) < 1e-14 and recession[1] > 0
    state = independent_support_state(coefficients, moment)
    classified = ("interior" if recession[0] > 1e-10 else
                  "exterior" if recession[0] < -1e-10 else "boundary")
    assert state == classified and (expected is None or state == expected)
    return {"state": state, "positive_slope_recession": float(recession[0])}


def feasible_supremum(mean_magnitude):
    return min(1.0, (1 / 8) / max(1 / 8, 0.5 - mean_magnitude))


def verify_square_grid():
    count = 0
    for n in range(1, 8):
        observed_minimum = defaultdict(lambda: math.inf)
        for squared in itertools.product(range(5), repeat=n):
            total = sum(squared)
            observed_minimum[total] = min(
                observed_minimum[total], sum(math.sqrt(value / 4) for value in squared)
            )
        for total, minimum in observed_minimum.items():
            low, high = moment_interval(n, total / 4, 1.0)
            assert abs(sum(low) - minimum) < 2e-14
            assert abs(sum(high) ** 2 - n * total / 4) < 3e-13
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    label = labelled_rows()
    table = {
        "A": [0.5] * 8,
        "B": [1.0, 0.5, 0.5, 0.5, 0.5, 0., 0., 0.],
        "C": [1.0, 1.0, 0., 0., 0., 0., 0., 0.],
    }
    low, high = moment_interval(8, 2.0, 1.0)
    assert high == table["A"] and low == table["C"]
    table_states = {name: direct_witness_check(label, nodes, 1.0, state)
                    for (name, nodes), state in zip(table.items(), ("interior", "boundary", "exterior"))}
    assert [feasible_supremum(np.mean(nodes)) for nodes in table.values()] == [1.0, 1.0, 0.5]

    low, high = map(np.array, moment_interval(3, 0.75, 1.0))
    continuous_relaxation = [0.25, 0.5]
    assert np.mean(low) > continuous_relaxation[0]
    guaranteed_at_055 = direct_witness_check(label, low, 0.55, "interior")
    all_three_power = 0.6
    target = (3 / 8 - (1 - all_three_power) / 2) / all_three_power
    interpolate = lambda t: np.sqrt((1 - t) * low**2 + t * high**2)
    root = brentq(lambda t: np.mean(interpolate(t)) - target, 0.0, 1.0, xtol=1e-14)
    boundary = interpolate(root)
    witnesses = {"minimum_mean": low, "intermediate_mean": boundary, "maximum_mean": high}
    small_states = {name: direct_witness_check(label, nodes, all_three_power, state)
                    for (name, nodes), state in zip(witnesses.items(), ("exterior", "boundary", "interior"))}

    maximum_difference = 0.0
    for designs in (table, witnesses):
        for power in np.linspace(0.0, 1.0, 101):
            diagnostics = np.array([
                diagnostics_at_zero(label, auxiliary_rows(nodes), power) for nodes in designs.values()
            ])
            maximum_difference = max(maximum_difference, np.max(np.abs(diagnostics - diagnostics[0])))
    assert maximum_difference < 1e-14
    rng = np.random.default_rng(20261119)
    for number in range(1, 30):
        for _ in range(80):
            nodes = rng.uniform(0, 1, number)
            low_candidate, high_candidate = moment_interval(number, nodes @ nodes, 1.0)
            assert sum(low_candidate) - 1e-13 <= sum(nodes) <= sum(high_candidate) + 1e-13
    result = {
        "status": "passed",
        "grid_minimum_checks": verify_square_grid(),
        "continuous_design_bound_checks": 29 * 80,
        "power_values_checked_per_design_set": 101,
        "largest_zero_diagnostic_difference": float(maximum_difference),
        "table_1": {
            "mean_magnitude_interval": [0.25, 0.5], "existence_limit_interval": [0.5, 1.0],
            "unit_states": table_states,
        },
        "three_pair_example": {
            "mean_magnitude_interval": [float(np.mean(low)), float(np.mean(high))],
            "continuous_relaxation_interval": continuous_relaxation,
            "smallest_existence_limit": feasible_supremum(np.mean(low)),
            "worst_case_at_0_55": guaranteed_at_055,
            "states_at_0_6": small_states,
            "matched_magnitudes": {name: nodes.tolist() for name, nodes in witnesses.items()},
            "bach_at_zero": {
                "newton_decrement": math.sqrt(9 / 32),
                "sufficient_threshold_with_R_sqrt2": 1 / 16,
            },
        },
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
