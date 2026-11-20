"""Build exact menu-frequency counterexamples at a calibrated logistic model."""

import argparse
from fractions import Fraction as F
import json
from math import lcm
from pathlib import Path


def derivative_polynomials(order):
    """Derivatives of softplus in p = logistic(t)."""
    polys = {1: [0, 1]}
    for i in range(2, order + 1):
        prior = polys[i - 1]
        result = [0] * (len(prior) + 1)
        for j, value in enumerate(prior[1:], 1):
            result[j] += j * value
            result[j + 1] -= j * value
        polys[i] = result
    return polys


def poly_at(poly, p):
    result = F(0)
    for coef in reversed(poly):
        result = result * p + coef
    return result


def oracle(u):
    ratio = F(2) ** u
    return ratio / (1 + ratio)


def constraints(u, polynomials):
    features = [F(1)]
    p = oracle(u)
    for r, poly in polynomials.items():
        if r == 1:
            continue
        v = poly_at(poly, p)
        assert poly_at(poly, 1 - p) == (-1) ** r * v
        features.extend(F(u ** j) * v for j in range(r + 1) if (r + j) % 2 == 0)
    return features + [u * (1 - p)]


def independent_magnitudes(polynomials):
    """Take the first positive integers extending the row span."""
    basis = {}
    points, rows = [], []
    dimension = len(constraints(1, polynomials))
    candidate = 0
    while len(points) < dimension:
        candidate += 1
        assert candidate < 4 * dimension
        row = constraints(candidate, polynomials)
        reduced = row.copy()
        for pivot, source in sorted(basis.items()):
            if reduced[pivot]:
                reduced = [a - reduced[pivot] * b for a, b in zip(reduced, source, strict=True)]
        lead = next((i for i, value in enumerate(reduced) if value), None)
        if lead is None:
            continue
        scale = reduced[lead]
        basis[lead] = [a / scale for a in reduced]
        points.append(candidate)
        rows.append(row)
    assert points[:2] == [1, 2]
    return points, rows


def solve_exact(matrix, target):
    """Gaussian elimination on a square rational system."""
    a = [list(row) + [y] for row, y in zip(matrix, target, strict=True)]
    n = len(a)
    for i in range(n):
        pivot = next(j for j in range(i, n) if a[j][i])
        a[i], a[pivot] = a[pivot], a[i]
        scale = a[i][i]
        a[i] = [u / scale for u in a[i]]
        for j in range(i + 1, n):
            multiplier = a[j][i]
            if multiplier:
                a[j] = [u - multiplier * v for u, v in zip(a[j], a[i], strict=True)]
    sol = [F(0)] * n
    for i in range(n - 1, -1, -1):
        sol[i] = a[i][-1] - sum(a[i][j] * sol[j] for j in range(i + 1, n))
    assert [sum(u * v for u, v in zip(row, sol, strict=True)) for row in matrix] == target
    return sol


def mean(weight, values):
    return sum(p * v for p, v in zip(weight, values, strict=True))


def label_counts(weight, positive_fraction):
    cell_probabilities = [p * q / 2 for p in weight for q in (positive_fraction, 1 - positive_fraction)]
    size = lcm(*(p.denominator for p in cell_probabilities))
    assert min(int(p * size) for p in cell_probabilities) > 0
    return size


def example(order):
    polys = derivative_polynomials(order)
    support, rows = independent_magnitudes(polys)
    dimension = len(rows)
    matrix = list(zip(*rows, strict=True))
    movement = solve_exact(matrix, [F(0)] * (dimension - 1) + [F(1)])
    base = [F(1, dimension)] * dimension
    epsilon = min(p / (2 * abs(v)) for p, v in zip(base, movement, strict=True) if v)
    designs = [[p + s * epsilon * v for p, v in zip(base, movement, strict=True)]
               for s in (-1, 0, 1)]
    forecast = [oracle(u) for u in support]
    assert all(F(0) < v < F(1) for v in forecast)
    assert all(a < b for a, b in zip(forecast, forecast[1:]))
    assert all(oracle(-u) == 1 - g for u, g in zip(support, forecast, strict=True))
    risks = [F(u) * (1 - g) for u, g in zip(support, forecast, strict=True)]
    most = max(range(dimension), key=risks.__getitem__)
    human = [p / 2 for p in base]
    human[most] += F(1, 2)
    h = [F(u) * (g - F(1, 2)) for u, g in zip(support, forecast, strict=True)]
    lam = F(1, 2)
    human_abs = mean(human, support)
    human_choice_moment = human_abs / 2 - lam * (mean(human, risks) - mean(base, risks))
    human_plus_choice = F(1, 2) + human_choice_moment / human_abs
    assert F(1, 2) < human_plus_choice < F(1)
    m = label_counts(human, human_plus_choice)
    n = 2 * lcm(*(p.denominator for w in designs for p in w))
    result = []
    for w in designs:
        assert sum(w) == 1 and min(w) > 0
        assert all((p * n / 2).denominator == 1 for p in w)
        assert all(mean(w, a) == mean(base, a) for a in matrix[:-1])
        target = human_choice_moment + lam * (mean(w, h) - mean(human, h))
        support_slope = ((1 - lam) * human_abs + lam * mean(w, support)) / 2
        positive, negative = support_slope - target, support_slope + target
        assert positive == lam * (mean(w, risks) - mean(base, risks))
        assert negative > 0
        result.append((positive, negative))
    assert [r[0] for r in result] == [-lam * epsilon, F(0), lam * epsilon]
    return {"highest_matched_derivative_at_truth": order, "design_magnitudes": support,
            "labelled_rows": m, "common_auxiliary_rows": n,
            "human_choice_fraction_at_positive_profile": str(human_plus_choice),
            "positive_recession": [str(cost[0]) for cost in result],
            "negative_recession": [str(cost[1]) for cost in result]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Expected byte-identical result")
    args = parser.parse_args()
    results = [example(k) for k in range(1, 10)]
    payload = json.dumps({"status": "passed", "examples": results}, indent=2) + "\n"
    if args.reference:
        assert args.reference.read_text() == payload
    args.out.write_text(payload)
    print(f"Exact arithmetic checked all three outcomes through orders 1–{len(results)} at the true coefficient.")
