"""Compute exact existence probabilities for fixed menus and random oracle labels."""

import argparse
from collections import Counter
from fractions import Fraction
from itertools import product
import json
from math import comb, factorial, lcm, prod
from pathlib import Path
import sys


def sum_distribution(rows, forecast):
    # Every configuration shares one denominator. All updates are integer exact.
    distribution = {(0, 0): 1}
    for x, n in rows.items():
        p = forecast[x]
        terms = [
            comb(n, k) * p.numerator**k * (p.denominator - p.numerator)**(n - k)
            for k in range(n + 1)
        ]
        updated = Counter()
        for (count, moment), value in distribution.items():
            for k, coefficient in enumerate(terms):
                updated[count + k, moment + k*x] += value*coefficient
        distribution = updated
    denominator = prod(forecast[x].denominator**n for x, n in rows.items())
    assert sum(distribution.values()) == denominator
    return distribution, denominator


def bound(rows, auxiliary, forecast, weight):
    n_labels = sum(rows.values())
    human = {x: Fraction(n, n_labels) for x, n in rows.items()}
    coefficients = {x: (1 - weight)*human[x] + weight*auxiliary[x] for x in rows}
    shift = {x: weight*(auxiliary[x] - human[x])*forecast[x] for x in rows}
    facets = []
    for cutoff in rows:
        for sign in (-1, 1):
            upper = n_labels*sum(
                coefficients[x]*max(0, sign*(x - cutoff)) - shift[x]*sign*(x - cutoff)
                for x in rows
            )
            facets.append((cutoff, sign, upper.numerator, upper.denominator))
    return facets


def state(count, moment, facets):
    tight = False
    for cutoff, sign, numerator, denominator in facets:
        gap = numerator - sign*(moment - cutoff*count)*denominator
        if gap < 0:
            return "exterior"
        tight |= gap == 0
    return "boundary" if tight else "interior"


def cross(a, b, c):
    return (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])


def attainable_polygon(rows, auxiliary, forecast, weight):
    n = sum(rows.values())
    human = {x: Fraction(n_x, n) for x, n_x in rows.items()}
    shift = tuple(sum(weight*(auxiliary[x] - human[x])*forecast[x]*x**j for x in rows)
                  for j in range(2))
    scale = {x: (1 - weight)*human[x] + weight*auxiliary[x] for x in rows}
    vertices = sorted({
        tuple(n*(sum(scale[x]*q*x**j for x, q in zip(rows, flags, strict=True)) - shift[j])
              for j in range(2))
        for flags in product((0, 1), repeat=len(rows))
    })

    def half(points):
        edges = []
        for point in points:
            while len(edges) >= 2 and cross(edges[-2], edges[-1], point) <= 0:
                edges.pop()
            edges.append(point)
        return edges[:-1]

    hull = half(vertices) + half(reversed(vertices))
    return list(zip(hull, hull[1:] + hull[:1], strict=True))


def check_states_against_hull(distribution, facets, hull):
    assert len(hull) == len(facets)
    for key in distribution:
        edge_values = [cross(a, b, key) for a, b in hull]
        if min(edge_values) < 0:
            expected = "exterior"
        elif min(edge_values) == 0:
            expected = "boundary"
        else:
            expected = "interior"
        assert state(*key, facets) == expected


def verify_derivative_agreement(auxiliary, forecast, order):
    # Independently expand the fitted probability at the true coefficients.
    # The nth derivative of the log partition is the (n-1)th derivative of this probability.
    exponential = [Fraction(1, factorial(k)) for k in range(order)]
    expansions = {}
    for x, p in forecast.items():
        series = [p]
        for k in range(1, order):
            series.append(p*(exponential[k] - sum(exponential[j]*series[k-j] for j in range(1, k+1))))
        expansions[x] = series
    for derivative in range(2, order+1):
        for j in range(derivative+1):
            moments = [
                sum(weights[x]*expansions[x][derivative-1]*x**j for x in forecast)
                for weights in auxiliary
            ]
            assert moments[0] == moments[1] == moments[2]


def frequency_designs(oracle_cases, order):
    support, vectors = oracle_cases.independent_magnitudes(oracle_cases.derivative_polynomials(order))
    size = len(support)
    movement = oracle_cases.solve_exact(list(zip(*vectors, strict=True)),
                                       [Fraction(0)]*(size-1) + [Fraction(1)])
    base = Fraction(1, size)
    epsilon = min(base/(2*abs(value)) for value in movement if value)
    auxiliary = [
        {x: (base + direction*epsilon*value)/2
         for u, value in zip(support, movement, strict=True) for x in (-u, u)}
        for direction in (-1, 0, 1)
    ]
    assert all(sum(weights.values()) == 1 and min(weights.values()) > 0 for weights in auxiliary)
    most = max(support, key=lambda u: u*(1 - oracle_cases.oracle(u)))
    return auxiliary, most


def audit_setting(auxiliary, distinguished, forecast, replications, order, concentrated):
    rows = {
        x: replications*(len(forecast)//2 + 1 if concentrated and abs(x) == distinguished else 1)
        for x in forecast
    }
    distribution, denominator = sum_distribution(rows, forecast)
    n_labels = sum(rows.values())
    if n_labels <= 10:
        # Verify the compressed probability calculation by counting every outcome vector.
        x_rows = [x for x, n in rows.items() for _ in range(n)]
        direct = Counter()
        for y in product((0, 1), repeat=n_labels):
            probability = prod(forecast[x] if label else 1-forecast[x]
                               for x, label in zip(x_rows, y, strict=True))
            key = (sum(y), sum(x*label for x, label in zip(x_rows, y, strict=True)))
            direct[key] += probability
        assert direct == {k: Fraction(v, denominator) for k, v in distribution.items()}
    ordinary = bound(rows, auxiliary[0], forecast, Fraction(0))
    corrected = [bound(rows, w, forecast, Fraction(1, 2)) for w in auxiliary]
    hull_checks = 0
    if order <= 2 and replications == 1:
        check_states_against_hull(distribution, ordinary, attainable_polygon(rows, auxiliary[0], forecast, Fraction(0)))
        for w, facets in zip(auxiliary, corrected, strict=True):
            check_states_against_hull(distribution, facets, attainable_polygon(rows, w, forecast, Fraction(1, 2)))
        hull_checks = 4*len(distribution)
    strata = Counter()
    finite = 0
    for (count, moment), value in distribution.items():
        normal = state(count, moment, ordinary)
        assert normal in ("boundary", "interior")
        if normal == "interior":
            finite += value
            key = tuple(state(count, moment, facets) for facets in corrected)
            strata[key] += value
    assert finite > 0 and sum(strata.values()) == finite
    events = {
        "any_exterior_and_interior": lambda states: "exterior" in states and "interior" in states,
        "states_differ": lambda states: len(set(states)) > 1,
        "all_three_states": lambda states: len(set(states)) == 3,
        "any_exterior": lambda states: "exterior" in states,
    }
    return dict(order=order, label_allocation="concentrated" if concentrated else "balanced",
                labelled_rows=n_labels, labelled_profiles=rows, distinct_moments=len(distribution),
                common_auxiliary_rows=lcm(*(p.denominator for w in auxiliary for p in w.values())),
                independent_hull_states_verified=hull_checks, brute_force_distribution_verified=n_labels <= 10,
                ordinary_finite=float(Fraction(finite, denominator)),
                exact_ordinary_finite=str(Fraction(finite, denominator)),
                conditional_probabilities={
                    name: float(Fraction(sum(mass for states, mass in strata.items() if keep(states)), finite))
                    for name, keep in events.items()
                },
                exact_joint_strata={" ".join(k): str(Fraction(v, finite)) for k, v in sorted(strata.items())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle_case_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.oracle_case_dir.resolve()))
    import verify as oracle_cases

    results = []
    for order in range(1, 5):
        weights, distinguished = frequency_designs(oracle_cases, order)
        forecast = {x: oracle_cases.oracle(x) for x in sorted(weights[0])}
        verify_derivative_agreement(weights, forecast, order)
        for concentrated in (False, True):
            for n in (1, 2, 4, 8):
                result = audit_setting(weights, distinguished, forecast, n, order, concentrated)
                results.append(result)
                print(order, concentrated, result["labelled_rows"], result["ordinary_finite"],
                      result["conditional_probabilities"], flush=True)
    payload = json.dumps(results, indent=2) + "\n"
    if args.reference:
        assert payload == args.reference.read_text()
    args.out.write_text(payload)


if __name__ == "__main__":
    main()
