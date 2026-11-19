"""Reproduce a matched two-point example with equal empirical row weights."""

import argparse
from decimal import Decimal, localcontext
import json
from pathlib import Path


D = Decimal
ZERO = D(0)
ONE = D(1)
HALF = D('0.5')
SPAN = D('0.75')
SEED_NODES = tuple(map(D, ('0.04', '0.12', '0.23', '0.37', '0.55', '0.72')))


def probability(u):
    return ONE / (ONE + (-u).exp())


def matching_functions(x, step):
    p = probability(step * x)
    curvature = p * (ONE - p)
    centred_partition = (ONE + (step * x).exp()).ln() - HALF * step * x
    return (x * x, centred_partition, x * (p - HALF), curvature,
            x * x * curvature, x)


def matching_function_derivatives(x, step):
    p = probability(step * x)
    curvature = p * (ONE - p)
    curvature_derivative = step * curvature * (ONE - 2 * p)
    return (2 * x, step * (p - HALF), p - HALF + step * x * curvature,
            curvature_derivative, 2 * x * curvature + x * x * curvature_derivative, ONE)


def totals(nodes, step):
    return [sum(column) for column in zip(*(matching_functions(x, step) for x in nodes))]


def solve_linear(matrix, right):
    rows = [list(row) + [value] for row, value in zip(matrix, right)]
    n = len(rows)
    for j in range(n):
        pivot = max(range(j, n), key=lambda i: abs(rows[i][j]))
        rows[j], rows[pivot] = rows[pivot], rows[j]
        scale = rows[j][j]
        if scale == 0:
            raise ArithmeticError('Singular matching Jacobian')
        rows[j] = [v / scale for v in rows[j]]
        for i in range(n):
            if i != j:
                coefficient = rows[i][j]
                rows[i] = [a - coefficient * b for a, b in zip(rows[i], rows[j])]
    return [row[-1] for row in rows]


def matched_nodes(offset, step):
    nodes = list(SEED_NODES)
    target = totals(nodes, step)
    target[-1] += offset
    for _ in range(20):
        residual = [goal - got for goal, got in zip(target, totals(nodes, step))]
        if max(map(abs, residual)) < D('1e-65'):
            assert all(ZERO < x < SPAN for x in nodes)
            return nodes
        jacobian = list(zip(*(matching_function_derivatives(x, step) for x in nodes)))
        nodes = [x + change for x, change in zip(nodes, solve_linear(jacobian, residual))]
    raise ArithmeticError('The fixed two-point example did not converge')


def labelled_rows():
    groups = [(ZERO, -1, 2), (ZERO, 1, 6), (-HALF, 0, 1), (HALF, 0, 7)]
    return [(x, z, int(i < choices)) for x, z, choices in groups for i in range(8)]


def auxiliary_rows(nodes):
    magnitudes = [*nodes, *(SPAN - x for x in SEED_NODES)]
    pairs = [(sign * x, 0) for x in magnitudes for sign in (-1, 1)]
    informative = [(ZERO, sign) for sign in (-1, 1) for _ in magnitudes]
    return informative + pairs


def feature_vector(x):
    return (ONE, x)


def average(vectors, width):
    return tuple(sum(row[j] for row in vectors) / len(vectors) for j in range(width))


def row_diagnostics(rows, alpha, beta, field):
    losses, gradients, hessians, second_moments, thirds = [], [], [], [], []
    for x, z, *values in rows:
        response = D(values[0]) if field == 'observed' else HALF + D(z) / 4
        features = feature_vector(x)
        u = alpha + beta * x
        p = ONE / (ONE + (-u).exp())
        loss = (ONE + u.exp()).ln() - response * u
        score = tuple((p - response) * f for f in features)
        hessian = tuple(p * (1 - p) * (x ** j if j else ONE) for j in range(3))
        second_moment = tuple(score[j] * score[k] for j, k in ((0, 0), (0, 1), (1, 1)))
        third = tuple(p * (ONE - p) * (ONE - 2 * p) * (x ** j if j else ONE) for j in range(4))
        losses.append((loss,))
        gradients.append(score)
        hessians.append(hessian)
        second_moments.append(second_moment)
        thirds.append(third)
    return {
        'objective': average(losses, 1), 'gradient': average(gradients, 2),
        'hessian': average(hessians, 3), 'score_second_moments': average(second_moments, 3),
        'third_derivatives': average(thirds, 4),
    }


def corrected_diagnostics(labelled, auxiliary, alpha, beta):
    observed = row_diagnostics(labelled, alpha, beta, 'observed')
    label_predictions = row_diagnostics(labelled, alpha, beta, 'predicted')
    aux = row_diagnostics(auxiliary, alpha, beta, 'predicted')
    result = {key: tuple(a + b - c for a, b, c in zip(values, observed[key], label_predictions[key]))
              for key, values in aux.items() if key != 'score_second_moments'}
    result['auxiliary_score_second_moments'] = aux['score_second_moments']
    return result


def recession(labelled, auxiliary, direction):
    x_for = lambda x: x * direction

    def prediction(z):
        return HALF + D(z) / 4

    def weighted(records, f):
        return sum(f(*row) for row in records) / len(records)

    return (weighted(auxiliary, lambda x, z: max(ZERO, x_for(x)) - prediction(z) * x_for(x))
            + weighted(labelled, lambda x, z, y: (prediction(z) - y) * x_for(x)))


def find_finite_slope(labelled, auxiliary):
    def gradient(beta):
        return corrected_diagnostics(labelled, auxiliary, ZERO, beta)['gradient'][1]

    low, high = ZERO, ONE
    while gradient(high) < ZERO:
        high *= 2
        assert high <= 1024
    for _ in range(150):
        middle = (low + high) / 2
        if gradient(middle) < 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def newton_with_score_tolerance(labelled, auxiliary, tolerance):
    beta = ZERO
    for step in range(100):
        fit = corrected_diagnostics(labelled, auxiliary, ZERO, beta)
        score = fit['gradient'][1]
        if abs(score) <= tolerance:
            return {'stopping_iteration': step, 'slope': str(beta), 'slope_score': str(score)}
        beta -= score / fit['hessian'][2]
    raise ArithmeticError('The score-tolerance Newton run did not stop')


def run(offset):
    labelled = labelled_rows()
    seed = auxiliary_rows(SEED_NODES)
    base = corrected_diagnostics(labelled, seed, ZERO, ZERO)
    step = -base['gradient'][1] / base['hessian'][2]
    designs = {name: auxiliary_rows(matched_nodes(shift, step))
               for name, shift in (('finite', offset), ('boundary', ZERO), ('exterior', -offset))}
    assert len({len(aux) for aux in designs.values()}) == 1
    reference = corrected_diagnostics(labelled, designs['boundary'], ZERO, ZERO)
    first_newton_slope = -reference['gradient'][1] / reference['hessian'][2]
    assert first_newton_slope == step
    differences = {}
    for beta in (ZERO, step):
        base = corrected_diagnostics(labelled, designs['boundary'], ZERO, beta)
        fields = tuple(base) if beta == ZERO else tuple(k for k in base if k != 'third_derivatives')
        for name in ('finite', 'exterior'):
            other = corrected_diagnostics(labelled, designs[name], ZERO, beta)
            differences[f'{name}_at_{beta}'] = max(
                abs(v - w) for field in fields for v, w in zip(base[field], other[field]))
    assert max(differences.values()) < D('1e-60')
    rates = {name: tuple(recession(labelled, rows, d) for d in (1, -1))
             for name, rows in designs.items()}
    assert all(pair[1] > ZERO for pair in rates.values())
    assert rates['finite'][0] > 0 and rates['exterior'][0] < 0
    assert abs(rates['boundary'][0]) < D('1e-75')
    finite_slope = find_finite_slope(labelled, designs['finite'])
    finite_root = corrected_diagnostics(labelled, designs['finite'], ZERO, finite_slope)
    assert max(map(abs, finite_root['gradient'])) < D('1e-40')
    second = corrected_diagnostics(labelled, designs['boundary'], ZERO, step)
    return {
        'arithmetic_precision_digits': 85, 'labelled_rows': len(labelled),
        'auxiliary_rows_each': len(designs['boundary']),
        'first_newton_slope': str(first_newton_slope),
        'common_gradient_at_first_newton_point': [str(v) for v in second['gradient']],
        'maximum_absolute_diagnostic_differences': {k: str(v) for k, v in differences.items()},
        'recession_positive_and_negative_slope': {k: list(map(str, v)) for k, v in rates.items()},
        'finite_slope': str(finite_slope),
        'newton_stopping_at_1e_minus_6': {k: newton_with_score_tolerance(labelled, v, D('1e-6'))
                                           for k, v in designs.items()},
        'finite_score_max_abs': str(max(map(abs, finite_root['gradient']))),
        'matching_offset_in_magnitude_sum': str(offset),
        'auxiliary_variable_positive_magnitudes': {
            k: [str(x) for x, z in v if z == 0 and x > 0] for k, v in designs.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with localcontext() as context:
        context.prec = 85
        result = run(D('0.000001'))
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('labelled_rows', 'auxiliary_rows_each',
                    'first_newton_slope', 'recession_positive_and_negative_slope', 'finite_slope')}))


if __name__ == '__main__':
    main()
