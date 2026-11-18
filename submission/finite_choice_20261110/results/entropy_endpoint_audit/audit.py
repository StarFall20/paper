"""Check entropy duality and tuning-path derivatives on saved positive-endpoint fits."""
import argparse
import csv
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import null_space
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def summary_stats(values):
    return {name: float(x) for name, x in zip(('median', 'p95', 'max'), np.quantile(values, [0.5, 0.95, 1]))}


def menu_terms(features, coefficient, mask=None):
    utilities = features @ coefficient
    if mask is not None:
        utilities = np.where(mask, utilities, -np.inf)
    log_z = logsumexp(utilities, axis=1)
    log_prob = utilities - log_z[:, None]
    probabilities = np.exp(log_prob)
    means = np.einsum('nj,njp->np', probabilities, features)
    entropy = -np.einsum('nj,nj->n', probabilities, np.where(probabilities > 0, log_prob, 0.0))
    return log_z, means, entropy, probabilities


def moment(features, probability):
    return np.einsum('nj,njp->p', probability, features) / len(features)


def audit_case(data, predictors, rows, coefficients, case, labelled, auxiliary):
    labelled_f = data['f'][labelled].reshape(-1, 3, 11)
    auxiliary_f = data['f'][auxiliary].reshape(-1, 3, 11)
    evaluation = data['f'][data['test']].reshape(-1, 3, 11)
    eval_choice = data['y'][data['test']].ravel()
    choices = data['y'][labelled].ravel()
    b0 = labelled_f[np.arange(len(choices)), choices].mean(axis=0)
    db = moment(auxiliary_f, predictors[auxiliary].reshape(-1, 3)) - moment(
        labelled_f, predictors[labelled].reshape(-1, 3)
    )
    features = np.concatenate((labelled_f, auxiliary_f))
    split = len(labelled_f)
    contrast = np.rint(20 * (features - features[:, -1:, :])).astype(np.int64)
    assert np.max(np.abs(contrast / 20 - (features - features[:, -1:, :]))) < 1e-12
    integer = np.asarray(case['exact_integer_normal'], dtype=np.int64)
    norm = np.linalg.norm(integer)
    direction = integer / norm
    normal_utils = contrast @ integer
    gaps = (normal_utils.max(axis=1, keepdims=True) - normal_utils) / (20 * norm)
    face = gaps == 0
    delta_star = gaps[gaps > 0].min()
    assert abs(delta_star - case['minimum_normal_gap']) < 1e-14
    maximum = (features @ direction).max(axis=1)
    a = maximum[:split].mean() - maximum[split:].mean() + db @ direction
    assert abs(a - case['slack_slope']) < 1e-12
    endpoint = case['recomputed_cap']
    face_coef = np.asarray(case['tangent_coefficients'])
    face_z, face_means, face_entropy, _ = menu_terms(features, face_coef, face)
    mixing = lambda values, power: (1-power) * values[:split].mean(axis=0) + power * values[split:].mean(axis=0)
    difference = lambda values: values[split:].mean(axis=0) - values[:split].mean(axis=0)
    endpoint_value = mixing(face_entropy, endpoint)
    endpoint_score = mixing(face_means, endpoint) - b0 - endpoint * db
    face_value = mixing(face_z, endpoint) - (b0 + endpoint * db) @ face_coef
    assert max(abs(endpoint_score)) < 1e-9
    assert abs(face_value - endpoint_value) < 1e-10
    B = difference(face_z) - db @ face_coef
    A = case['leading_slack_constant']
    rate = a / delta_star
    intercept = math.log(A / a)
    tangent_basis = null_space(direction[None, :])
    tangent_features = features @ tangent_basis
    records = []
    for row in rows:
        number = int(row['record'])
        power = float(row['power'])
        distance = endpoint - power
        assert distance > 0
        beta = coefficients[number]
        t = direction @ beta
        # Relative menu utilities use the separately reconstructed integer gaps.
        theta = beta - t * direction
        relative = tangent_features @ (tangent_basis.T @ theta) - t * gaps
        log_z = logsumexp(relative, axis=1)
        log_prob = relative - log_z[:, None]
        probabilities = np.exp(log_prob)
        means = np.einsum('nj,njp->np', probabilities, features)
        entropy = -np.einsum('nj,nj->n', probabilities, log_prob)
        fitted_entropy = mixing(entropy, power)
        fitted_value = a * distance * t + mixing(log_z, power) - (b0 + power * db) @ theta
        score = mixing(means, power) - b0 - power * db
        assert max(abs(score)) < 1e-10
        assert abs(fitted_value - fitted_entropy) < 2e-10
        derivative = -a * t + difference(log_z) - db @ theta

        # Use explicit relative utilities for the normal entries of the Hessian.
        coords = np.concatenate((-gaps[:, :, None], tangent_features), axis=-1)
        local_means = np.einsum('nj,njp->np', probabilities, coords)
        offsets = coords - local_means[:, None, :]
        weight = np.r_[np.full(split, (1-power) / split),
                       np.full(len(features)-split, power / (len(features)-split))]
        hessian = np.einsum('n,nj,nja,njb->ab', weight, probabilities, offsets, offsets)
        power_gradient = np.r_[-a + difference(local_means[:, 0]),
                               difference(local_means[:, 1:]) - tangent_basis.T @ db]
        coefficient_derivative = -np.linalg.solve(hessian, power_gradient)
        curvature = power_gradient @ coefficient_derivative
        assert curvature < 0
        beta_derivative = direction * coefficient_derivative[0] + tangent_basis @ coefficient_derivative[1:]
        eval_z, eval_means, _, _ = menu_terms(evaluation, beta)
        actual = evaluation[np.arange(len(eval_choice)), eval_choice]
        eval_score = (eval_means - actual).mean(axis=0)
        eval_derivative = eval_score @ beta_derivative
        eval_loss = (eval_z - actual @ beta).mean()
        assert abs(eval_loss - float(row['heldout_logloss'])) < 1e-10
        log_distance = math.log(1 / distance) + intercept
        value_expansion = rate * (log_distance + 1) - B
        derivative_expansion = -rate * log_distance + B
        records.append(dict(
            frontier_record=int(row['frontier_record']), fit_record=number,
            labelled_people=int(row['labelled_people']), relative_distance=float(row['relative_distance']),
            lambda_value=power, distance=distance, rate=rate, endpoint_entropy=endpoint_value,
            fitted_entropy=fitted_entropy, fitted_objective=fitted_value,
            entropy_value_identity_abs_error=abs(fitted_entropy-fitted_value),
            difference_per_distance=(fitted_value-face_value)/distance, predicted_difference_per_distance=value_expansion,
            training_derivative=derivative, predicted_training_derivative=derivative_expansion,
            scaled_training_curvature=-distance*curvature/rate,
            scaled_normal_coefficient_derivative=distance*delta_star*coefficient_derivative[0],
            heldout_loss=eval_loss, heldout_derivative=eval_derivative,
            scaled_heldout_derivative=distance*eval_derivative,
            predicted_scaled_heldout_derivative=case['heldout_slope_per_log_distance'],
        ))
    return records


def audit(args):
    path = args.paths
    replay = read_json(path / 'isolated_replay.json')
    chosen = ['results/cases.json', 'results/paths.csv', 'results/coefficients.json.gz']
    assert all(sha(path / rel) == replay['sha256'][rel] for rel in chosen)
    sys.path.insert(0, str(args.frontier / 'analysis'))
    from existing_panel_model import load

    plan = read_json(args.frontier / 'inputs/model_plan.json')
    data = load(args.workbook, plan)
    assert sha(args.workbook) == read_json(path / 'results/status.json')['workbook_sha256']
    teacher = np.asarray(read_json(args.frontier / 'inputs/preflight.json')['teacher_coefficients'])
    util = data['teacher'] @ teacher
    prediction = np.exp(util - logsumexp(util, axis=2, keepdims=True))
    rng = np.random.default_rng(plan['allocation_seed'])
    orders = [rng.permutation(data['pool']) for _ in range(500)]
    coeffs = {number: np.asarray(coef) for number, coef in json.loads(
        gzip.decompress((path / 'results/coefficients.json.gz').read_bytes())
    )}
    cases = {c['frontier_record']: c for c in read_json(path / 'results/cases.json')
             if c['group'] == 'uncapped_positive'}
    paths = defaultdict(list)
    for row in read_csv(path / 'results/paths.csv'):
        if int(row['frontier_record']) in cases:
            paths[int(row['frontier_record'])].append(row)
    records = []
    assert len(cases) == 107 and set(cases) == set(paths)
    for number, case in cases.items():
        order = orders[case['allocation']]
        budget = case['labelled_people']
        records.extend(audit_case(data, prediction, paths[number], coeffs, case, order[:budget], order[budget:]))
    assert len(records) == 856 and len({r['fit_record'] for r in records}) == 856
    args.out.mkdir(parents=True)
    with (args.out / 'records.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    groups = defaultdict(list)
    for record in records:
        groups[record['relative_distance']].append(record)
    stats = []
    for distance, values in sorted(groups.items(), reverse=True):
        assert len(values) == 107
        stats.append(dict(
            relative_distance=distance, fits=len(values),
            negative_training_derivative=int(sum(r['training_derivative'] < 0 for r in values)),
            positive_holdout_derivative=int(sum(r['heldout_derivative'] > 0 for r in values)),
            opposing_derivatives=int(sum(r['training_derivative'] < 0 < r['heldout_derivative'] for r in values)),
            objective_expansion_per_distance_error=summary_stats([abs(r['difference_per_distance']-r['predicted_difference_per_distance']) for r in values]),
            derivative_expansion_error=summary_stats([abs(r['training_derivative']-r['predicted_training_derivative']) for r in values]),
            scaled_training_curvature_error=summary_stats([abs(r['scaled_training_curvature']-1) for r in values]),
            scaled_coefficient_derivative_error=summary_stats([abs(r['scaled_normal_coefficient_derivative']-1) for r in values]),
            scaled_heldout_derivative_error=summary_stats([abs(r['scaled_heldout_derivative']-r['predicted_scaled_heldout_derivative']) for r in values]),
        ))
    result = dict(state='passed', records=len(records), positive_endpoints=len(cases),
                  data_sha256=sha(args.workbook), script_sha256=sha(Path(__file__)),
                  inherited_path_sha256={name: replay['sha256'][name] for name in chosen},
                  original_raw_menu_rows_reconstructed=True, coefficient_fits_added=0,
                  entropy_identity_max_abs_error=max(r['entropy_value_identity_abs_error'] for r in records),
                  statistics=stats, records_sha256=sha(args.out / 'records.csv'))
    (args.out / 'summary.json').write_text(json.dumps(result, allow_nan=False, indent=2)+'\n')
    print(json.dumps(dict(state='passed', records=len(records), positive_endpoints=len(cases))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('workbook', 'paths', 'frontier', 'out'):
        parser.add_argument('--' + arg, type=Path, required=True)
    with threadpool_limits(limits=1):
        audit(parser.parse_args())
