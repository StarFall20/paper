"""Independent objective, invariance and retained-result checks for Firth replay."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm
from threadpoolctl import threadpool_limits
from firth_choice_comparator import Firth, fit_firth, logistic_fit, draws, save


def direct(x, y, b):
    u = x @ b
    information = x.T @ ((expit(u) * expit(-u))[:, None] * x)
    sign, ld = np.linalg.slogdet(information)
    return np.sum(np.logaddexp(0, u) - y * u) - .5 * ld if sign > 0 else np.inf


def finite_gradient(fun, b, step=2e-5):
    eye = np.eye(len(b))
    return np.array([(fun(b + step * e) - fun(b - step * e)) / (2 * step) for e in eye])


def checks(out):
    rng = np.random.default_rng(2026111307)
    gradient_error = hessian_error = covariance_error = intercept_error = invariance_error = complement_error = 0.
    for n, p in [(7, 2), (20, 11), (150, 11)]:
        for _ in range(4):
            x = rng.normal(size=(n, p)); y = rng.integers(0, 2, n).astype(float)
            b = rng.normal(size=p) / 4
            model = Firth(x, y); val, grad = model.value(b)
            assert abs(val - direct(x, y, b)) < 1e-10
            fd = finite_gradient(lambda z: direct(x, y, z), b)
            gradient_error = max(gradient_error, float(np.max(np.abs(fd - grad))))
            step = 2e-5
            fh = np.column_stack([(model.value(b + step * e)[1] - model.value(b - step * e)[1]) / (2 * step) for e in np.eye(p)])
            hessian_error = max(hessian_error, float(np.max(np.abs(fh - model.hessian(b)))))
            w = expit(x @ b) * expit(-x @ b)
            covariance_error = max(covariance_error, float(np.max(np.abs(model.covariance(b) - np.linalg.inv(x.T @ (w[:, None] * x))))))
    for n in [1, 8, 20]:
        for successes in sorted(set([0, 1, n // 2, n])):
            y = np.r_[np.ones(successes), np.zeros(n - successes)]
            x = np.ones((n, 1))
            b, _, _ = fit_firth(x, y)
            assert b is not None
            expected = logit((successes + .5) / (n + 1))
            intercept_error = max(intercept_error, float(abs(b[0] - expected)))
    for _ in range(6):
        x = rng.normal(size=(20, 4)); y = rng.integers(0, 2, 20).astype(float)
        b, _, _ = fit_firth(x, y); bc, _, _ = fit_firth(x, 1 - y)
        assert b is not None and bc is not None
        complement_error = max(complement_error, float(np.max(np.abs(bc + b))))
        t = np.eye(4) + rng.normal(size=(4, 4)) / 5
        bt, _, _ = fit_firth(x @ t, y)
        assert bt is not None
        invariance_error = max(invariance_error, float(np.max(np.abs(t @ bt - b))))
    assert gradient_error < 1e-6 and hessian_error < 1e-6 and covariance_error < 1e-8
    assert intercept_error < 1e-6 and invariance_error < 1e-6 and complement_error < 1e-6
    record = {'gradient_finite_difference_max_error': gradient_error, 'hessian_finite_difference_max_error': hessian_error,
              'information_covariance_max_error': covariance_error, 'analytic_intercept_max_error': intercept_error,
              'linear_reparameterization_max_error': invariance_error, 'outcome_complement_max_error': complement_error}
    save(out / 'numeric_checks.json', record)
    return record


def verify(inputs, plan_path, results, out):
    plan = json.loads(plan_path.read_text())
    records = pd.read_csv(results / 'fit_records.csv')
    co = pd.read_csv(results / 'coefficients.csv').set_index(['condition', 'replication'])
    assert len(records) == 1000
    selected = set(range(0, 500, 20))
    multiple_mode_cases = {(r.condition, int(r.replication)) for r in records.itertuples()
                           if r.accepted_start_coefficient_spread > 1e-4}
    maximum_gradient = max_cov = max_metric = max_direct_coefficient = max_direct_objective = 0.
    independent = []
    recon = pd.read_csv(results / 'original_sample_checks.csv')
    assert len(recon) == 1000 and (recon.human_available == recon.original_available).all()
    assert recon.human_MSE_relative_difference.max() < 1e-5
    for condition, rep, base, truth, x, y in draws(inputs, plan):
        row = records[(records.condition == condition) & (records.replication == rep)].iloc[0]
        if not row.estimate_available:
            continue
        b = co.loc[(condition, rep)].to_numpy(dtype=float)
        score = finite_gradient(lambda z: direct(x, y, z), b)
        maximum_gradient = max(maximum_gradient, float(np.max(np.abs(score))))
        u = x @ b
        information = x.T @ ((expit(u) * expit(-u))[:, None] * x)
        inv = np.linalg.inv(information)
        max_cov = max(max_cov, float(np.max(np.abs(inv - Firth(x, y).covariance(b)))))
        error = b - truth
        expected_coverage = float(np.mean(np.abs(error) <= norm.ppf(.975) * np.sqrt(np.diag(inv))))
        expected_mse = float(np.mean(error ** 2))
        up = base @ b; ut = base @ truth
        expected_kl = float(np.mean(np.logaddexp(0, up) - np.logaddexp(0, ut) - expit(ut) * (up - ut)))
        max_metric = max(max_metric, abs(expected_coverage - row.marginal_coverage), abs(expected_mse - row.MSE), abs(expected_kl - row.excess_logloss))
        if rep in selected or (condition, rep) in multiple_mode_cases:
            fun = lambda z: direct(x, y, z)
            is_multiple = (condition, rep) in multiple_mode_cases
            starts = [('zero', np.zeros_like(b))]
            if is_multiple:
                ridge, _ = logistic_fit(x, y, rho=1 / len(y))
                starts += [('human_ridge', ridge), ('negative_ridge', -ridge)]
            fits = []
            for name, start in starts:
                fit = minimize(fun, start, method='BFGS',
                               jac=lambda z: finite_gradient(fun, z), options={'gtol': 2e-7, 'maxiter': 1800})
                independent.append({'condition': condition, 'replication': rep, 'start': name,
                                    'optimizer_success': bool(fit.success), 'multiple_mode_case': is_multiple,
                                    'objective': float(fun(fit.x)),
                                    'max_coefficient_difference': float(np.max(np.abs(fit.x - b))),
                                    'objective_difference': abs(float(fun(fit.x) - fun(b))),
                                    'independent_score_max': float(np.max(np.abs(finite_gradient(fun, fit.x))))})
                fits.append(fit)
            best = min(fits, key=lambda z: fun(z.x))
            difference = float(np.max(np.abs(best.x - b)))
            objective_difference = abs(float(fun(best.x) - fun(b)))
            max_direct_coefficient = max(max_direct_coefficient, difference)
            max_direct_objective = max(max_direct_objective, objective_difference)
    pd.DataFrame(independent).to_csv(out / 'direct_objective_crosschecks.csv', index=False)
    assert maximum_gradient < 2e-6 and max_cov < 1e-8 and max_metric < 1e-10
    assert len(independent) == 50 + sum(3 if rep not in selected else 2 for _, rep in multiple_mode_cases)
    assert max_direct_coefficient < 1e-4 and max_direct_objective < 1e-8
    result = {'original_draws_verified': 1000, 'new_fit_records': len(records),
              'available_new_estimates': int(records.estimate_available.sum()),
              'all_retained_root_finite_difference_max_score': maximum_gradient,
              'all_retained_information_covariance_max_error': max_cov,
              'all_retained_metric_max_error': max_metric, 'independent_direct_objective_fits': len(independent),
              'independent_direct_objective_designs': 50 + sum(rep not in selected for _, rep in multiple_mode_cases),
              'independent_fit_max_coefficient_difference': max_direct_coefficient,
              'independent_fit_max_objective_difference': max_direct_objective,
              'multiple_mode_cases': len(multiple_mode_cases),
              'global_optimality_proven': False, 'profile_intervals_verified': False}
    save(out / 'checks.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--checks-only', action='store_true')
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        print(json.dumps(checks(args.out)), flush=True)
        if not args.checks_only:
            verify(args.inputs, args.plan, args.results, args.out)
