"""Established human-only Firth logit on unchanged simulated choice samples.

Independent implementation of the Jeffreys-penalized likelihood formula.
No external package source is executed or copied. No new random cohorts.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.optimize import minimize, linprog
from scipy.special import expit
from scipy.stats import norm
from threadpoolctl import threadpool_limits


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, obj):
    p.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


class Firth:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def quantities(self, b):
        x, y = self.x, self.y
        u = x @ b
        mu = expit(u)
        w = mu * expit(-u)
        q, r = np.linalg.qr(np.sqrt(w)[:, None] * x, mode='reduced')
        diagonal = np.abs(np.diag(r))
        if np.any(diagonal <= np.finfo(float).tiny):
            raise np.linalg.LinAlgError('Singular information')
        hat = np.sum(q * q, axis=1)
        value = np.sum(np.logaddexp(0, u) - y * u) - np.sum(np.log(diagonal))
        gradient = x.T @ (mu - y - hat * (.5 - mu))
        return float(value), gradient, w, mu, hat, q, r

    def value(self, b):
        try:
            value, gradient, *_ = self.quantities(b)
            return value, gradient
        except np.linalg.LinAlgError:
            return np.inf, np.zeros_like(b)

    def hessian(self, b):
        _, _, w, mu, hat, q, _ = self.quantities(b)
        x = self.x
        c = 1 - 2 * mu
        projection = q @ q.T
        second = .5 * c[:, None] * c[None, :] * projection ** 2
        h = x.T @ ((w + hat * w - .5 * hat * c ** 2)[:, None] * x) + x.T @ second @ x
        return (h + h.T) / 2

    def covariance(self, b):
        *_, r = self.quantities(b)
        inverse_r = solve_triangular(r, np.eye(r.shape[0]))
        return inverse_r @ inverse_r.T


def logistic_fit(x, y, rho=0.):
    def fun(b):
        u = x @ b
        value = np.mean(np.logaddexp(0, u) - y * u) + rho * (b @ b) / 2
        grad = x.T @ (expit(u) - y) / len(y) + rho * b
        return value, grad
    b = minimize(fun, np.zeros(x.shape[1]), jac=True, method='BFGS',
                 options={'gtol': 1e-10, 'maxiter': 2000}).x
    for _ in range(12):
        value, grad = fun(b)
        if np.max(np.abs(grad)) < 1e-10:
            break
        mu = expit(x @ b)
        h = x.T @ ((mu * expit(-x @ b))[:, None] * x) / len(y) + rho * np.eye(len(b))
        try:
            direction = np.linalg.solve(h, grad)
        except np.linalg.LinAlgError:
            break
        step = 1.
        while step > 1e-10 and fun(b - step * direction)[0] > value + 1e-13:
            step /= 2
        if step <= 1e-10:
            break
        b -= step * direction
    return b, float(np.max(np.abs(fun(b)[1])))


def fit_firth(x, y):
    criterion = Firth(x, y)
    if np.linalg.matrix_rank(x) < x.shape[1]:
        return None, [], {'status': 'rank_deficient'}
    ridge, _ = logistic_fit(x, y, rho=1 / len(y))
    starts = [('zero', np.zeros(x.shape[1])), ('human_ridge', ridge), ('negative_ridge', -ridge)]
    attempts, candidates = [], []
    for name, start in starts:
        fit = minimize(criterion.value, start, jac=True, method='BFGS',
                       options={'gtol': 2e-9, 'maxiter': 2000})
        b = fit.x.copy()
        for _ in range(12):
            value, grad = criterion.value(b)
            if np.max(np.abs(grad)) < 2e-9:
                break
            try:
                h = criterion.hessian(b)
                if np.linalg.eigvalsh(h).min() <= 1e-10:
                    break
                direction = np.linalg.solve(h, grad)
            except np.linalg.LinAlgError:
                break
            step = 1.
            while step > 1e-10 and criterion.value(b - step * direction)[0] > value + 1e-12:
                step /= 2
            if step <= 1e-10:
                break
            b -= step * direction
        value, grad = criterion.value(b)
        score = float(np.max(np.abs(grad)))
        try:
            eig = float(np.linalg.eigvalsh(criterion.hessian(b)).min())
        except np.linalg.LinAlgError:
            eig = -np.inf
        accepted = bool(np.isfinite(value) and np.isfinite(b).all() and score < 1e-7 and eig > 1e-10)
        attempts.append({'start': name, 'optimizer_success': bool(fit.success),
                         'optimizer_status': int(fit.status), 'iterations': int(fit.nit),
                         'score_max': score, 'objective': float(value),
                         'penalized_hessian_min': eig, 'accepted': accepted})
        if accepted:
            candidates.append((value, b))
    if not candidates:
        return None, attempts, {'status': 'fit_unresolved'}
    candidates.sort(key=lambda z: z[0])
    value, b = candidates[0]
    spread = max(np.linalg.norm(z[1] - b) for z in candidates)
    return b, attempts, {'status': 'finite_adjusted_score_fit', 'objective': float(value),
        'score_max': float(np.max(np.abs(criterion.value(b)[1]))),
        'accepted_starts': len(candidates), 'accepted_start_coefficient_spread': float(spread),
        'penalized_hessian_min': float(np.linalg.eigvalsh(criterion.hessian(b)).min())}


def human_support(x, y):
    m, p = x.shape
    objective = np.r_[np.zeros(m), -1.]
    eq = np.column_stack([x.T / m, np.zeros(p)])
    ub = np.vstack([np.column_stack([-np.eye(m), np.ones(m)]),
                    np.column_stack([np.eye(m), np.ones(m)])])
    result = linprog(objective, A_ub=ub, b_ub=np.r_[np.zeros(m), np.ones(m)],
                     A_eq=eq, b_eq=x.T @ y / m,
                     bounds=[(0, 1)] * m + [(0, .5)], method='highs')
    assert result.success, result.message
    return float(result.x[-1])


def draws(inputs, plan):
    base = pd.read_csv(inputs / 'attribute_support.csv', float_precision='round_trip').to_numpy()
    truth0 = np.asarray(json.loads((inputs / 'truth.json').read_text())['coefficient'])
    for setting in plan['conditions']:
        truth = setting['truth_scale'] * truth0
        rng = np.random.default_rng(setting['seed'])
        for replication in range(plan['replications_per_condition']):
            primary = base[rng.integers(0, len(base), 150)]
            auxiliary = base[rng.integers(0, len(base), 600)]
            yp = (rng.random(150) < expit(primary @ truth)).astype(float)
            _ = rng.random(600) < expit(auxiliary @ truth)
            _ = rng.random(150)
            _ = rng.random(600)
            yield setting['name'], replication, base, truth, primary[:setting['labels']], yp[:setting['labels']]


def summaries(rows, baselines, out):
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'fit_records.csv', index=False)
    result = []
    for condition, part in frame.groupby('condition'):
        row = {'condition': condition, 'replications': len(part),
               'estimates': int(part.estimate_available.sum()), 'intervals': int(part.interval_available.sum())}
        for metric in ['MSE', 'excess_logloss', 'marginal_coverage', 'mean_interval_width']:
            values = part[metric].dropna()
            row[metric] = float(values.mean())
            row[metric + '_MCSE'] = float(values.std(ddof=1) / np.sqrt(len(values)))
        row['median_MSE'] = float(part.MSE.median())
        row['q90_MSE'] = float(part.MSE.quantile(.9))
        result.append(row)
    pd.DataFrame(result).to_csv(out / 'summary.csv', index=False)
    paired = []
    for (condition, method), ref in baselines.groupby(['condition', 'method']):
        joined = frame[frame.condition == condition].merge(ref, on='replication', suffixes=('_Firth', '_reference'))
        for metric in ['MSE', 'excess_logloss']:
            values = joined[[metric + '_Firth', metric + '_reference']].dropna()
            d = values.iloc[:, 0] - values.iloc[:, 1]
            paired.append({'condition': condition, 'comparator': method, 'metric': metric,
                           'shared_available_fits': len(values), 'mean_Firth': values.iloc[:, 0].mean(),
                           'mean_comparator': values.iloc[:, 1].mean(), 'mean_difference': d.mean(),
                           'paired_MCSE': d.std(ddof=1) / np.sqrt(len(d))})
    pd.DataFrame(paired).to_csv(out / 'paired.csv', index=False)


def run(inputs, plan_path, out):
    plan = json.loads(plan_path.read_text())
    baseline = pd.read_csv(inputs / 'baseline_records.csv')
    assert len(baseline) == 9000
    rows, attempts, coefficients, reconstruction = [], [], [], []
    refhash = {p.name: digest(p) for p in inputs.iterdir() if p.is_file()}
    critical = norm.ppf(.975)
    for condition, rep, base, truth, x, y in draws(inputs, plan):
        ref = baseline[(baseline.condition == condition) & (baseline.method == 'human_MLE')].set_index('replication').loc[rep]
        slack = human_support(x, y)
        available = slack > 1e-8
        assert available == bool(ref.estimate_available), (condition, rep, slack, ref.estimate_available)
        relative = 0.
        if available:
            human, grad = logistic_fit(x, y)
            relative = abs(float(np.mean((human - truth) ** 2)) - ref.MSE) / (1 + ref.MSE)
            assert relative < 1e-5, (condition, rep, relative, grad)
        reconstruction.append({'condition': condition, 'replication': rep, 'human_available': available,
                               'original_available': bool(ref.estimate_available), 'human_support_slack': slack,
                               'human_MSE_relative_difference': relative,
                               'simulated_design_sha256': hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()})
        beta, tries, check = fit_firth(x, y)
        for attempt in tries:
            attempts.append({'condition': condition, 'replication': rep, **attempt})
        row = {'condition': condition, 'replication': rep, 'method': 'human_Firth',
               'label_rows': len(y), 'auxiliary_rows_used': 0, 'human_available': available,
               'estimate_available': beta is not None, 'interval_available': False,
               'MSE': np.nan, 'excess_logloss': np.nan, 'marginal_coverage': np.nan,
               'mean_interval_width': np.nan, **check}
        if beta is not None:
            error = beta - truth
            u, u0 = base @ beta, base @ truth
            kl = float(np.mean(np.logaddexp(0, u) - np.logaddexp(0, u0) - expit(u0) * (u - u0)))
            covariance = Firth(x, y).covariance(beta)
            se = np.sqrt(np.diag(covariance))
            coverage = float(np.mean(np.abs(error) <= critical * se))
            assert np.isfinite(se).all() and np.min(se) > 0 and kl > -1e-12
            row.update(MSE=float(np.mean(error ** 2)), excess_logloss=kl,
                       interval_available=True, marginal_coverage=coverage,
                       mean_interval_width=float(np.mean(2 * critical * se)))
            coefficients.append({'condition': condition, 'replication': rep,
                                 **{f'b{j}': float(b) for j, b in enumerate(beta)}})
        rows.append(row)
        if (rep + 1) % 100 == 0:
            summaries(rows, baseline, out)
            pd.DataFrame(attempts).to_csv(out / 'optimization_attempts.csv', index=False)
            pd.DataFrame(coefficients).to_csv(out / 'coefficients.csv', index=False)
            pd.DataFrame(reconstruction).to_csv(out / 'original_sample_checks.csv', index=False)
            save(out / 'run.json', {'completed_original_draws': len(rows), 'requested_original_draws': 1000,
                 'completed_optimizer_starts': len(attempts), 'unavailable_fits': sum(not r['estimate_available'] for r in rows),
                 'source_sha256': digest(Path(__file__)), 'plan_sha256': digest(plan_path),
                 'input_sha256': refhash, 'new_independent_cohorts': 0,
                 'external_R_source_copied_or_executed': False, 'global_maximum_certified': False,
                 'profile_likelihood_intervals_computed': False, 'interval_type': 'model_information_Wald',
                 'Monte_Carlo_unit': 'one original repetition, not one coefficient', 'submission_threshold_proven': False})
            print(condition, rep + 1, 'of 500', flush=True)
    assert all(digest(inputs / name) == h for name, h in refhash.items())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        run(args.inputs, args.plan, args.out)
