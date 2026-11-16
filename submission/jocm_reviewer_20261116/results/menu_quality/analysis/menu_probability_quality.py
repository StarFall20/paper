"""Parameter and forecast checks on the existing hard-menu Gaussian cohorts.

Uses the same compensated objective and existing fixed oracle power. All
original cells and draws are retained. Human ridge and hard-label pooling are
established comparators. This script creates no human or model-generated answers.
"""
from pathlib import Path
import argparse
import ast
import gzip
import hashlib
import json
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtri, softmax
from scipy.stats import qmc
from threadpoolctl import threadpool_limits
from menu_corrected_logit import MenuLoss, menu_features, support


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_closure():
    base = Path(__file__).resolve().parent
    found = {}
    def visit(path):
        if path.name in found:
            return
        found[path.name] = sha(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([node.module.split('.')[0]] if isinstance(node, ast.ImportFrom) and node.module
                     else [x.name.split('.')[0] for x in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                target = base / (name + '.py')
                if target.is_file():
                    visit(target)
    visit(Path(__file__).resolve())
    return found


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class StableMenuLoss:
    """Cross entropy plus an explicit affine correction, using stable scores."""
    def __init__(self, features, y, g, m, power, penalty=0., pooling=False):
        self.f = np.asarray(features, float)
        self.n, self.J, self.p = self.f.shape
        self.flat = self.f.reshape(-1, self.p)
        self.penalty = penalty
        self.target = np.vstack([np.eye(self.J)[y[:m]], g[m:]])
        if pooling:
            self.w = np.full(self.n, 1 / self.n)
            self.c = np.zeros(self.p)
        else:
            self.w = np.r_[np.full(m, (1 - power) / m),
                           np.full(self.n - m, power / (self.n - m))]
            self.c = power * np.einsum('nj,njp->p',
                                      g[:m] - self.target[:m], self.f[:m]) / m

    def value(self, beta):
        u = self.f @ beta
        logp = u - logsumexp(u, axis=1, keepdims=True)
        probabilities = np.exp(logp)
        ce = -np.sum(self.target * logp, axis=1)
        value = self.w @ ce + self.c @ beta + self.penalty * (beta @ beta) / 2
        gradient = ((self.w[:, None] * (probabilities - self.target)).ravel()
                    @ self.flat + self.c + self.penalty * beta)
        return float(value), gradient

    def hessian(self, beta):
        probabilities = softmax(self.f @ beta, axis=1)
        mean = np.einsum('nj,njp->np', probabilities, self.f)
        centered = self.f - mean[:, None, :]
        weighted = (centered * np.sqrt(self.w[:, None] * probabilities)[:, :, None]).reshape(-1, self.p)
        return weighted.T @ weighted + self.penalty * np.eye(self.p)


def fit(loss, initial, settings):
    beta = initial.copy()
    flags = []
    iterations = 0

    def newton(beta, limit):
        nonlocal iterations
        for _ in range(limit):
            iterations += 1
            value, gradient = loss.value(beta)
            hessian = loss.hessian(beta)
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                flags.append('singular_Newton_system')
                break
            step_norm = np.linalg.norm(step)
            if (np.linalg.norm(gradient) <= settings['gradient_l2_tolerance'] and
                    step_norm <= settings['relative_Newton_step_tolerance'] * (1 + np.linalg.norm(beta))):
                return beta
            if not np.isfinite(step).all() or gradient @ step <= 0:
                flags.append('invalid_Newton_direction')
                break
            rate = 1.
            while rate >= 2 ** -35:
                trial = beta - rate * step
                next_value = loss.value(trial)[0]
                if (np.isfinite(next_value) and
                        next_value <= value - 1e-4 * rate * (gradient @ step) + 1e-13 * (1 + abs(value))):
                    beta = trial
                    break
                rate /= 2
            else:
                flags.append('Newton_line_search_stopped')
                break
        return beta

    beta = newton(beta, settings['maximum_Newton_iterations'])

    def metrics(beta):
        value, gradient = loss.value(beta)
        hessian = loss.hessian(beta)
        eigenvalues = np.linalg.eigvalsh(hessian)
        try:
            step = np.linalg.solve(hessian, gradient)
            step_norm = float(np.linalg.norm(step))
        except np.linalg.LinAlgError:
            step_norm = None
        okay = bool(np.isfinite(beta).all() and np.linalg.norm(gradient) <= settings['gradient_l2_tolerance']
                    and eigenvalues[0] > 0 and step_norm is not None
                    and step_norm <= settings['relative_Newton_step_tolerance'] * (1 + np.linalg.norm(beta)))
        return okay, dict(objective=value, gradient_l2=float(np.linalg.norm(gradient)),
                          newton_step_l2=step_norm, hessian_min=float(eigenvalues[0]))

    okay, diagnostics = metrics(beta)
    if not okay:
        flags.append('BFGS_rescue')
        result = minimize(loss.value, beta, jac=True, method='BFGS',
                          options={'gtol': 1e-11, 'maxiter': settings['BFGS_rescue_max_iterations']})
        beta = newton(result.x.copy(), settings['rescue_Newton_iterations'])
        okay, diagnostics = metrics(beta)
    diagnostics.update(fit_status='numerical_finite_root' if okay else 'numerical_unresolved',
                       iterations=iterations, solver_flags=';'.join(flags))
    return beta, diagnostics


class GaussianMenuRisk:
    """Exact distributional rotation followed by a randomized normal integral."""
    def __init__(self, J, gamma, bits, scrambles, seed):
        self.J = J
        self.gamma = gamma
        nodes = []
        for k in range(scrambles):
            uniforms = qmc.Sobol(2 * J, scramble=True, seed=seed + k).random_base2(bits)
            nodes.append(ndtri(uniforms))
        nodes = np.asarray(nodes)
        self.signal = nodes[..., :J]
        self.noise = nodes[..., J:]
        self.truth = softmax(gamma * self.signal, axis=-1)
        self.truth_loss = float(np.mean(-np.sum(self.truth * np.log(self.truth), axis=-1)))
        self.truth_brier = float(np.mean(1 - np.sum(self.truth ** 2, axis=-1)))
        self.nodes_per_fit = scrambles * 2 ** bits

    def evaluate(self, beta):
        J = self.J
        constants = np.r_[beta[:J-1], 0.]
        noise_norm = np.linalg.norm(beta[J:])
        u = constants + beta[J-1] * self.signal + noise_norm * self.noise
        logp = u - logsumexp(u, axis=-1, keepdims=True)
        probabilities = np.exp(logp)
        losses = -np.sum(self.truth * logp, axis=-1)
        brier = 1 + np.sum(probabilities ** 2 - 2 * self.truth * probabilities, axis=-1)
        selected = np.argmax(u, axis=-1)
        error = 1 - np.take_along_axis(self.truth, selected[..., None], axis=-1)[..., 0]
        scramble_means = losses.mean(axis=1)
        return dict(human_log_loss=float(losses.mean()),
                    excess_human_log_loss=float(losses.mean() - self.truth_loss),
                    human_Brier_loss=float(brier.mean()),
                    human_classification_error=float(error.mean()),
                    integration_log_loss_scramble_SE=float(scramble_means.std(ddof=1) / np.sqrt(len(scramble_means))))


def summaries(rows, out):
    frame = pd.DataFrame([{k: v for k, v in r.items() if k != 'beta'} for r in rows])
    frame.to_csv(out / 'quality_records.csv', index=False)
    results = []
    measures = ['squared_coefficient_error', 'coefficient_norm', 'human_log_loss',
                'excess_human_log_loss', 'human_Brier_loss', 'human_classification_error']
    for key, group in frame.groupby(['label_menus', 'orthogonal_slopes', 'method']):
        available = group[group.fit_status.eq('numerical_finite_root')]
        record = dict(zip(['label_menus', 'orthogonal_slopes', 'method'], key))
        record.update(requested=len(group), interior_support=int(group.support_status.eq('interior').sum()),
                      penalized_coercive=int(group.support_status.eq('penalized_coercive').sum()),
                      numerical_finite=len(available), numerical_unresolved=int(group.fit_status.eq('numerical_unresolved').sum()))
        for measure in measures:
            values = available[measure].to_numpy()
            record['mean_' + measure] = float(np.mean(values)) if len(values) else None
            record['median_' + measure] = float(np.median(values)) if len(values) else None
            record['q95_' + measure] = float(np.quantile(values, .95)) if len(values) else None
            record['maximum_' + measure] = float(np.max(values)) if len(values) else None
        results.append(record)
    pd.DataFrame(results).to_csv(out / 'quality_summary.csv', index=False)
    pairs, fallback = [], []
    for key, group in frame.groupby(['label_menus', 'orthogonal_slopes']):
        ridge = group[group.method.eq('human_ridge')].set_index('trial')
        for method, part in group.groupby('method'):
            part = part.set_index('trial')
            common = sorted(set(part.index) & set(ridge.index))
            good = [i for i in common if part.loc[i, 'fit_status'] == ridge.loc[i, 'fit_status'] == 'numerical_finite_root']
            record = dict(label_menus=key[0], orthogonal_slopes=key[1], method=method,
                          paired_allocations=len(common), shared_numerical_finite=len(good))
            for measure in measures:
                diff = np.asarray([part.loc[i, measure] - ridge.loc[i, measure] for i in good])
                record['mean_difference_' + measure] = float(diff.mean()) if len(diff) else None
                record['paired_MCSE_' + measure] = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else None
            pairs.append(record)
            if method in ['ordinary', 'oracle_null_slope_variance', 'unit']:
                good_ridge = [i for i in common if ridge.loc[i, 'fit_status'] == 'numerical_finite_root']
                used = [i for i in good_ridge if part.loc[i, 'fit_status'] == 'numerical_finite_root']
                changes = [float(part.loc[i, 'human_log_loss'] - ridge.loc[i, 'human_log_loss']) if i in used else 0. for i in good_ridge]
                risk = [float(part.loc[i, 'human_log_loss']) if i in used else float(ridge.loc[i, 'human_log_loss']) for i in good_ridge]
                fallback.append(dict(label_menus=key[0], orthogonal_slopes=key[1], method=method,
                                     evaluated=len(good_ridge), direct_method_used=len(used),
                                     ridge_fallbacks=len(good_ridge)-len(used), mean_human_log_loss=float(np.mean(risk)),
                                     mean_difference_from_ridge=float(np.mean(changes)),
                                     paired_MCSE=float(np.std(changes, ddof=1)/np.sqrt(len(changes)))))
    pd.DataFrame(pairs).to_csv(out / 'paired_with_ridge.csv', index=False)
    pd.DataFrame(fallback).to_csv(out / 'ridge_fallback.csv', index=False)


def run(args):
    repo = Path(__file__).resolve().parents[1]
    plan = json.loads(args.plan.read_text())
    source_plan = repo / plan['source_plan']
    source_trials = repo / plan['source_trials']
    assert sha(source_plan) == plan['source_plan_sha256']
    assert sha(source_trials) == plan['source_trials_sha256']
    assert sha(repo / plan['source_generator']) == plan['source_generator_sha256']
    original = json.loads(source_plan.read_text())
    old = pd.read_csv(source_trials)
    lookup = {(int(r.label_menus), int(r.trial), int(r.orthogonal_slopes), r.method): r.status
              for r in old.itertuples()}
    assert len(lookup) == 9000
    source_hash = sha(Path(__file__))
    dependencies = source_closure()
    settings = plan['numerics']
    rows, completed = [], set()
    checkpoint = args.out / 'checkpoint.json.gz'
    if args.resume:
        previous = json.loads(gzip.decompress(checkpoint.read_bytes()))
        assert previous['source_sha256'] == source_hash and previous['plan_sha256'] == sha(args.plan)
        assert previous['source_hashes'] == dependencies
        rows = previous['records']
        completed = {tuple(x) for x in previous['completed_cohorts']}
        assert len(rows) == 15 * len(completed)
    else:
        args.out.mkdir(parents=True, exist_ok=False)
    integration = plan['integration']
    risk = GaussianMenuRisk(original['alternatives'], original['gamma'], integration['bits'],
                            integration['scrambles'], integration['seed'])
    rng = np.random.default_rng(original['seed'])
    counts = {'original_support_matches': 0, 'pool_support_programs': 0}
    if args.resume:
        counts.update(previous['counts'])
    start = time.monotonic()

    def store(state):
        data = dict(state=state, source_sha256=source_hash, plan_sha256=sha(args.plan),
                    source_hashes=dependencies, completed_cohorts=sorted(completed), records=rows, counts=counts)
        temporary = checkpoint.with_suffix('.tmp')
        temporary.write_bytes(gzip.compress(json.dumps(data, allow_nan=False, separators=(',', ':')).encode(), mtime=0))
        temporary.replace(checkpoint)
        save(args.out / 'quality_status.json', {k: v for k, v in data.items() if k not in ['records', 'completed_cohorts']} | dict(
            completed_cohorts=len(completed), planned_cohorts=1000, retained_records=len(rows),
            planned_records=plan['planned_records'], elapsed_this_invocation_seconds=time.monotonic()-start,
            integration_nodes_per_fit=risk.nodes_per_fit, new_model_cohorts=0, new_human_or_LLM_answers=0))

    store('running')
    J = original['alternatives']
    for m in original['label_menus']:
        N = int(original['tau'] * m)
        n = m + N
        dimensions = sorted({max(1, round(m*k)) for k in original['dimension_ratios']})
        for trial in range(original['replications_per_budget']):
            S = rng.normal(size=(n, J, 1))
            W = rng.normal(size=(n, J, max(dimensions)))
            mu = softmax(original['gamma'] * S[..., 0], axis=1)
            y = np.sum(rng.random(n)[:, None] > np.cumsum(mu, axis=1), axis=1)
            z = (y + (rng.random(n) < original['epsilon']) * rng.integers(1, J, size=n)) % J
            g = np.eye(J)[z]
            key = (m, trial)
            if key in completed or (args.pilot and trial >= args.pilot):
                continue
            cohort = []
            for q in dimensions:
                features = menu_features(np.concatenate([S, W[..., :q]], axis=2))
                truth = np.zeros(features.shape[-1])
                truth[J-1] = original['gamma']
                ridge_loss = StableMenuLoss(features, y, g, m, 0., penalty=1/m)
                ridge_beta, ridge_metrics = fit(ridge_loss, np.zeros(len(truth)), settings)
                for method in plan['methods']:
                    power = original['powers'].get(method, 0.)
                    penalty = 1/m if method == 'human_ridge' else 0.
                    if method == 'human_ridge':
                        state = 'penalized_coercive'
                        beta, diagnostics = ridge_beta.copy(), ridge_metrics.copy()
                    else:
                        if method == 'hard_pool':
                            pool_g = np.vstack([np.eye(J)[y[:m]], g[m:]])
                            actual = MenuLoss(features, y, pool_g, m, N/n)
                            state = support(actual)['status']
                            counts['pool_support_programs'] += 1
                        else:
                            actual = MenuLoss(features, y, g, m, power)
                            state = support(actual)['status']
                            expected = lookup[(m, trial, q, method)]
                            if state != expected:
                                save(args.out / 'support_mismatch.json', dict(m=m, trial=trial, q=q, method=method, actual=state, expected=expected))
                                raise AssertionError('Original support reconstruction mismatch; evidence saved.')
                            counts['original_support_matches'] += 1
                        loss = StableMenuLoss(features, y, g, m, power, pooling=method == 'hard_pool')
                        if state == 'interior':
                            beta, diagnostics = fit(loss, ridge_beta, settings)
                        else:
                            beta = None
                            diagnostics = dict(fit_status='not_fitted_noninterior', iterations=0)
                    record = dict(label_menus=m, auxiliary_menus=N, trial=trial, orthogonal_slopes=q,
                                  dimension_ratio=q/m, method=method,
                                  power=None if method == 'hard_pool' else power, penalty=penalty,
                                  support_status=state, **diagnostics)
                    record['beta'] = beta.tolist() if beta is not None else None
                    if beta is not None:
                        record['coefficient_norm'] = float(np.linalg.norm(beta))
                    if diagnostics['fit_status'] == 'numerical_finite_root':
                        record['squared_coefficient_error'] = float(np.sum((beta-truth)**2))
                        record.update(risk.evaluate(beta))
                    cohort.append(record)
            assert len(cohort) == 15
            rows.extend(cohort)
            completed.add(key)
            if len(completed) % 10 == 0 or args.pilot:
                store('running')
                print(json.dumps(dict(completed_cohorts=len(completed), retained_records=len(rows),
                                      unresolved=sum(r['fit_status']=='numerical_unresolved' for r in rows),
                                      elapsed_seconds=round(time.monotonic()-start, 2))), flush=True)
    summaries(rows, args.out)
    if not args.pilot:
        assert len(completed) == 1000 and len(rows) == 15000
        assert counts == {'original_support_matches': 9000, 'pool_support_programs': 3000}
    assert source_closure() == dependencies
    store('pilot_complete' if args.pilot else 'complete')
    save(args.out / 'risk_definition.json', dict(
        population='Independent new Gaussian alternative profiles under the original known human model.',
        exact_rotation='Given fitted beta, noise utilities have distribution norm(beta_noise) times J independent standard normals.',
        human_truth_log_loss_integral=risk.truth_loss, human_truth_Brier_integral=risk.truth_brier,
        risk_is_conditional_on_the_fitted_coefficient=True, new_test_choices_generated=False,
        coefficient_error_basis='Original J-1 reference-category constants, common signal slope, common noise slopes.',
        interval_coverage_evaluated=False, claimed_new_estimator=False))
    print((args.out / 'quality_status.json').read_text(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--pilot', type=int, default=0, help='Software check on first k existing cohorts per budget; not primary evidence.')
    arguments = parser.parse_args()
    with threadpool_limits(limits=1):
        run(arguments)
