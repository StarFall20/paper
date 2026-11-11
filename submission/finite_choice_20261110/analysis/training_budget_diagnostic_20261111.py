"""Independent-training label-cost diagnostic; all outcomes are simulated.

Import the manuscript's canonical objective/support/finite-root solver without
changing it. The experiment is specified in the adjacent JSON plan.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from power_tuned_choice_audit import WeightedLogit, objective, solve


def ridge(x, y, rho=0.1):
    """Strongly convex penalized mean loss, with a checked stationary point."""
    n, p = x.shape
    def value(beta):
        u = x @ beta
        return (float(np.mean(np.logaddexp(0, u) - y*u) + rho/2*(beta @ beta)),
                x.T @ (expit(u)-y)/n + rho*beta)
    fit = minimize(value, np.zeros(p), jac=True, method='L-BFGS-B',
                   options={'gtol': 1e-10, 'ftol': 1e-15, 'maxiter': 1000})
    beta = fit.x.copy()
    for _ in range(5):
        val, grad = value(beta)
        if np.max(np.abs(grad)) < 1e-10:
            break
        mu = expit(x @ beta)
        hess = x.T @ ((mu*(1-mu))[:, None]*x)/n + rho*np.eye(p)
        direction = np.linalg.solve(hess, grad)
        step = 1.
        while step > 1e-8 and value(beta-step*direction)[0] > val+1e-13:
            step /= 2
        beta -= step*direction
    residual = float(np.max(np.abs(value(beta)[1])))
    if residual > 1e-8 or not np.isfinite(beta).all():
        return None, {'status': 'ridge_fit_unresolved', 'score_max': residual}
    return beta, {'status': 'finite_root', 'score_max': residual}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def summaries(rows, out):
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'training_budget_trials.csv', index=False)
    group = ['total_labels', 'slope_dimension', 'training_labels', 'method']
    counts = frame.groupby(group+['status'], as_index=False).size()
    counts.to_csv(out/'training_budget_status.csv', index=False)
    summary = frame.groupby(group, as_index=False).agg(
        draws=('trial', 'size'), verified_fits=('coefficient_error', 'count'),
        coefficient_error=('coefficient_error', 'mean'),
        excess_log_loss=('excess_log_loss', 'mean'),
        median_excess_log_loss=('excess_log_loss', 'median'),
        coefficient_norm=('coefficient_norm', 'mean'),
        maximum_score_residual=('score_max', 'max'))
    summary['fit_frequency'] = summary.verified_fits / summary.draws
    summary.to_csv(out/'training_budget_summary.csv', index=False)
    paired = []
    for key, part in frame.groupby(group[:-1]):
        risk = part.pivot(index='trial', columns='method', values='excess_log_loss')
        for baseline in ['all_labels_mle', 'all_labels_ridge_0.1']:
            for method in risk.columns:
                if method == baseline:
                    continue
                diff = (risk[method]-risk[baseline]).dropna()
                paired.append(dict(zip(group[:-1], key), method=method, baseline=baseline,
                                   planned_draws=len(risk), shared_verified_fits=len(diff),
                                   excess_loss_difference=diff.mean(),
                                   Monte_Carlo_SE=diff.std(ddof=1)/np.sqrt(len(diff)) if len(diff)>1 else np.nan))
    pd.DataFrame(paired).to_csv(out/'training_budget_paired.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plan_path = Path(__file__).with_name('training_budget_plan_20261111.json')
    plan = json.loads(plan_path.read_text())
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    specs = [(b, round(b*r), round(b*f)) for b in plan['total_human_labels']
             for r in plan['slope_dimensions_per_total_label']
             for f in plan['independent_training_label_fractions']]
    rows = []
    independent_labels = 0
    for condition, (budget, d, train) in enumerate(specs):
        m = budget-train
        n_aux = plan['auxiliary_rows_per_correction_label']*m
        truth = np.r_[0., 1.5, np.zeros(d-1)]
        for trial in range(plan['repetitions_per_condition']):
            rng = np.random.default_rng(np.random.SeedSequence([plan['seed'], condition, trial]))
            features = rng.normal(size=(budget+n_aux, d))
            x = np.c_[np.ones(len(features)), features]
            # Only the budgeted human-label union receives simulated labels.
            y = (rng.random(budget) < expit(x[:budget] @ truth)).astype(float)
            independent_labels += budget
            xt, yt = x[:train], y[:train]
            audit_x = x[train:]
            audit_y = np.r_[y[train:], np.zeros(n_aux)]  # Auxiliary dummy entries never enter objective().
            train_beta, train_check = ridge(xt, yt)
            if train_beta is None:
                raise RuntimeError(('training ridge unresolved', condition, trial, train_check))
            g = expit(audit_x @ train_beta)
            if np.any(g <= 0) or np.any(g >= 1):
                raise RuntimeError('Frozen trained prediction lost strict interior precision')
            all_ridge, all_ridge_check = ridge(x[:budget], y)
            if all_ridge is None:
                raise RuntimeError(('all-label ridge unresolved', condition, trial, all_ridge_check))
            ordinary = WeightedLogit(x[:budget], np.full(budget, 1/budget), x[:budget].T @ y/budget)
            all_mle, all_mle_check = solve(ordinary, all_ridge)
            cases = [('training_ridge_only', train_beta, train_check),
                     ('all_labels_ridge_0.1', all_ridge, all_ridge_check),
                     ('all_labels_mle', all_mle, all_mle_check)]
            for power in [.1, .5, 1.]:
                beta, check = solve(objective(audit_x, audit_y, g, m, power), train_beta)
                cases.append((f'corrected_power_{power:.1f}', beta, check))
            test = np.c_[np.ones(4096), rng.normal(size=(4096, d))]
            true_eta = test @ truth
            true_mu = expit(true_eta)
            true_loss = np.mean(np.logaddexp(0, true_eta)-true_mu*true_eta)
            for method, beta, check in cases:
                row = dict(condition=condition, trial=trial, total_labels=budget,
                           slope_dimension=d, training_labels=train, correction_labels=m,
                           auxiliary_rows=n_aux, method=method, status=check['status'],
                           score_max=check.get('score_max'),
                           support_margin=check.get('interior_margin'),
                           separation_gap=check.get('separation_gap'),
                           coefficient_error=np.nan, excess_log_loss=np.nan, coefficient_norm=np.nan)
                if beta is not None:
                    eta = test @ beta
                    row.update(coefficient_error=float(np.sum((beta-truth)**2)),
                               excess_log_loss=float(np.mean(np.logaddexp(0, eta)-true_mu*eta)-true_loss),
                               coefficient_norm=float(np.linalg.norm(beta)))
                    assert row['excess_log_loss'] >= -1e-12
                rows.append(row)
            if (trial+1) % 10 == 0:
                pd.DataFrame(rows).to_csv(args.out/'training_budget_trials.csv', index=False)
                save(args.out/'training_budget_status.json', {
                    'state': 'running', 'completed_model_draws': len(rows)//6,
                    'planned_model_draws': len(specs)*plan['repetitions_per_condition'],
                    'source_sha256': source_hash, 'plan_sha256': plan_hash})
                print(json.dumps({'condition': condition, 'completed': trial+1,
                                  'total_completed_draws': len(rows)//6}), flush=True)
        summaries(rows, args.out)
    save(args.out/'training_budget_status.json', {
        'state': 'complete', 'completed_model_draws': len(rows)//6,
        'planned_model_draws': len(specs)*plan['repetitions_per_condition'],
        'planned_method_rows': len(specs)*plan['repetitions_per_condition']*6,
        'returned_method_rows': len(rows), 'source_sha256': source_hash, 'plan_sha256': plan_hash,
        'total_simulated_label_draws': independent_labels,
        'human_questionnaire_records_used': 0, 'new_human_or_LLM_responses': 0,
        'unresolved_method_rows': sum('unresolved' in r['status'] for r in rows),
        'independent_novelty_or_submission_readiness_established': False})
    print((args.out/'training_budget_status.json').read_text(), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
