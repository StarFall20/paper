"""Held-out-record diagnostic on existing public conjoint responses.

This is not an independent-respondent study or a new estimator. The unchanged
manuscript fitting routines are imported. Evaluation labels are never passed
to fitting or tuning. Every planned allocation, method and failure is retained.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from public_conjoint_optimizer_audit import load_data, save
from public_conjoint_matched_audit import nuisance
from firth_choice_comparator import fit_firth, logistic_fit
from power_tuned_choice_audit import WeightedLogit, objective, solve, power_weight


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def risk(x, y, beta):
    u = x @ beta
    p = expit(u)
    return dict(logloss=float(np.mean(np.logaddexp(0, u)-y*u)),
                brier=float(np.mean((p-y)**2)),
                error=float(np.mean((p >= .5) != y)))


def ridge_corrected(loss, m, initial):
    # This adds the existing ridge penalty to the original corrected objective.
    # A dummy pseudo-response would risk changing the objective's normalization.
    from scipy.optimize import minimize
    def value(beta):
        val, grad = loss.value(beta)
        return val + beta@beta/(2*m), grad + beta/m
    fit = minimize(value, initial, jac=True, method='BFGS',
                   options={'gtol': 1e-10, 'maxiter': 2000})
    beta = fit.x.copy()
    for _ in range(12):
        val, grad = value(beta)
        if np.max(abs(grad)) < 1e-10:
            break
        direction = np.linalg.solve(loss.hessian(beta)+np.eye(len(beta))/m, grad)
        step = 1.
        while step > 1e-10 and value(beta-step*direction)[0] > val+1e-13:
            step /= 2
        if step <= 1e-10:
            break
        beta -= step*direction
    score = float(np.max(abs(value(beta)[1])))
    assert score < 1e-7 and np.isfinite(beta).all(), score
    return beta, dict(status='finite_penalized_root', score_max=score,
                      objective=float(value(beta)[0]))


def summarize(rows, out):
    data = pd.DataFrame(rows)
    data.to_csv(out/'fit_records.csv', index=False)
    keys = ['label_rows', 'prediction', 'method']
    data.groupby(keys+['status'], as_index=False).size().to_csv(out/'status_counts.csv', index=False)
    data.groupby(keys, as_index=False).agg(
        allocations=('trial', 'size'), available=('logloss', 'count'),
        logloss=('logloss', 'mean'), brier=('brier', 'mean'), error=('error', 'mean'),
        logloss_median=('logloss', 'median'), logloss_q90=('logloss', lambda s:s.quantile(.9)),
        power=('power', 'mean')).to_csv(out/'summary.csv', index=False)
    paired = []
    policy = []
    for (m, pred), group in data.groupby(['label_rows', 'prediction']):
        baseline = group[group.method.eq('human_ridge')].set_index('trial')
        for method in group.method.unique():
            other = group[group.method.eq(method)].set_index('trial')
            joined = baseline.join(other, lsuffix='_ridge', rsuffix='_other')
            for metric in ['logloss', 'brier', 'error']:
                diff = (joined[metric+'_other']-joined[metric+'_ridge']).dropna()
                paired.append(dict(label_rows=m, prediction=pred, method=method, metric=metric,
                    allocations=len(joined), shared_available=len(diff),
                    mean_difference=diff.mean(),
                    allocation_MCSE=diff.std(ddof=1)/np.sqrt(len(diff)) if len(diff)>1 else np.nan))
            if method == 'unit_PPI':
                for metric in ['logloss', 'brier', 'error']:
                    values = other[metric].fillna(baseline[metric])
                    assert values.notna().all()
                    diff = values-baseline[metric]
                    policy.append(dict(label_rows=m, prediction=pred, metric=metric,
                        allocations=len(diff), ridge_fallbacks=int(other[metric].isna().sum()),
                        policy_mean=values.mean(), ridge_mean=baseline[metric].mean(),
                        mean_difference=diff.mean(), allocation_MCSE=diff.std(ddof=1)/np.sqrt(len(diff))))
    pd.DataFrame(paired).to_csv(out/'paired_against_ridge.csv', index=False)
    pd.DataFrame(policy).to_csv(out/'unit_power_ridge_fallback.csv', index=False)


def run(input_dir, plan_path, out, canonical):
    plan = json.loads(plan_path.read_text())
    assert plan['label_budgets'] == [50, 150] and plan['allocations'] == 500
    out.mkdir(parents=True, exist_ok=True)
    x, y, z, _, preflight = load_data(input_dir)
    test_x, test_y = x[1200:].copy(), y[1200:].copy()
    x, y = x[:1200].copy(), y[:1200].copy()
    assert len(test_y) == 3405 and len(x) == len(y) == len(z) == 1200
    # Fitting loops see only these arrays. The suffix is used solely in risk().
    source_names = ['public_conjoint_optimizer_audit.py', 'public_conjoint_matched_audit.py',
                    'power_tuned_choice_audit.py', 'choice_moment_support.py',
                    'firth_choice_comparator.py']
    provenance = dict(plan_sha256=sha(plan_path), analysis_sha256=sha(Path(__file__)),
        canonical_source_sha256={n:sha(canonical/n) for n in source_names},
        input_manifest_sha256=sha(input_dir/'source_manifest.json'),
        input_numeric_git_blob_sha1=[f['git_blob_sha1'] for f in preflight['source']['files']
                                    if f['path'].endswith('.pkl')])
    save(out/'provenance.json', provenance)
    save(out/'preflight.json', dict(training_rows=1200, evaluation_rows=3405, contrasts=11,
        training_outcome_mean=float(y.mean()), evaluation_outcome_mean=float(test_y.mean()),
        hard_AI_training_accuracy=float(np.mean(z == y)),
        respondent_ids_available=False, new_answers=0, reference_coefficients_used=False,
        holdout_used_for_training_or_tuning=False, scope=plan['evaluation_caveat']))
    rows, coefficients, attempts, order_hashes = [], [], [], []
    rng = np.random.RandomState(plan['allocation_seed'])
    for trial in range(plan['allocations']):
        order = (rng.permutation(240)[:, None]*5+np.arange(5)[None]).reshape(-1)
        order_hashes.append(hashlib.sha256(order.astype('<i8').tobytes()).hexdigest())
        xx, yy, zz = x[order], y[order], z[order]
        for m in plan['label_budgets']:
            ridge, ridge_grad = logistic_fit(xx[:m], yy[:m], rho=1/m)
            assert ridge_grad < 1e-7
            # lambda=0 uses the same original support/root routine as PPI.
            human, hc = solve(objective(xx, yy, np.zeros(len(yy)), m, 0.), ridge)
            firth, fa, fc = fit_firth(xx[:m], yy[:m])
            for attempt in fa:
                attempts.append(dict(trial=trial, label_rows=m, **attempt))
            for pred, use_ai in [('AI', True), ('X_only', False)]:
                g, fallback = nuisance(xx, yy, zz, m, use_ai)
                unit = objective(xx, yy, g, m, 1.)
                fixed, check_fixed = solve(unit, ridge)
                lam, tune = power_weight(xx, yy, g, m, ridge)
                if lam is None:
                    tuned, check_tuned = None, dict(status='weight_unresolved', **tune)
                else:
                    tuned, check_tuned = solve(objective(xx, yy, g, m, lam), ridge)
                regular, check_regular = ridge_corrected(unit, m, ridge)
                soft_target = np.r_[yy[:m], g[m:]]
                soft = WeightedLogit(xx, np.full(len(yy),1/len(yy)),
                                     xx.T@soft_target/len(yy))
                assert np.max(abs(soft.weights-1/1200)) < 1e-15
                assert np.max(abs(soft.moment-xx.T@soft_target/1200)) < 1e-12
                pooling, check_pool = solve(soft, ridge)
                cases = [('human_MLE', human, hc, 0.),
                         ('human_ridge', ridge, dict(status='finite_penalized_root',score_max=ridge_grad),0.),
                         ('human_Firth', firth, fc, 0.),
                         ('unit_PPI', fixed, check_fixed, 1.),
                         ('power_ridge_init', tuned, check_tuned, lam),
                         ('ridge_PPI', regular, check_regular, 1.),
                         ('soft_pooling', pooling, check_pool, None)]
                for method, beta, check, power in cases:
                    row = dict(trial=trial,label_rows=m,prediction=pred,method=method,
                        status=check['status'],power=power,nuisance_fallback_rows=fallback,
                        logloss=np.nan,brier=np.nan,error=np.nan,
                        score_max=check.get('score_max'),support_margin=check.get('interior_margin'),
                        separation_gap=check.get('separation_gap'))
                    if beta is not None:
                        row.update(risk(test_x, test_y, beta))
                        coefficients.append(dict(trial=trial,label_rows=m,prediction=pred,
                            method=method,beta=beta.tolist(),check=check))
                    else:
                        coefficients.append(dict(trial=trial,label_rows=m,prediction=pred,
                            method=method,beta=None,check=check))
                    rows.append(row)
        if (trial+1) % 25 == 0:
            summarize(rows, out)
            save(out/'status.json',dict(state='running',completed_allocations=trial+1,
                 planned_allocations=plan['allocations'],method_rows=len(rows)))
            print('completed allocations', trial+1, '/', plan['allocations'], flush=True)
    summarize(rows, out)
    save(out/'coefficients.json', coefficients)
    save(out/'Firth_starts.json', attempts)
    save(out/'allocation_order_sha256.json',order_hashes)
    save(out/'status.json',dict(state='complete',completed_allocations=500,
        planned_allocations=500,method_rows=len(rows),distinct_label_allocations=1000,
        duplicated_human_baseline_rows_across_prediction_routines=True,
        independent_respondents_claimed=False,population_uncertainty_tested=False,
        fulltext_PPI_formal_comparison_completed=False))
    assert len(rows) == 14000 and len(attempts) == 3000
    print(pd.read_csv(out/'summary.csv').to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--canonical',type=Path,required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.input_dir,args.plan,args.out,args.canonical)
