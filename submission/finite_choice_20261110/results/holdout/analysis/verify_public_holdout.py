"""Independent original-loss and holdout-metric checks for the public diagnostic.

The verifier reconstructs allocation orders, original corrected gradients,
Firth log determinants and metrics. It does not call the primary risk or
corrected-ridge helpers, and it does not treat allocation MCSE as population SE.
"""
import argparse
import hashlib
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from public_conjoint_optimizer_audit import load_data, save
from public_conjoint_matched_audit import nuisance


def direct_firth(x, y, b):
    u = x@b
    info = x.T@((expit(u)*expit(-u))[:,None]*x)
    sign, determinant = np.linalg.slogdet(info)
    assert sign > 0
    return float(np.sum(np.logaddexp(0,u)-y*u)-determinant/2)


def verify(inputs, plan_path, results, out, canonical):
    out.mkdir(parents=True,exist_ok=True)
    plan = json.loads(plan_path.read_text())
    status = json.loads((results/'status.json').read_text())
    assert status['state'] == 'complete' and status['method_rows'] == 14000
    provenance = json.loads((results/'provenance.json').read_text())
    for name, expected in provenance['canonical_source_sha256'].items():
        assert hashlib.sha256((canonical/name).read_bytes()).hexdigest() == expected
    assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == provenance['plan_sha256']
    x,y,z,_,_ = load_data(inputs)
    tx,ty = x[1200:], y[1200:]
    x,y = x[:1200],y[:1200]
    coefficient_path = results/'coefficients.json'
    if coefficient_path.exists():
        co = json.loads(coefficient_path.read_text())
    else:
        co = json.loads(gzip.decompress((results/'coefficients.json.gz').read_bytes()))
    assert len(co) == 14000
    lookup = {(r['trial'],r['label_rows'],r['prediction'],r['method']):r for r in co}
    assert len(lookup) == len(co)
    records = pd.read_csv(results/'fit_records.csv').set_index(['trial','label_rows','prediction','method'])
    allocation_hashes = json.loads((results/'allocation_order_sha256.json').read_text())
    starts = pd.DataFrame(json.loads((results/'Firth_starts.json').read_text()))
    assert len(starts) == 3000
    rng = np.random.RandomState(plan['allocation_seed'])
    max_metric = max_score = max_firth_fd = max_firth_obj = max_power = max_invariance = 0.
    directions = []
    reconstructed = []
    unique_firth = 0
    root_count = missing_count = 0
    for trial in range(500):
        order = (rng.permutation(240)[:,None]*5+np.arange(5)[None]).reshape(-1)
        assert hashlib.sha256(order.astype('<i8').tobytes()).hexdigest() == allocation_hashes[trial]
        xx,yy,zz = x[order],y[order],z[order]
        for m in plan['label_budgets']:
            for method in ['human_MLE','human_ridge','human_Firth']:
                left,right = [lookup[(trial,m,p,method)] for p in ['AI','X_only']]
                assert left['beta'] == right['beta'] and left['check'] == right['check']
            for pred, use_ai in [('AI',True),('X_only',False)]:
                g,_ = nuisance(xx,yy,zz,m,use_ai)
                if trial in [0,249,499]:
                    changed = yy.copy();changed[m:] = 1-changed[m:]
                    alt,_ = nuisance(xx,changed,zz,m,use_ai)
                    error = float(np.max(abs(alt-g)))
                    max_invariance = max(max_invariance,error)
                    assert error == 0.
                for method in plan['methods']:
                    key = (trial,m,pred,method)
                    item = lookup[key];b = item['beta'];row = records.loc[key]
                    assert row.status == item['check']['status']
                    if b is None:
                        missing_count += 1
                        assert row[['logloss','brier','error']].isna().all()
                        check = item['check']
                        if check['status'] == 'outside_mean_support':
                            ray = np.array(check['direction'])
                            lam = float(row.power)
                            w = np.r_[np.full(m,(1-lam)/m),np.full(1200-m,lam/(1200-m))]
                            target = xx[:m].T@(yy[:m]-lam*g[:m])/m+lam*xx[m:].T@g[m:]/(1200-m)
                            gap = float(target@ray-w@np.maximum(xx@ray,0))
                            assert gap > 1e-8 and abs(gap-check['separation_gap']) < 1e-10
                            ray_value = float(w@np.logaddexp(0,1e6*(xx@ray))-1e6*(target@ray))
                            assert abs(ray_value-check['ray_objective_1e6']) < 1e-7
                            directions.append(dict(trial=trial,label_rows=m,prediction=pred,method=method,
                                gap=gap,objective_at_million=ray_value))
                        continue
                    root_count += 1
                    b = np.array(b,float)
                    u = tx@b;prob = expit(u)
                    metrics = dict(logloss=float(np.mean(np.logaddexp(0,u)-ty*u)),
                        brier=float(np.mean((prob-ty)**2)),error=float(np.mean((prob>=.5)!=ty)))
                    max_metric = max(max_metric,*[abs(metrics[k]-row[k]) for k in metrics])
                    reconstructed.append(dict(trial=trial,label_rows=m,prediction=pred,method=method,**metrics))
                    p = expit(xx@b)
                    if method in ['human_MLE','human_ridge']:
                        gradient = xx[:m].T@(p[:m]-yy[:m])/m
                        if method == 'human_ridge':gradient += b/m
                    elif method == 'human_Firth':
                        if pred == 'AI':
                            unique_firth += 1
                            fun = lambda bb:direct_firth(xx[:m],yy[:m],bb)
                            step = 2e-5
                            fd = [(fun(b+step*e)-fun(b-step*e))/(2*step) for e in np.eye(len(b))]
                            max_firth_fd = max(max_firth_fd,float(np.max(abs(np.array(fd)))))
                            max_firth_obj = max(max_firth_obj,abs(fun(b)-item['check']['objective']))
                            accepted = starts[(starts.trial==trial)&(starts.label_rows==m)&starts.accepted]
                            assert len(accepted) == item['check']['accepted_starts']
                            assert abs(fun(b)-accepted.objective.min()) < 1e-8
                        continue
                    elif method == 'soft_pooling':
                        gradient = xx.T@(p-np.r_[yy[:m],g[m:]])/1200
                    else:
                        lam = float(row.power)
                        # Direct derivative of H + lambda(A - H_g), not the
                        # weighted-loss class used by the primary implementation.
                        gradient = xx[:m].T@(p[:m]-yy[:m])/m + lam*(
                            xx[m:].T@(p[m:]-g[m:])/(1200-m)-xx[:m].T@(p[:m]-g[:m])/m)
                        if method == 'ridge_PPI':gradient += b/m
                        if method == 'power_ridge_init':
                            rb = np.array(lookup[(trial,m,pred,'human_ridge')]['beta'])
                            mu = expit(xx@rb)
                            inv = np.linalg.inv(xx.T@((mu*(1-mu))[:,None]*xx)/1200)
                            human_score = xx[:m]*(mu[:m]-yy[:m])[:,None]
                            predicted_score = xx*(mu-g)[:,None]
                            a = human_score-human_score.mean(0)
                            d = predicted_score[:m]-predicted_score[:m].mean(0)
                            numerator = np.sum((a@inv)*(d@inv))/m
                            s = predicted_score-predicted_score.mean(0)
                            denominator = (1+m/(1200-m))*np.sum((s@inv)**2)/(1200-1)
                            expected_power = float(np.clip(numerator/denominator,0.,1.))
                            max_power = max(max_power,abs(expected_power-lam))
                    max_score = max(max_score,float(np.max(abs(gradient))))
        if (trial+1) % 100 == 0:print('verified allocations',trial+1,'/500',flush=True)
    assert max_metric < 1e-12 and max_score < 1e-7 and max_power < 1e-9
    assert max_firth_fd < 1e-6 and max_firth_obj < 1e-8
    rebuilt = pd.DataFrame(reconstructed)
    original = pd.read_csv(results/'summary.csv').set_index(['label_rows','prediction','method'])
    grouped = rebuilt.groupby(['label_rows','prediction','method'])
    max_summary = 0.
    for key,group in grouped:
        assert len(group) == int(original.loc[key,'available'])
        for metric in ['logloss','brier','error']:
            max_summary = max(max_summary,abs(float(group[metric].mean())-original.loc[key,metric]))
    assert max_summary < 1e-12
    # Reconcile all paired comparisons and all-allocation fallback policies.
    paired = pd.read_csv(results/'paired_against_ridge.csv')
    fallback = pd.read_csv(results/'unit_power_ridge_fallback.csv')
    max_paired = 0.
    for r in paired.itertuples():
        group = records.xs((r.label_rows,r.prediction),level=('label_rows','prediction'))
        ridge = group.xs('human_ridge',level='method')[r.metric]
        other = group.xs(r.method,level='method')[r.metric]
        diff = (other-ridge).dropna()
        assert len(diff) == r.shared_available
        max_paired = max(max_paired,abs(float(diff.mean())-r.mean_difference),
            abs(float(diff.std(ddof=1)/np.sqrt(len(diff)))-r.allocation_MCSE))
    for r in fallback.itertuples():
        group = records.xs((r.label_rows,r.prediction),level=('label_rows','prediction'))
        ridge = group.xs('human_ridge',level='method')[r.metric]
        other = group.xs('unit_PPI',level='method')[r.metric]
        assert int(other.isna().sum()) == r.ridge_fallbacks
        diff = other.fillna(ridge)-ridge
        max_paired = max(max_paired,abs(float(diff.mean())-r.mean_difference),
            abs(float(diff.std(ddof=1)/np.sqrt(len(diff)))-r.allocation_MCSE))
    assert max_paired < 1e-12
    pd.DataFrame(directions).to_csv(out/'original_objective_exterior_checks.csv',index=False)
    save(out/'verification.json',dict(state='passed',method_records=14000,
        available_records=root_count,unavailable_records=missing_count,
        unique_Firth_designs=unique_firth,exterior_directions=len(directions),
        maximum_holdout_metric_difference=max_metric,maximum_original_gradient=max_score,
        maximum_Firth_direct_finite_difference_gradient=max_firth_fd,
        maximum_Firth_log_determinant_difference=max_firth_obj,
        maximum_power_recalculation_difference=max_power,
        maximum_summary_difference=max_summary,maximum_paired_summary_difference=max_paired,
        auxiliary_hidden_outcome_invariance=max_invariance,
        baseline_duplicates_checked=True,population_independence_certified=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['input-dir','plan','results','out','canonical']:
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    with threadpool_limits(limits=1):verify(a.input_dir,a.plan,a.results,a.out,a.canonical)
