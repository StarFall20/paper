"""Follow-up on all fixed DCE allocations: support intervals and two explicit rules."""
from pathlib import Path
import argparse, gzip, hashlib, json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from existing_panel_model import load, corrected, fit_loss, metrics

BASE = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, d): p.write_text(json.dumps(d, indent=2, allow_nan=False) + '\n')
OPTS = {'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9}

def frontier(zero, unit):
    f, w0, b0 = zero.f, zero.w, zero.b
    dw, db = unit.w-w0, unit.b-b0
    n, J, p = f.shape
    eq = np.block([[np.kron(np.eye(n), np.ones((1, J))), -dw[:, None]],
                   [f.reshape(-1, p).T, -db[:, None]]])
    rhs = np.r_[w0, b0]
    fit = linprog(np.r_[np.zeros(n*J), -1.], A_eq=eq, b_eq=rhs,
                  bounds=[(0, None)]*(n*J)+[(0, 1)], method='highs', options=OPTS)
    assert fit.success, (fit.status, fit.message)
    assert np.max(abs(eq @ fit.x-rhs)) < 1e-7
    raw_power = float(fit.x[-1])
    assert -1e-9 <= raw_power <= 1+1e-9
    power = float(np.clip(raw_power, 0., 1.))
    dual = fit.eqlin.marginals[n:]
    return power, dict(raw_power=raw_power, weighted_probabilities=fit.x[:-1].reshape(n,J).tolist(),
                       equality_dual=fit.eqlin.marginals.tolist(),
                       power_lower_dual=float(fit.lower.marginals[-1]),
                       power_upper_dual=float(fit.upper.marginals[-1]),
                       direction=dual.tolist(),
                       primal_equality_error=float(np.max(abs(eq @ fit.x-rhs))))

def interior(loss):
    keep = loss.w > 0
    f, w = loss.f[keep], loss.w[keep]
    n, J, p = f.shape
    rank = int(np.linalg.matrix_rank((f[:, :-1]-f[:, -1:]).reshape(-1,p)))
    if rank < p: return dict(support='rank_deficient', rank=rank)
    eq = np.block([[np.kron(np.eye(n),np.ones((1,J))),np.zeros((n,1))],
                   [(w[:,None,None]*f).reshape(-1,p).T,np.zeros((p,1))]])
    ub = np.c_[-np.eye(n*J),np.ones(n*J)]
    fit = linprog(np.r_[np.zeros(n*J),-1.],A_ub=ub,b_ub=np.zeros(n*J),
                  A_eq=eq,b_eq=np.r_[np.ones(n),loss.b],
                  bounds=[(0,1)]*(n*J)+[(0,1/J)],method='highs',options=OPTS)
    if fit.status == 2: return dict(support='outside',rank=rank)
    if not fit.success: return dict(support='unresolved',rank=rank,solver_status=int(fit.status))
    error=float(np.max(abs(eq@fit.x-np.r_[np.ones(n),loss.b])))
    assert error < 1e-7
    return dict(support='interior' if fit.x[-1]>1e-8 else 'boundary_or_near_boundary',
                rank=rank,margin=float(fit.x[-1]),equality_error=error,
                probabilities=fit.x[:-1].reshape(n,J).tolist(),kept_designs=np.flatnonzero(keep).tolist())

def main(path,out):
    assert not out.exists(), 'Use a new directory; never restart over saved outputs.'
    model_plan=json.loads((BASE/'inputs/model_plan.json').read_text())
    data=load(path,model_plan)
    pre=json.loads((BASE/'inputs/preflight.json').read_text())
    g=softmax(data['teacher']@np.array(pre['teacher_coefficients']),axis=2)
    rec=pd.read_csv(BASE/'inputs/baseline_records.csv')
    pilots={r['record']:np.array(r['beta']) for r in json.loads((BASE/'inputs/labelled_ridge_coefficients.json').read_text())}
    out.mkdir(parents=True)
    rows=[];fits=[];certificates=[];rng=np.random.default_rng(model_plan['allocation_seed'])
    for trial in range(500):
        order=rng.permutation(data['pool'])
        for m in model_plan['labelled_respondent_budgets']:
            L,U=order[:m],order[m:]
            old=rec[(rec.allocation==trial)&(rec.labelled_people==m)]
            pilot=pilots[int(old[old.method.eq('labelled_ridge')].iloc[0].record)]
            power=float(old[old.method.eq('corrected_cluster_power')].iloc[0].power)
            zero,unit=corrected(data,L,U,g,0.),corrected(data,L,U,g,1.)
            cap,fc=frontier(zero,unit)
            middle=interior(corrected(data,L,U,g,cap/2)) if cap>1e-10 else dict(support='zero_or_numerically_near_zero_cap')
            fid=len(rows)
            rows.append(dict(frontier_record=fid,allocation=trial,labelled_people=m,frontier=cap,raw_frontier=fc['raw_power'],
                             frontier_capped_at_one=cap>=1-1e-9,midpoint_state=middle['support'],
                             original_cluster_power=power,human_support=old[old.method.eq('labelled_MLE')].iloc[0].support))
            cert=dict(frontier_record=fid,**fc,midpoint=middle,policies={})
            for name,lam in [('half_frontier',cap/2),('projected_cluster_power',float(np.clip(power,.1*cap,.9*cap)))]:
                sid=len(fits)
                if middle['support']!='interior':
                    fits.append(dict(record=sid,frontier_record=fid,allocation=trial,labelled_people=m,method=name,
                                     power=lam,support='no_positive_interior_certified',fit='unavailable_by_support'))
                    continue
                loss=corrected(data,L,U,g,lam)
                state=middle if name=='half_frontier' else interior(loss)
                cert['policies'][name]=state
                row=dict(record=sid,frontier_record=fid,allocation=trial,labelled_people=m,method=name,power=lam,
                         **{k:v for k,v in state.items() if k not in ['probabilities','kept_designs']})
                if state['support']=='interior':
                    beta,check=fit_loss(loss,pilot);row.update(check)
                    if check['fit']=='passed_score':
                        row.update(metrics(data['f'][data['test']],data['y'][data['test']],beta));row['beta']=beta.tolist()
                else:row['fit']='unavailable_by_support'
                fits.append(row)
            certificates.append(cert)
        if (trial+1)%25==0:
            write(out/'status.json',dict(state='running',allocations_complete=trial+1,planned=500))
            print(json.dumps(dict(allocations_complete=trial+1,frontiers=len(rows),policy_records=len(fits))),flush=True)
    frame=pd.DataFrame(fits);frame.drop(columns=['beta']).to_csv(out/'policy_records.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'frontiers.csv',index=False)
    write(out/'coefficients.json',[dict(record=int(r['record']),beta=r['beta']) for r in fits if 'beta' in r])
    (out/'certificates.json.gz').write_bytes(gzip.compress(json.dumps(certificates,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    summaries=[]
    for m,part in frame.groupby('labelled_people'):
        old=rec[rec.labelled_people==m].pivot(index='allocation',columns='method',values='heldout_logloss')
        for name,group in part.groupby('method'):
            vals=group.set_index('allocation').heldout_logloss.reindex(old.index).fillna(old.labelled_ridge)
            paired={}
            for baseline in old.columns:
                reference=old[baseline].fillna(old.labelled_ridge);d=vals-reference
                paired[baseline]=dict(mean=float(d.mean()),allocation_MCSE=float(d.std(ddof=1)/np.sqrt(500)))
            good=group[group.fit.eq('passed_score')]
            summaries.append(dict(labelled_people=int(m),method=name,denominator=500,available=len(good),
                                  support_counts=group.support.value_counts().to_dict(),fit_counts=group.fit.value_counts().to_dict(),
                                  conditional_mean_logloss=float(good.heldout_logloss.mean()) if len(good) else None,
                                  all_allocation_fallback_mean_logloss=float(vals.mean()),comparisons=paired))
    write(out/'summary.json',summaries)
    write(out/'status.json',dict(state='complete',frontiers=len(rows),policy_records=len(fits),
                                available_coefficients=int(frame.fit.eq('passed_score').sum()),
                                source_sha256=sha(path),plan_sha256=sha(BASE/'plan.json'),analysis_sha256=sha(Path(__file__)),
                                no_new_predictor_training=True,no_new_MC_cohorts=True))
    print(json.dumps(dict(state='complete',frontiers=len(rows),policy_records=len(fits))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=1): main(a.input,a.out)
