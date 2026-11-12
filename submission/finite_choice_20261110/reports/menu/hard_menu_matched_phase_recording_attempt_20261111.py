"""Matched finite-menu availability for hard predictions and established power.

All methods share each draw and dimension. Ordinary separation is a feasible
boundary moment and gets an independent weak-separator certificate.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from menu_corrected_logit import MenuLoss,menu_features,support,residual_support,save


def weak_separator(features,y):
    n,J,p=features.shape
    difference=np.concatenate([features[i,y[i]]-features[i,np.arange(J)!=y[i]] for i in range(n)],axis=0)
    summed=difference.mean(0)
    fit=linprog(np.r_[-summed,summed],A_ub=np.vstack([np.c_[-difference,difference],np.ones((1,2*p))]),
                b_ub=np.r_[np.zeros(len(difference)),1.],bounds=(0,None),method='highs')
    if not fit.success:return dict(state='unresolved')
    direction=fit.x[:p]-fit.x[p:];margins=difference@direction
    verified=bool(margins.min()>-1e-8 and margins.mean()>1e-8)
    return dict(state='verified' if verified else 'unresolved',minimum_margin=float(margins.min()),
                mean_margin=float(margins.mean()),direction=direction.tolist())


def residual_crosscheck(features,y,g,m,power):
    original=residual_support(features,y,g,m,power)
    if original!='unresolved':return original,None
    # Translate lower-bounded residuals to nonnegative variables. This is the
    # same feasible set,solved by a different HiGHS algorithm after an unknown.
    n,J,p=features.shape;tau=(n-m)/m
    lower=np.vstack([power*g[:m]-np.eye(J)[y[:m]],-power*g[m:]/tau])
    eq=sparse.vstack([sparse.kron(sparse.eye(n),np.ones((1,J))),sparse.csr_matrix(features.reshape(n*J,p).T)])
    target=-eq@lower.ravel()
    fit=linprog(np.zeros(n*J),A_eq=eq,b_eq=target,bounds=(0,None),method='highs-ipm',
                options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    residual=float(np.max(abs(eq@fit.x-target))) if fit.x is not None else None
    state='outside' if fit.status==2 else 'feasible' if fit.success and residual<1e-7 else 'unresolved'
    return state,dict(initial_status=original,method='highs-ipm',translated_nonnegative_variables=True,
        solver_status=fit.status,message=fit.message,equality_residual=residual,resolved_status=state)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    plan=json.loads(a.plan.read_text());source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rng=np.random.default_rng(plan['seed']);records=[];certificates=[];fallbacks=[];crosschecks=0;draws=0
    J=plan['alternatives']
    for m in plan['label_menus']:
      N=int(plan['tau']*m);n=m+N;dimensions=sorted({max(1,round(m*k)) for k in plan['dimension_ratios']})
      for trial in range(plan['replications_per_budget']):
        S=rng.normal(size=(n,J,1));W=rng.normal(size=(n,J,max(dimensions)))
        mu=softmax(plan['gamma']*S[...,0],axis=1)
        y=np.sum(rng.random(n)[:,None]>np.cumsum(mu,axis=1),axis=1)
        z=(y+(rng.random(n)<plan['epsilon'])*rng.integers(1,J,size=n))%J;g=np.eye(J)[z]
        for q in dimensions:
          features=menu_features(np.concatenate([S,W[...,:q]],axis=2))
          for method,power in plan['powers'].items():
            loss=MenuLoss(features,y,g,m,power);state=support(loss);certificate=None;independent=None
            if method=='ordinary' and state['status']=='boundary_or_near_boundary':
                certificate=weak_separator(loss.features[:m],y[:m]);certificate.update(label_menus=m,trial=trial,orthogonal_slopes=q)
                certificates.append(certificate)
                if certificate['state']!='verified':state['status']='unresolved_boundary'
            if power>0 and trial%50==0:
                independent,fallback=residual_crosscheck(features,y,g,m,power);crosschecks+=1
                if fallback is not None:fallbacks.append(dict(label_menus=m,trial=trial,orthogonal_slopes=q,method=method,**fallback))
                expected='feasible' if state['status'] in ['interior','boundary_or_near_boundary'] else state['status']
                if independent!='unresolved':assert independent==expected,(m,trial,q,method,state,independent)
            records.append(dict(label_menus=m,auxiliary_menus=N,trial=trial,orthogonal_slopes=q,dimension_ratio=q/m,
                method=method,power=power,critical_ratio=plan['critical_ratios'][method],status=state['status'],
                interior_margin=state.get('margin'),exterior_gap=state.get('separation_gap'),
                ordinary_separator_verified=certificate['state']=='verified' if certificate else None,
                independent_residual_status=independent))
        draws+=1
        if (trial+1)%20==0 or trial+1==plan['replications_per_budget']:
            pd.DataFrame(records).to_csv(a.out/'hard_menu_matched_phase_trials.csv',index=False)
            save(a.out/'hard_menu_ordinary_separation_certificates.json',certificates)
            save(a.out/'hard_menu_residual_fallbacks.json',fallbacks)
            save(a.out/'hard_menu_matched_phase_status.json',dict(state='running',completed_model_draws=draws,planned_model_draws=1000,completed_classifications=len(records),planned_classifications=9000,source_sha256=source))
            print(json.dumps(dict(menus=m,completed_repetitions=trial+1,classifications=len(records))),flush=True)
    frame=pd.DataFrame(records);keys=['label_menus','auxiliary_menus','orthogonal_slopes','dimension_ratio','method','power','critical_ratio']
    summary=frame.groupby(keys,as_index=False).agg(replications=('status','size'),finite=('status',lambda x:int(x.eq('interior').sum())),
        exterior=('status',lambda x:int(x.eq('outside').sum())),boundary=('status',lambda x:int(x.eq('boundary_or_near_boundary').sum())),
        unresolved=('status',lambda x:int(x.isin(['unresolved','unresolved_boundary']).sum())),rank_deficient=('status',lambda x:int(x.eq('rank_deficient').sum())))
    summary['finite_frequency']=summary.finite/summary.replications
    summary['binomial_MCSE']=np.sqrt(summary.finite_frequency*(1-summary.finite_frequency)/summary.replications)
    summary.to_csv(a.out/'hard_menu_matched_phase_summary.csv',index=False)
    save(a.out/'hard_menu_ordinary_separation_certificates.json',certificates)
    assert len(frame)==9000 and len(summary)==18 and summary.replications.eq(500).all()
    save(a.out/'hard_menu_matched_phase_status.json',dict(state='complete',model_draws=draws,support_classifications=len(frame),
        cells=len(summary),replications_per_cell=500,independent_residual_LP_checks=crosschecks,
        ordinary_boundary_certificates=len(certificates),unresolved=int(summary.unresolved.sum()),
        unresolved_independent_crosschecks=int(frame.independent_residual_status.eq('unresolved').sum()),residual_fallbacks=len(fallbacks),
        rank_deficient=int(summary.rank_deficient.sum()),positive_power_boundary=int(summary.loc[summary.power>0,'boundary'].sum()),
        source_sha256=source,plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),
        scope='Matched availability illustration for hard predicted categories; no coefficient accuracy or interval validation',
        new_human_or_LLM_responses=0))
    print((a.out/'hard_menu_matched_phase_status.json').read_text(),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
