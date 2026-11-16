"""Matched point-estimation audit of clipping and auxiliary features.

The design, target and nuisance predictions are held fixed within each clipping
comparison. This reanalysis changes the source's prefix-pool convention and is
not represented as reproduction of its published tables.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from public_conjoint_optimizer_audit import load_data,Logistic,adam,converged,save
from choice_moment_support import solve_logit


def nuisance(x,y,z,m,use_ai):
    prior=float(np.mean(y[:m]));g=np.zeros(len(y));aug=np.zeros(len(y)-m)
    fallback=0;groups=np.unique(z[:m]) if use_ai else np.array([0])
    for tr,te in KFold(5,shuffle=True,random_state=0).split(x[:m]):
        for group in groups:
            train=tr[z[tr]==group] if use_ai else tr
            test=te[z[te]==group] if use_ai else te
            aid=np.where(z[m:]==group)[0] if use_ai else np.arange(len(aug))
            if len(train)>=2 and len(np.unique(y[train]))==2:
                fit=LogisticRegression(C=.05,max_iter=2000,random_state=0).fit(x[train],y[train])
                if len(test):g[test]=fit.predict_proba(x[test])[:,1]
                if len(aid):aug[aid]+=fit.predict_proba(x[m:][aid])[:,1]
            else:
                g[test]=prior;aug[aid]+=prior;fallback+=len(test)
        if use_ai:
            absent=~np.isin(z[m:],groups)
            aug[absent]+=prior
    g[m:]=aug/5
    assert np.isfinite(g).all() and np.min(g)>=0 and np.max(g)<=1
    return g,fallback


def summarize(rows,out):
    data=pd.DataFrame(rows);data.to_csv(out/'trial_diagnostics.csv',index=False)
    rates=data.groupby(['label_rows','method','status'],as_index=False).size()
    rates.to_csv(out/'status_counts.csv',index=False)
    metrics=[c for c in data.columns if c.startswith('MAPE_')]+['mean_score_corrected','mean_score_own','parameter_norm']
    data.groupby(['label_rows','method'],as_index=False).agg(
        allocations=('trial','size'),finite_estimates=('parameter_norm','count'),
        **{c:(c,'mean') for c in metrics}).to_csv(out/'point_estimation_summary.csv',index=False)
    paired=[]
    reference=data.query("method=='human_converged'").set_index(['trial','label_rows'])
    for method in data.method.unique():
        other=data[data.method.eq(method)].set_index(['trial','label_rows'])
        for m in [50,100,150,200]:
            both=reference.xs(m,level='label_rows').join(other.xs(m,level='label_rows'),lsuffix='_human',rsuffix='_other')
            keep=both.MAPE_prefix_projection_human.notna() & both.MAPE_prefix_projection_other.notna()
            diff=both.loc[keep,'MAPE_prefix_projection_other']-both.loc[keep,'MAPE_prefix_projection_human']
            paired.append({'label_rows':m,'method':method,'allocations':len(both),
                'shared_finite_allocations':int(keep.sum()),'mean_error_difference':float(diff.mean()) if len(diff) else None,
                'Monte_Carlo_SE_of_difference':float(diff.std(ddof=1)/np.sqrt(len(diff))) if len(diff)>1 else None,
                'scope':'conditional on shared finite fits; interpret with full status counts'})
    pd.DataFrame(paired).to_csv(out/'paired_point_comparisons.csv',index=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--allocations',type=int,default=500);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    x,y,z,reference,preflight=load_data(a.input_dir)
    full,_=converged(Logistic(x,y));prefix,_=converged(Logistic(x[:1200],y[:1200]))
    x=x[:1200];y=y[:1200];n=len(y);rng=np.random.RandomState(20261109)
    targets={'released_reference':reference,'full_human_projection':full,'prefix_projection':prefix}
    rows=[];maxclip=0.;fallback_count=0;failures=[]
    for trial in range(a.allocations):
        blocks=rng.permutation(240);order=(blocks[:,None]*5+np.arange(5)[None]).reshape(-1)
        xx=x[order];yy=y[order];zz=z[order]
        for m in [50,100,150,200]:
            g,fall=nuisance(xx,yy,zz,m,True);g0,fall0=nuisance(xx,yy,zz,m,False)
            fallback_count+=fall+fall0
            weight=np.r_[np.full(m,n/m),np.zeros(n-m)]
            raw=g+weight*(yy-g);raw0=g0+weight*(yy-g0)
            # This is exactly componentwise clipping plus binary renormalization.
            pair=np.column_stack([1-raw,raw]);pair=np.clip(pair,0.,1.)
            pair/=pair.sum(axis=1,keepdims=True);clipped=pair[:,1]
            expected=np.r_[yy[:m],g[m:]];err=float(max(abs(expected-clipped)))
            assert err<1e-12;maxclip=max(maxclip,err)
            primary=Logistic(xx[:m],yy[:m]);b_adam=adam(primary,200)
            cases=[('human_Adam200',b_adam,{'status':'fixed_iteration_output'},xx[:m],yy[:m])]
            specifications=[('human_converged',xx[:m],yy[:m]),
                ('naive_hard_AI',xx,np.r_[yy[:m],zz[m:]]),
                ('clipped_AI_self_training',xx,clipped),
                ('corrected_AI',xx,raw),('corrected_X_only',xx,raw0)]
            for method,features,target in specifications:
                b,check=solve_logit(features,target)
                cases.append((method,b,check,features,target))
            for method,b,check,features,target in cases:
                row={'trial':trial,'label_rows':m,'source_defined_label_blocks':m//5,
                     'auxiliary_rows':n-m,'method':method,'status':check['status'],
                     'support_margin':check.get('interior_margin')}
                if b is not None:
                    for name,truth in targets.items():row['MAPE_'+name]=float(100*np.mean(abs(b-truth)/(1+abs(truth))))
                    row.update(parameter_norm=float(np.linalg.norm(b)),
                        mean_score_own=float(max(abs(Logistic(features,target).value(b)[1]))),
                        mean_score_corrected=float(max(abs(Logistic(xx,raw0 if method=='corrected_X_only' else raw).value(b)[1]))))
                else:
                    failures.append({'trial':trial,'label_rows':m,'method':method,**check})
                rows.append(row)
        if (trial+1)%10==0 or trial+1==a.allocations:
            summarize(rows,a.out)
            save(a.out/'matched_numeric_checks.json',{'completed_allocations':trial+1,'requested_allocations':a.allocations,
                'fixed_clipping_comparisons':4*(trial+1),'maximum_clipping_identity_error':maxclip,
                'nuisance_fallback_primary_predictions':fallback_count,'methods':6,
                'target':'fixed1200-row empirical human projection','new_AI_generation':False,
                'new_human_participants':0,'confidence_interval_validity_evaluated':False})
            save(a.out/'unresolved_and_nonfinite_cases.json',failures)
            print('completed allocations',trial+1,'of',a.allocations,'recorded rows',len(rows),flush=True)
    print(pd.read_csv(a.out/'point_estimation_summary.csv').to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
