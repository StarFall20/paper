"""Fixed-human public-choice auxiliary-budget audit of an existing power rule.

No predictor, power rule or finite-support criterion is introduced here.
Human auxiliary outcomes are replaced by NaN before fitting or auditing.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.special import expit
from threadpoolctl import threadpool_limits
from public_conjoint_optimizer_audit import load_data, save
from public_conjoint_matched_audit import nuisance
from power_tuned_choice_audit import objective, support, power_weight
from observable_low_label_path import ridge_anchor


def ordinary_check(x, y):
    signed=(2*y-1)[:,None]*x
    fit=linprog(-signed.mean(0),A_ub=-signed,b_ub=np.zeros(len(y)),
                bounds=[(-1.,1.)]*x.shape[1],method='highs-ipm')
    assert fit.success
    d=fit.x; margins=signed@d
    assert margins.min()>-1e-8
    mean=float(margins.mean())
    if mean>1e-8:
        return dict(status='nonattained_separated',direction=d.tolist(),
                    minimum_margin=float(margins.min()),mean_margin=mean)
    assert mean<1e-10
    return dict(status='overlap',maximum_separation_objective=mean)


def diagnostics(x, y, g, m, b, power):
    mu=expit(x@b)
    info=x.T@((mu*(1-mu))[:,None]*x)/len(x)
    inverse=np.linalg.inv(info)
    h=x[:m]*(mu[:m]-y[:m])[:,None]
    pred=x*(mu-g)[:,None]
    aa=h-h.mean(0);bb=pred[:m]-pred[:m].mean(0)
    cross=(aa.T@bb+bb.T@aa)/m
    pooled=np.cov(pred.T,ddof=1)
    n=len(x)-m
    source=(aa.T@aa/m-power*cross+(1+m/n)*power**2*pooled)/m
    labelled=h-power*pred[:m]
    psd=np.cov(labelled.T,ddof=1)/m+power**2*np.cov(pred[m:].T,ddof=1)/n
    return dict(source_quadratic_trace=float(np.trace(inverse@source@inverse)),
                psd_score_trace=float(np.trace(inverse@psd@inverse)),
                source_middle_minimum_eigenvalue=float(np.linalg.eigvalsh(source).min()),
                pooled_information_minimum_eigenvalue=float(np.linalg.eigvalsh(info).min()))


def summaries(rows, out):
    f=pd.DataFrame(rows)
    f.to_csv(out/'records.csv',index=False)
    f.groupby(['label_rows','prediction','auxiliary_rows','method','status'],dropna=False).size().reset_index(name='allocations').to_csv(out/'status_counts.csv',index=False)
    summary=[];pairs=[]
    for (m,pred,method),group in f.groupby(['label_rows','prediction','method']):
        for n,c in group.groupby('auxiliary_rows'):
            summary.append(dict(label_rows=m,prediction=pred,method=method,auxiliary_rows=n,
                allocations=len(c),interior=int(c.status.eq('interior').sum()),
                exterior=int(c.status.eq('outside_mean_support').sum()),
                human_separated=int(c.ordinary_status.eq('nonattained_separated').sum()),
                boundary=int(c.status.eq('boundary_or_near_boundary').sum()),
                unresolved=int((~c.status.isin(['interior','outside_mean_support','boundary_or_near_boundary'])).sum()),
                mean_power=float(c.power.mean()),zero_powers=int(c.power.eq(0).sum()),
                mean_psd_score_trace=float(c.psd_score_trace.mean())))
        first=group[group.auxiliary_rows.eq(group.auxiliary_rows.min())].set_index('trial')
        previous=first
        for n,current in group.groupby('auxiliary_rows',sort=True):
            if n==first.auxiliary_rows.iloc[0]:continue
            current=current.set_index('trial')
            for basis,base in [('first',first),('previous',previous)]:
                assert current.index.equals(base.index)
                a=base.status.eq('interior');b=current.status.eq('interior')
                lost=a&current.status.eq('outside_mean_support')
                gained=base.status.eq('outside_mean_support')&b
                diff=b.to_numpy(int)-a.to_numpy(int)
                trace_fell=current.psd_score_trace<base.psd_score_trace
                pairs.append(dict(label_rows=m,prediction=pred,method=method,basis=basis,
                    from_auxiliary_rows=int(base.auxiliary_rows.iloc[0]),to_auxiliary_rows=int(n),
                    allocations=len(a),finite_to_exterior=int(lost.sum()),exterior_to_finite=int(gained.sum()),
                    finite_difference=float(diff.mean()),paired_allocation_MCSE=float(diff.std(ddof=1)/np.sqrt(len(diff))),
                    finite_to_exterior_with_lower_psd_score_trace=int((lost&trace_fell).sum())))
            previous=current
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'paired_transitions.csv',index=False)


def main():
    a=argparse.ArgumentParser();a.add_argument('--plan',type=Path,required=True)
    a.add_argument('--input-dir',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a.add_argument('--resume',action='store_true');args=a.parse_args()
    plan=json.loads(args.plan.read_text());out=args.out;out.mkdir(parents=True,exist_ok=True)
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();plan_sha=hashlib.sha256(args.plan.read_bytes()).hexdigest()
    x,y,z,_,preflight=load_data(args.input_dir);x=x[:1200];y=y[:1200]
    rng=np.random.RandomState(plan['seed']);rows=[];directions=[];human=[];done=set();calls=0
    if args.resume:
        status=json.loads((out/'status.json').read_text())
        assert status['source_sha256']==source_sha and status['plan_sha256']==plan_sha
        rows=pd.read_csv(out/'records.csv').replace({np.nan:None}).to_dict('records')
        directions=json.loads((out/'exterior_directions.json').read_text())
        human=json.loads((out/'ordinary_checks.json').read_text());done={int(r['trial']) for r in rows}
        assert all(sum(int(r['trial'])==t for r in rows)==72 for t in done)
        calls=status['unique_support_programs']
    else:assert not (out/'status.json').exists(),'Use a new output directory or matching --resume'
    def checkpoint(state):
        if rows:summaries(rows,out)
        save(out/'exterior_directions.json',directions);save(out/'ordinary_checks.json',human)
        save(out/'status.json',dict(state=state,completed_allocations=len(done),requested_allocations=plan['allocations'],
            method_records=len(rows),unique_support_programs=calls,source_sha256=source_sha,plan_sha256=plan_sha,
            new_human_or_AI_answers=0,auxiliary_human_labels_hidden=True))
    checkpoint('running')
    for trial in range(plan['allocations']):
        order=(rng.permutation(240)[:,None]*5+np.arange(5)[None]).reshape(-1)
        if trial in done:continue
        xx=x[order];zz=z[order]
        for m in plan['label_budgets']:
            # Keep the row count, with every auxiliary human outcome hidden.
            yy=np.r_[y[order[:m]],np.full(len(xx)-m,np.nan)]
            anchor,pilot_error=ridge_anchor(xx[:m],yy[:m])
            ordinary=ordinary_check(xx[:m],yy[:m]);human.append(dict(trial=trial,label_rows=m,**ordinary))
            for pred,use_ai in [('AI',True),('X_only',False)]:
                g,fallback=nuisance(xx,yy,zz,m,use_ai)
                initial=None
                for n in [int(v) if isinstance(v,int) else len(xx)-m for v in plan['auxiliary_budgets']]:
                    nx=xx[:m+n];ny=yy[:m+n];ng=g[:m+n]
                    power,tune=power_weight(nx,ny,ng,m,anchor)
                    assert power is not None,tune
                    if initial is None:initial=power
                    cache={}
                    for method,lam in [('source_tuned_each_N',power),('fixed_first_N_power',initial),('unit',1.)]:
                        loss=objective(nx,ny,ng,m,lam)
                        if lam not in cache:
                            result=support(loss);calls+=1;cache[lam]=result
                        else:result=cache[lam]
                        if result['status']=='boundary_or_near_boundary' and lam==0:
                            assert ordinary['status']=='nonattained_separated'
                        elif result['status'] not in ['interior','outside_mean_support','boundary_or_near_boundary']:
                            save(out/'unresolved_case.json',dict(trial=trial,label_rows=m,prediction=pred,auxiliary_rows=n,method=method,power=lam,check=result));checkpoint('stopped_on_unresolved')
                            raise RuntimeError('Unresolved primary support program')
                        if result['status']=='outside_mean_support':
                            direction=np.array(result['direction']);u=nx@direction
                            # Direct original two-sample recession, not a solver objective.
                            direct=float(np.mean(np.maximum(u[:m],0)-ny[:m]*u[:m])+lam*(np.mean(np.maximum(u[m:],0)-ng[m:]*u[m:])-np.mean(np.maximum(u[:m],0)-ng[:m]*u[:m])))
                            assert direct<-1e-8
                            directions.append(dict(trial=trial,label_rows=m,prediction=pred,auxiliary_rows=n,method=method,power=lam,direction=direction.tolist(),original_recession=direct))
                        rows.append(dict(trial=trial,label_rows=m,prediction=pred,auxiliary_rows=n,method=method,power=lam,
                            status=result['status'],ordinary_status=ordinary['status'],interior_margin=result.get('interior_margin'),
                            separation_gap=result.get('separation_gap'),nuisance_fallbacks=fallback,pilot_score_max=pilot_error,
                            **diagnostics(nx,ny,ng,m,anchor,lam)))
        done.add(trial)
        if len(done)%20==0 or len(done)==plan['allocations']:
            checkpoint('running');print(json.dumps(dict(allocations=len(done),records=len(rows),exterior_directions=len(directions))),flush=True)
    assert len(rows)==36000 and len(human)==1000
    checkpoint('complete')


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
