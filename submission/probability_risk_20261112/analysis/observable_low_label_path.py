"""Low-label public-design follow-up with a finite ridge anchor and all failures."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.optimize import minimize,linprog
from scipy.special import expit
from threadpoolctl import threadpool_limits
from public_conjoint_optimizer_audit import load_data,Logistic,save
from public_conjoint_matched_audit import nuisance
from power_tuned_choice_audit import objective,support,power_weight
from observable_score_path_audit import path_prediction,score_rows,middle_matrix,residual_check,summaries


def ridge_anchor(x,y):
    loss=Logistic(x,y);rho=1/len(y)
    def value(b):
        v,g=loss.value(b);return v+rho*(b@b)/2,g+rho*b
    fit=minimize(value,np.zeros(x.shape[1]),jac=True,method='BFGS',options={'gtol':1e-10,'maxiter':1500})
    b=fit.x.copy()
    for _ in range(6):
        v,g=value(b)
        if np.max(abs(g))<1e-11:break
        delta=np.linalg.solve(loss.hessian(b)+rho*np.eye(len(b)),g);step=1.
        while value(b-step*delta)[0]>v+1e-13 and step>1e-8:step/=2
        b-=step*delta
    error=float(np.max(abs(value(b)[1])));assert error<1e-9,error
    return b,error


def separator(x,y):
    signed=x*(2*y-1)[:,None];p=x.shape[1]
    fit=linprog(np.r_[-signed.mean(0),signed.mean(0)],A_ub=np.r_[np.c_[-signed,signed],np.ones((1,2*p))],b_ub=np.r_[np.zeros(len(y)),1.],bounds=(0.,None),method='highs-ipm')
    assert fit.success
    d=fit.x[:p]-fit.x[p:];margins=signed@d
    assert margins.min()>-1e-8 and margins.mean()>1e-8
    return dict(direction=d.tolist(),minimum_margin=float(margins.min()),mean_margin=float(margins.mean()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--input-dir',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--resume',action='store_true');a=parser.parse_args()
    plan=json.loads(a.plan.read_text());a.out.mkdir(parents=True,exist_ok=True)
    source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();plan_sha=hashlib.sha256(a.plan.read_bytes()).hexdigest()
    x,y,z,_,_=load_data(a.input_dir);x=x[:plan['prefix_rows']];y=y[:plan['prefix_rows']];m=plan['label_rows']
    rows=[];ordinary=[];certificates=[];complete=set();status_file=a.out/'observable_low_label_status.json'
    if a.resume:
        state=json.loads(status_file.read_text());assert state['source_sha256']==source and state['plan_sha256']==plan_sha
        rows=pd.read_csv(a.out/'observable_score_path_records.csv').to_dict('records');ordinary=json.loads((a.out/'ordinary_support_checks.json').read_text());certificates=json.loads((a.out/'path_certificates.json').read_text())
        complete={int(r['trial']) for r in rows};assert len(ordinary)==len(complete) and all(sum(int(r['trial'])==t for r in rows)==10 for t in complete)
    else:assert not status_file.exists(),'Use a new output directory or --resume'
    def checkpoint(state):
        if rows:summaries(rows,a.out)
        save(a.out/'ordinary_support_checks.json',ordinary);save(a.out/'path_certificates.json',certificates)
        save(status_file,dict(state=state,completed_allocations=len(complete),path_records=len(rows),ordinary_checks=len(ordinary),source_sha256=source,plan_sha256=plan_sha,new_human_or_LLM_responses=0))
    checkpoint('running');rng=np.random.RandomState(plan['seed'])
    for trial in range(plan['allocations']):
        order=(rng.permutation(240)[:,None]*5+np.arange(5)[None]).reshape(-1)
        if trial in complete:continue
        xx=x[order];yy=y[order];zz=z[order];anchor,error=ridge_anchor(xx[:m],yy[:m]);mu=expit(xx@anchor)
        human=support(objective(xx,yy,np.full(len(yy),.5),m,0.))
        if human['status']=='boundary_or_near_boundary':human.update(separator(xx[:m],yy[:m]))
        assert human['status'] in ['interior','boundary_or_near_boundary']
        ordinary.append(dict(trial=trial,**human))
        for pred,use_ai in [('AI',True),('X_only',False)]:
            g,fallback=nuisance(xx,yy,zz,m,use_ai);alpha,tune=power_weight(xx,yy,g,m,anchor);assert alpha is not None
            base_loss=objective(xx,yy,g,m,alpha);base_gradient=base_loss.value(anchor)[1]
            base_l,base_u=score_rows(xx,yy,g,m,alpha,anchor);base_middle=middle_matrix(base_l,base_u)
            for fraction in plan['path_fractions']:
                power=float(alpha+fraction*(1-alpha));gs=path_prediction(mu,g,alpha,power);loss=objective(xx,yy,gs,m,power)
                labelled,auxiliary=score_rows(xx,yy,gs,m,power,anchor)
                gradient_error=float(np.max(abs(loss.value(anchor)[1]-base_gradient)))
                row_error=float(max(np.max(abs(labelled-base_l)),np.max(abs(auxiliary-base_u))))
                middle_error=float(np.max(abs(middle_matrix(labelled,auxiliary)-base_middle)))
                assert max(gradient_error,row_error,middle_error)<1e-10
                weight,weightmeta=power_weight(xx,yy,gs,m,anchor);weighterror=None if weight is None else abs(weight-power)
                if 0<alpha<1:assert weight is not None and weighterror<1e-8
                result=support(loss);independent=None
                if power==0 and result['status']=='boundary_or_near_boundary':
                    assert human['status']=='boundary_or_near_boundary'
                elif result['status'] not in ['interior','outside_mean_support']:
                    save(a.out/'unresolved_case.json',dict(trial=trial,prediction=pred,fraction=fraction,result=result));checkpoint('stopped_on_unresolved');raise RuntimeError('Retained unresolved support state')
                if power>0 and (trial%25==0 or result['status']=='outside_mean_support'):
                    check=residual_check(xx,yy,gs,m,power);independent=check['status']
                    assert independent==('infeasible' if result['status']=='outside_mean_support' else 'feasible')
                if result['status']=='outside_mean_support':
                    d=np.asarray(result['direction']);cost=float(loss.weights@np.maximum(xx@d,0)-loss.moment@d)
                    assert cost< -1e-8
                    certificates.append(dict(trial=trial,prediction=pred,fraction=fraction,power=power,base_power=alpha,direction=d.tolist(),recession_cost=cost))
                rows.append(dict(trial=trial,prediction=pred,fraction=fraction,base_power=alpha,power=power,status=result['status'],ordinary_status=human['status'],
                    interior_margin=result.get('interior_margin'),separation_gap=result.get('separation_gap'),anchor_gradient_error=gradient_error,full_score_row_error=row_error,full_score_middle_error=middle_error,
                    recomputed_power=weight,recomputed_power_error=weighterror,recomputed_weight_status=weightmeta['tuning_status'],independent_residual_status=independent,nuisance_fallbacks=fallback,anchor_ridge_score_max=error))
        complete.add(trial)
        if len(complete)%20==0:
            checkpoint('running');print(json.dumps(dict(allocations=len(complete),path_records=len(rows),exterior=len(certificates))),flush=True)
    assert len(rows)==plan['requested_path_records'] and len(ordinary)==plan['requested_ordinary_support_checks']
    checkpoint('complete');print('Low-label observable path complete.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
