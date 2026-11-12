"""Matched point audit of published PPI power tuning for binary choice.

The published auxiliary-only loss and its source-style pooled-information
tuning convention are independently implemented. No reference script runs.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog, minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits
from public_conjoint_optimizer_audit import load_data, Logistic, converged, save
from public_conjoint_matched_audit import nuisance


class WeightedLogit:
    def __init__(self,x,weights,moment):
        self.x=x;self.weights=weights;self.moment=moment
        assert np.min(weights)>=0 and abs(weights.sum()-1)<1e-10
    def value(self,b):
        u=self.x@b
        return float(self.weights@np.logaddexp(0,u)-self.moment@b),self.x.T@(self.weights*expit(u))-self.moment
    def hessian(self,b):
        p=expit(self.x@b)
        return self.x.T@((self.weights*p*(1-p))[:,None]*self.x)


def objective(x,y,g,m,lam):
    n=len(y);weights=np.r_[np.full(m,(1-lam)/m),np.full(n-m,lam/(n-m))]
    moment=x[:m].T@(y[:m]-lam*g[:m])/m+lam*x[m:].T@g[m:]/(n-m)
    return WeightedLogit(x,weights,moment)


def support(loss):
    keep=loss.weights>0;x=loss.x[keep];w=loss.weights[keep];n,d=x.shape
    if np.linalg.matrix_rank(x)<d:return {'status':'rank_deficient'}
    scale=np.maximum(abs(x).max(0),1e-12);features=w[:,None]*x/scale
    eye=sparse.eye(n,format='csr');one=sparse.csr_matrix(np.ones((n,1)))
    fit=linprog(np.r_[np.zeros(n),-1.],
        A_ub=sparse.vstack([sparse.hstack([-eye,one]),sparse.hstack([eye,one])],format='csr'),
        b_ub=np.r_[np.zeros(n),np.ones(n)],
        A_eq=sparse.hstack([sparse.csr_matrix(features.T),sparse.csr_matrix((d,1))],format='csr'),
        b_eq=loss.moment/scale,bounds=[(0.,1.)]*n+[(0.,.5)],method='highs')
    if fit.status==2:
        # Independent separating-ray LP, retaining the actual weighted support.
        aub=sparse.vstack([sparse.hstack([sparse.csr_matrix(x),sparse.csr_matrix(-x),-eye]),
              sparse.csr_matrix(np.r_[np.ones(2*d),np.zeros(n)][None,:])],format='csr')
        ray=linprog(np.r_[-loss.moment,loss.moment,w],A_ub=aub,
                    b_ub=np.r_[np.zeros(n),1.],bounds=(0.,None),method='highs')
        if not ray.success:return {'status':'support_check_unresolved'}
        direction=ray.x[:d]-ray.x[d:2*d]
        gap=float(loss.moment@direction-w@np.maximum(x@direction,0))
        if gap<=1e-8:return {'status':'support_check_unresolved','separation_gap':gap}
        large=1e6;rayloss=loss.value(large*direction)[0]
        assert -1e-8<=rayloss+large*gap<=np.log(2)+1e-7
        return {'status':'outside_mean_support','separation_gap':gap,
                'ray_objective_1e6':rayloss,'direction':direction.tolist()}
    if not fit.success:return {'status':'support_check_unresolved'}
    residual=float(max(abs(features.T@fit.x[:-1]-loss.moment/scale)))
    if residual>1e-7:return {'status':'support_check_unresolved','equality_residual':residual}
    margin=float(fit.x[-1])
    return {'status':'interior' if margin>1e-8 else 'boundary_or_near_boundary',
            'interior_margin':margin,'equality_residual':residual}


def solve(loss,initial):
    check=support(loss)
    if check['status']!='interior':return None,check
    fit=minimize(loss.value,initial,jac=True,method='BFGS',options={'gtol':1e-9,'maxiter':1800})
    b=fit.x.copy()
    for _ in range(8):
        val,g=loss.value(b)
        if max(abs(g))<1e-9:break
        try:stepdir=np.linalg.solve(loss.hessian(b),g)
        except np.linalg.LinAlgError:break
        step=1.
        while step>1e-8 and loss.value(b-step*stepdir)[0]>val+1e-12:step/=2
        if step<=1e-8:break
        b-=step*stepdir
    _,score=loss.value(b);eig=np.linalg.eigvalsh(loss.hessian(b))
    check.update(score_max=float(max(abs(score))),information_min=float(min(eig)))
    if not np.isfinite(b).all() or max(abs(score))>1e-7 or min(eig)<1e-11:
        check['status']='fit_unresolved';return None,check
    check['status']='finite_root';return b,check


def power_weight(x,y,g,m,b):
    n=len(y);mu=expit(x@b)
    information=x.T@((mu*(1-mu))[:,None]*x)/n
    inverse=np.linalg.inv(information)
    true=x[:m]*(mu[:m]-y[:m])[:,None]
    pred=x*(mu-g)[:,None]
    aa=true-true.mean(0);bb=pred[:m]-pred[:m].mean(0)
    cross=(aa.T@bb+bb.T@aa)/m
    predvar=np.cov(pred.T)
    numerator=float(np.trace(inverse@cross@inverse))
    denominator=float(2*(1+m/(n-m))*np.trace(inverse@predvar@inverse))
    if not np.isfinite(denominator) or denominator<=1e-14:
        return None,{'tuning_status':'weight_unresolved','denominator':denominator}
    raw=numerator/denominator
    return float(np.clip(raw,0.,1.)),{'tuning_status':'weight_computed','raw_weight':raw,'denominator':denominator}


def summarize(rows,out):
    frame=pd.DataFrame(rows);frame.to_csv(out/'power_trial_diagnostics.csv',index=False)
    frame.groupby(['label_rows','prediction','method','status'],as_index=False).size().to_csv(out/'power_status_counts.csv',index=False)
    frame.groupby(['label_rows','prediction','method'],as_index=False).agg(
        allocations=('trial','size'),finite_estimates=('parameter_error','count'),
        error=('parameter_error','mean'),weight=('weight','mean'),score=('score_max','mean'),
        weight_zero=('weight',lambda s:int(s.eq(0).sum())),weight_one=('weight',lambda s:int(s.eq(1).sum()))
        ).to_csv(out/'power_point_summary.csv',index=False)
    paired=[]
    for (m,pred),part in frame.groupby(['label_rows','prediction']):
        pivot=part.pivot(index='trial',columns='method',values='parameter_error')
        for method in ['ppi_lambda1','power_human_init','power_ppi_init']:
            diff=(pivot[method]-pivot.human_converged).dropna()
            paired.append({'label_rows':m,'prediction':pred,'method':method,'allocations':len(pivot),
                'shared_finite':len(diff),'mean_error_difference':diff.mean(),
                'Monte_Carlo_SE':diff.std(ddof=1)/np.sqrt(len(diff)),
                'scope':'conditional on shared finite fits; full failures are separately retained'})
    pd.DataFrame(paired).to_csv(out/'power_paired_comparisons.csv',index=False)


def numeric_checks(x,y,z):
    x=x[:1200];y=y[:1200];m=100;g,_=nuisance(x,y,z,m,True)
    initial,_=converged(Logistic(x[:m],y[:m]));b=np.linspace(-.2,.3,x.shape[1]);step=1e-5
    graderr=hesserr=valueerr=0.
    for lam in [0.,.3,1.]:
        loss=objective(x,y,g,m,lam);val,grad=loss.value(b)
        direct=Logistic(x[:m],y[:m]).value(b)[0]+lam*(Logistic(x[m:],g[m:]).value(b)[0]-Logistic(x[:m],g[:m]).value(b)[0])
        valueerr=max(valueerr,abs(val-direct))
        eye=np.eye(x.shape[1]);fd=np.array([(loss.value(b+e*step)[0]-loss.value(b-e*step)[0])/(2*step) for e in eye])
        fd2=np.column_stack([(loss.value(b+e*step)[1]-loss.value(b-e*step)[1])/(2*step) for e in eye])
        graderr=max(graderr,float(max(abs(fd-grad))));hesserr=max(hesserr,float(abs(fd2-loss.hessian(b)).max()))
    zero,c=solve(objective(x,y,g,m,0.),initial)
    assert c['status']=='finite_root' and max(abs(zero-initial))<1e-6
    # A one-dimensional intercept checks analytic root and genuine exterior support.
    one=np.ones((4,1));inside=WeightedLogit(one,np.ones(4)/4,np.array([.3]))
    root,c=solve(inside,np.zeros(1));assert abs(root[0]-np.log(.3/.7))<1e-7
    exterior=support(WeightedLogit(one,np.ones(4)/4,np.array([1.2])))
    assert exterior['status']=='outside_mean_support'
    assert max(valueerr,graderr,hesserr)<1e-7
    return {'maximum_objective_identity_error':valueerr,'gradient_finite_difference_error':graderr,
            'Hessian_finite_difference_error':hesserr,'lambda0_matches_human':True,
            'analytic_intercept_root_and_exterior_ray':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--allocations',type=int,default=500);p.add_argument('--resume',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    x,y,z,_,_=load_data(a.input_dir);checks=numeric_checks(x,y,z)
    target,_=converged(Logistic(x[:1200],y[:1200]));x=x[:1200];y=y[:1200];rng=np.random.RandomState(20261109)
    script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();rows=[];failures=[];start=0
    checkpoint=a.out/'power_numeric_checks.json'
    if a.resume and checkpoint.exists():
        old=json.loads(checkpoint.read_text());assert old['script_sha256']==script_hash and old['requested_allocations']==a.allocations
        start=old['completed_allocations'];rows=pd.read_csv(a.out/'power_trial_diagnostics.csv').to_dict('records')
        failures=json.loads((a.out/'power_nonfinite_cases.json').read_text())
    for trial in range(a.allocations):
        order=(rng.permutation(240)[:,None]*5+np.arange(5)[None]).reshape(-1)
        if trial<start:continue
        xx=x[order];yy=y[order];zz=z[order]
        for m in [50,100,150,200]:
            human,_=converged(Logistic(xx[:m],yy[:m]))
            for pred,use_ai in [('AI',True),('X_only',False)]:
                g,fallback=nuisance(xx,yy,zz,m,use_ai);assert fallback==0
                fixed,fixedcheck=solve(objective(xx,yy,g,m,1.),human)
                cases=[('human_converged',human,{'status':'finite_root','score_max':max(abs(Logistic(xx[:m],yy[:m]).value(human)[1]))},0.),
                       ('ppi_lambda1',fixed,fixedcheck,1.)]
                for method,anchor in [('power_human_init',human),('power_ppi_init',fixed)]:
                    if anchor is None:
                        cases.append((method,None,{'status':'initialization_unavailable'},None));continue
                    lam,tune=power_weight(xx,yy,g,m,anchor)
                    if lam is None:
                        cases.append((method,None,{'status':'weight_unresolved',**tune},None));continue
                    beta,check=solve(objective(xx,yy,g,m,lam),human)
                    cases.append((method,beta,{**check,**tune},lam))
                for method,beta,check,lam in cases:
                    row={'trial':trial,'label_rows':m,'prediction':pred,'method':method,'status':check['status'],'weight':lam,
                         'raw_weight':check.get('raw_weight'),'score_max':check.get('score_max')}
                    if beta is not None:row['parameter_error']=float(100*np.mean(abs(beta-target)/(1+abs(target))))
                    else:failures.append({'trial':trial,'label_rows':m,'prediction':pred,'method':method,**check})
                    rows.append(row)
        if (trial+1)%10==0 or trial+1==a.allocations:
            summarize(rows,a.out);save(a.out/'power_nonfinite_cases.json',failures)
            save(checkpoint,{**checks,'completed_allocations':trial+1,'requested_allocations':a.allocations,'script_sha256':script_hash,
                 'independent_novelty_proven':False,'interval_coverage_tested':False,'downloaded_reference_code_executed':False})
            print('power allocations complete',trial+1,'of',a.allocations,flush=True)
    print('Completed all planned power-tuning allocations.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
