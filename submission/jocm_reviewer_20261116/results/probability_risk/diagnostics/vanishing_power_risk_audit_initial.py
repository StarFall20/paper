"""Retained probability-loss paths for the existing compensated logit objective.

No new estimator is introduced. Support states precede numerical fitting;
every planned case is retained. The finite-design lower bound allows both
gradient error and negative mass in an approximately separating direction.
"""
from pathlib import Path
import argparse,ast,hashlib,json
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize,linprog
from scipy.special import expit
from threadpoolctl import threadpool_limits
from public_conjoint_optimizer_audit import load_data,save
from public_conjoint_matched_audit import nuisance
from observable_low_label_path import ridge_anchor
from vanishing_power_choice import rescaled_support

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def source_closure():
    base=Path(__file__).resolve().parent;seen={}
    def visit(p):
        if p.name in seen:return
        seen[p.name]=sha(p)
        for n in ast.walk(ast.parse(p.read_text())):
            names=([n.module.split('.')[0]] if isinstance(n,ast.ImportFrom) and n.module else [a.name.split('.')[0] for a in n.names] if isinstance(n,ast.Import) else [])
            for name in names:
                f=base/(name+'.py')
                if f.is_file():visit(f)
    visit(Path(__file__).resolve());return seen

class StableLoss:
    def __init__(self,x,y,g,m,power):
        self.x=x;self.y=y;self.g=g;self.m=m;self.power=power
        self.sign=2*y[:m]-1
    def value(self,b):
        m=self.m;p=self.power;x=self.x;g=self.g;u=x@b
        h=np.mean(np.logaddexp(0,-self.sign*u[:m]))
        soft=g*np.logaddexp(0,-u)+(1-g)*np.logaddexp(0,u)
        d=np.mean(soft[m:])-np.mean(soft[:m])
        human=-(x[:m].T@(self.sign*expit(-self.sign*u[:m])))/m
        prediction=x[m:].T@(expit(u[m:])-g[m:])/len(u[m:])-x[:m].T@(expit(u[:m])-g[:m])/m
        return float(h+p*d),human+p*prediction
    def hessian(self,b):
        u=self.x@b;d=expit(u)*expit(-u);m=self.m;p=self.power
        return (1-p)*(self.x[:m].T@(d[:m,None]*self.x[:m]))/m+p*(self.x[m:].T@(d[m:,None]*self.x[m:]))/(len(u)-m)

def numerical_fit(loss,initial):
    b=initial.copy();target=1e-7*loss.power;flags=[];iterations=0
    def refine(b,maxiter):
        nonlocal iterations
        for _ in range(maxiter):
            iterations+=1;value,g=loss.value(b)
            if np.linalg.norm(g)<=target:return b
            h=loss.hessian(b)
            try:step=np.linalg.solve(h,g)
            except np.linalg.LinAlgError:
                flags.append('singular_Newton_matrix');return b
            if not np.isfinite(step).all() or g@step<=0:
                flags.append('invalid_Newton_direction');return b
            rate=1.;accepted=False
            while rate>=2**-35:
                trial=b-rate*step;new=loss.value(trial)[0]
                if new<=value-1e-4*rate*(g@step)+1e-14:
                    b=trial;accepted=True;break
                rate/=2
            if not accepted:
                flags.append('Newton_line_search_stopped');return b
        flags.append('Newton_iteration_limit');return b
    b=refine(b,160)
    if np.linalg.norm(loss.value(b)[1])>target:
        flags.append('BFGS_refinement_used')
        r=minimize(loss.value,b,jac=True,method='BFGS',options=dict(gtol=target/np.sqrt(len(b)),maxiter=600))
        b=refine(r.x.copy(),80)
    value,gradient=loss.value(b);eig=np.linalg.eigvalsh(loss.hessian(b))
    okay=bool(np.isfinite(b).all() and np.linalg.norm(gradient)<=target and eig[0]>0)
    return b,dict(fit_status='numerical_finite_root' if okay else 'numerical_unresolved',gradient_l2=float(np.linalg.norm(gradient)),gradient_over_power=float(np.linalg.norm(gradient)/loss.power),hessian_min=float(eig[0]),objective=value,iterations=iterations,refinement_flags=';'.join(flags))

def direction_bound(x,y,m,v,power,b,residual):
    if v is None:return dict(coefficient_lower_bound=0.,strict_lower_bound=0.,separator_positive_mean=0.,separator_negative_mean=0.)
    v=np.asarray(v,float);v=v/np.linalg.norm(v)
    z=(2*y[:m]-1)*(x[:m]@v);positive=np.maximum(z,0);negative=np.maximum(-z,0)
    s=float(positive.mean());t=float(negative.mean());nu=float(np.sqrt(np.mean(positive**2)))
    cl=float(np.linalg.norm(x[:m],2)/np.sqrt(m))
    dv=float(np.mean(abs(x[:m]@v))+np.mean(abs(x[m:]@v)))
    bound=0. if s==0 else s/(nu*cl)*max(0.,np.log(s/(2*(power*dv+residual+t))))
    gamma=float(z.min());strict=0. if gamma<=0 else max(0.,np.log(gamma/(2*(power*dv+residual))))/cl
    assert max(bound,strict)<=np.linalg.norm(b)+1e-6
    return dict(coefficient_lower_bound=float(bound),strict_lower_bound=float(strict),separator_positive_mean=s,separator_negative_mean=t,separator_minimum_margin=gamma,separator_rms_positive=nu,label_design_spectral_scale=cl,directional_prediction_bound=dv)

def gaussian_risk(b):
    intercept=float(b[0]);scale=float(np.linalg.norm(b[1:]));const=1/np.sqrt(2*np.pi)
    density=lambda z:const*np.exp(-z*z/2)
    logloss,error=quad(lambda z:float((np.logaddexp(0,intercept+scale*z)-(intercept+scale*z)/2)*density(z)),-12,12,epsabs=1e-10,epsrel=1e-10,limit=250)
    variance,error2=quad(lambda z:float(expit(intercept+scale*z)*expit(-intercept-scale*z)*density(z)),-12,12,epsabs=1e-11,epsrel=1e-10,limit=250)
    lower=np.linalg.norm(b)/np.sqrt(2*np.pi)-np.log(2)
    assert logloss-np.log(2)>=lower-1e-9
    return dict(test_log_loss=float(logloss),test_Brier_loss=float(.5-variance),test_classification_error=.5,test_integration_error=max(error,error2),null_excess_log_loss_lower_bound=float(lower))

def public_risk(b,x,y):
    u=x@b
    return dict(test_log_loss=float(np.mean(np.logaddexp(0,-(2*y-1)*u))),test_Brier_loss=float(np.mean((expit(u)-y)**2)),test_classification_error=float(np.mean((u>=0)!=y)))

def strict_direction(x,y):
    signed=x*(2*y-1)[:,None];p=x.shape[1]
    # Feasibility supplies a separator; minimum-norm refinement is optional.
    r=linprog(np.zeros(p),A_ub=-signed,b_ub=-np.ones(len(y)),bounds=[(None,None)]*p,method='highs')
    if not r.success:return None,dict(separator_status='strict_feasibility_failed',LP_status=int(r.status))
    start=r.x.copy()
    fit=minimize(lambda v:float(v@v/2),start,jac=lambda v:v,method='SLSQP',constraints=[dict(type='ineq',fun=lambda v:signed@v-1,jac=lambda v:signed)],options=dict(ftol=1e-10,maxiter=500))
    v=fit.x if np.min(signed@fit.x)>=1-1e-7 else start
    minimum=float(np.min(signed@v));assert minimum>=1-1e-7
    v=v/np.linalg.norm(v)
    return v,dict(separator_status='strict_feasible',minimum_unit_margin=float(np.min(signed@v)),norm_refinement_success=bool(fit.success),optimal_margin_claimed=False)

def tables(rows,out):
    frame=pd.DataFrame(rows);frame.to_csv(out/'path_records.csv',index=False)
    group=['mode','label_rows','prediction','power'];summary=[]
    for key,part in frame.groupby(group,dropna=False):
        finite=part[part.fit_status=='numerical_finite_root'];r=dict(zip(group,key));r.update(requested=len(part),finite=len(finite),unresolved=int((part.fit_status=='numerical_unresolved').sum()))
        for field in ['coefficient_norm','test_log_loss','test_Brier_loss','test_classification_error','coefficient_lower_bound','strict_lower_bound']:
            val=finite[field].dropna();r['mean_'+field]=float(val.mean()) if len(val) else None
            r['MCSE_'+field]=float(val.std(ddof=1)/np.sqrt(len(val))) if len(val)>1 and key[0]=='gaussian' else None
        summary.append(r)
    pd.DataFrame(summary).to_csv(out/'path_summary.csv',index=False)
    pairs=[]
    for key,part in frame.groupby(['mode','label_rows','prediction'],dropna=False):
        base=part[part.power==.5].set_index('case')
        for power,piece in part[part.power!=.5].groupby('power'):
            piece=piece.set_index('case');common=base.index.intersection(piece.index)
            good=[k for k in common if base.loc[k,'fit_status']=='numerical_finite_root' and piece.loc[k,'fit_status']=='numerical_finite_root']
            r=dict(zip(['mode','label_rows','prediction'],key));r.update(power=power,reference_power=.5,planned_common=len(common),shared_finite=len(good))
            for field in ['coefficient_norm','test_log_loss','test_Brier_loss']:
                differences=np.array([piece.loc[k,field]-base.loc[k,field] for k in good])
                r['mean_difference_'+field]=float(differences.mean()) if len(good) else None
                r['MCSE_difference_'+field]=float(differences.std(ddof=1)/np.sqrt(len(good))) if len(good)>1 and key[0]=='gaussian' else None
            pairs.append(r)
    pd.DataFrame(pairs).to_csv(out/'paired_to_half_power.csv',index=False)

def run(a):
    plan=json.loads(a.plan.read_text());a.out.mkdir(parents=True,exist_ok=True);statefile=a.out/'risk_path_status.json'
    closure=source_closure();rows=[];details=[];complete=set()
    if a.resume:
        status=json.loads(statefile.read_text());assert status['plan_sha256']==sha(a.plan) and status['source_hashes']==closure
        rows=pd.read_csv(a.out/'path_records.csv').to_dict('records');details=json.loads((a.out/'coefficient_records.json').read_text());complete=set(status['completed_cases'])
    else:assert not statefile.exists(),'Use a fresh output directory or --resume.'
    def checkpoint(state):
        if rows:tables(rows,a.out)
        save(a.out/'coefficient_records.json',details)
        save(statefile,dict(date='2026-11-12',state=state,mode=a.mode,completed_cases=sorted(complete),completed_model_cases=len(complete),retained_path_records=len(rows),plan_sha256=sha(a.plan),source_hashes=closure,new_human_or_AI_responses=0))
    checkpoint('running')
    if a.mode=='public':
        xall,yall,z,_,_=load_data(a.input_dir);prefix=plan['input_prefix_rows'];x=xall[:prefix];y=yall[:prefix];m=plan['label_rows'];rng=np.random.RandomState(plan['seed'])
        ordinary=json.loads(a.ordinary.read_text());cases=[]
        for trial in range(plan['allocations']):
            order=(rng.permutation(240)[:,None]*5+np.arange(5)[None]).reshape(-1)
            cases.append((f'public_{trial}',trial,x[order],y[order],z[order],m,ordinary[trial]))
        save(a.out/'input_scope.json',dict(prefix_rows=prefix,evaluation_rows=len(yall)-prefix,source_manifest_sha256=sha(a.input_dir/'source_manifest.json'),ordinary_directions_sha256=sha(a.ordinary),input_scope='Released rows and source-script block convention; no verified respondent IDs.'))
    else:
        rng=np.random.default_rng(plan['seed']);cases=[]
        for m in plan['label_rows']:
            for trial in range(plan['replications_per_label_budget']):
                n=m*(1+plan['auxiliary_ratio']);q=int(np.floor(plan['noise_slope_ratio']*m))
                xx=np.c_[np.ones(n),rng.normal(size=(n,q+1))];yy=rng.integers(0,2,n).astype(float)
                zz=np.logical_xor(yy,rng.random(n)<plan['proxy_flip_probability']).astype(float)
                cases.append((f'gaussian_{m}_{trial}',trial,xx,yy,zz,m,None))
    for key,trial,xx,yy,zz,m,human in cases:
        if key in complete:continue
        anchor,anchor_error=ridge_anchor(xx[:m],yy[:m]);ref_risk=public_risk(anchor,xall[prefix:],yall[prefix:]) if a.mode=='public' else gaussian_risk(anchor)
        if a.mode=='public':
            v=human.get('direction');sepmeta=dict(separator_status=human['status']);predictors=[]
            for name in plan['prediction_routines']:
                pred,fallback=nuisance(xx,yy,zz,m,name=='AI');predictors.append((name,pred,fallback))
        else:
            v,sepmeta=strict_direction(xx[:m],yy[:m]);e=plan['proxy_flip_probability'];predictors=[('oracle',e+(1-2*e)*zz,0)]
        detail=dict(case=key,label_rows=m,trial=trial,separator_direction=None if v is None else np.asarray(v).tolist(),separator_metadata=sepmeta,ridge_coefficient=anchor.tolist(),ridge_gradient_l2=anchor_error,ridge_risk=ref_risk,fits=[])
        for predname,pred,fallback in predictors:
            initial=anchor.copy()
            for power in plan['powers']:
                state=rescaled_support(xx,yy,pred,m,power);record=dict(mode=a.mode,case=key,trial=trial,label_rows=m,parameters=xx.shape[1],prediction=predname,power=power,support_status=state['status'],support_margin=state.get('margin'),support_residual=state.get('residual'),nuisance_fallbacks=fallback,fit_status='not_fitted_'+state['status'],ridge_test_log_loss=ref_risk['test_log_loss'],ridge_test_Brier_loss=ref_risk['test_Brier_loss'])
                fitdetail=dict(prediction=predname,power=power,support=state)
                if state['status']=='interior':
                    loss=StableLoss(xx,yy,pred,m,power);b,fit=numerical_fit(loss,initial);record.update(fit);fitdetail.update(fit,coefficient=b.tolist())
                    if fit['fit_status']=='numerical_finite_root':
                        initial=b.copy();record['coefficient_norm']=float(np.linalg.norm(b));record.update(direction_bound(xx,yy,m,v,power,b,fit['gradient_l2']))
                        record.update(public_risk(b,xall[prefix:],yall[prefix:]) if a.mode=='public' else gaussian_risk(b))
                rows.append(record);detail['fits'].append(fitdetail)
        details.append(detail);complete.add(key)
        if len(complete)%5==0:
            checkpoint('running');print(json.dumps(dict(mode=a.mode,completed_cases=len(complete),records=len(rows),numerical_unresolved=sum(r['fit_status']=='numerical_unresolved' for r in rows))),flush=True)
    assert len(rows)==plan['requested_corrected_records'];checkpoint('complete');print(json.dumps(dict(mode=a.mode,state='complete',records=len(rows))),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['public','gaussian'],required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--input-dir',type=Path);p.add_argument('--ordinary',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.mode=='public':assert a.input_dir is not None and a.ordinary is not None
    run(a)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
