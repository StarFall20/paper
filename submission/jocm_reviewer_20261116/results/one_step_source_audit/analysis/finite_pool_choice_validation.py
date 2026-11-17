"""Known-truth finite auxiliary-pool validation on a released menu support.

All outcomes and proxy labels in this file are simulated. No participant data
are created or fielded. Established estimators are comparators, not innovations.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm
from threadpoolctl import threadpool_limits
from public_conjoint_optimizer_audit import load_data,Logistic,converged,save
from power_tuned_choice_audit import WeightedLogit,objective,solve,power_weight


class Penalized:
    def __init__(self,loss,rho):self.loss=loss;self.rho=rho
    def value(self,b):
        value,gradient=self.loss.value(b)
        return value+self.rho*np.dot(b,b)/2,gradient+self.rho*b
    def hessian(self,b):return self.loss.hessian(b)+self.rho*np.eye(len(b))


def interior_fit(loss,initial,rho=0.):
    criterion=Penalized(loss,rho) if rho else loss
    fit=minimize(criterion.value,initial,jac=True,method='BFGS',options={'gtol':1e-9,'maxiter':1800})
    beta=fit.x.copy()
    for _ in range(8):
        val,gradient=criterion.value(beta)
        if max(abs(gradient))<1e-9:break
        try:direction=np.linalg.solve(criterion.hessian(beta),gradient)
        except np.linalg.LinAlgError:break
        step=1.
        while step>1e-8 and criterion.value(beta-step*direction)[0]>val+1e-12:step/=2
        if step<=1e-8:break
        beta-=step*direction
    _,gradient=criterion.value(beta);eig=np.linalg.eigvalsh(criterion.hessian(beta))
    check={'score_max':float(max(abs(gradient))),'information_min':float(min(eig))}
    if not np.isfinite(beta).all() or max(abs(gradient))>1e-7 or min(eig)<1e-11:
        return None,{'status':'fit_unresolved',**check}
    return beta,{'status':'finite_penalized_root' if rho else 'finite_root',**check}


def covariance(x,y,g,m,b,lam,method,rho):
    n=len(y);N=n-m;mu=expit(x@b)
    if method=='soft_oracle_pool':
        primary=x[:m]*(mu[:m]-y[:m])[:,None]
        auxiliary=x[m:]*(mu[m:]-g[m:])[:,None]
        meat=m/n**2*np.atleast_2d(np.cov(primary.T))+N/n**2*np.atleast_2d(np.cov(auxiliary.T))
        information=x.T@((mu*(1-mu))[:,None]*x)/n
    else:
        primary=x[:m]*((1-lam)*mu[:m]-y[:m]+lam*g[:m])[:,None]
        auxiliary=x[m:]*(mu[m:]-g[m:])[:,None]
        meat=np.atleast_2d(np.cov(primary.T))/m+lam**2*np.atleast_2d(np.cov(auxiliary.T))/N
        information=objective(x,y,g,m,lam).hessian(b)+rho*np.eye(x.shape[1])
    try:
        if np.linalg.eigvalsh(information).min()<1e-11:return None
        inverse=np.linalg.inv(information);result=inverse@meat@inverse
        if not np.isfinite(result).all() or np.min(np.diag(result))<=0:return None
        return result
    except np.linalg.LinAlgError:return None


def summarize(rows,out):
    frame=pd.DataFrame(rows);frame.to_csv(out/'finite_pool_trial_diagnostics.csv',index=False)
    keys=['label_rows','auxiliary_rows','epsilon','method']
    frame.groupby(keys+['status'],as_index=False).size().to_csv(out/'finite_pool_status_counts.csv',index=False)
    summary=frame.groupby(keys,as_index=False).agg(
        replications=('replication','size'),estimates=('estimate_available','sum'),intervals=('interval_available','sum'),
        MSE=('MSE','mean'),stabilized_parameter_error=('stabilized_parameter_error','mean'),
        excess_logloss=('excess_logloss','mean'),conditional_marginal_coverage=('marginal_coverage','mean'),
        conditional_first_coverage=('first_coverage','mean'),
        interval_and_marginal_coverage=('issued_and_marginal_coverage','mean'),
        interval_and_first_coverage=('issued_and_first_coverage','mean'),
        mean_interval_width=('mean_interval_width','mean'),weight=('weight','mean'),
        marginal_coverage_MCSE=('issued_and_marginal_coverage',lambda s:s.std(ddof=1)/np.sqrt(len(s))))
    summary.to_csv(out/'finite_pool_summary.csv',index=False)
    coords=frame.groupby(keys,as_index=False).agg(**{
        'coordinate'+str(j)+'_conditional_coverage':('cover_'+str(j),'mean') for j in range(11)})
    coords.to_csv(out/'finite_pool_coordinate_coverage.csv',index=False)
    paired=[]
    for condition,part in frame.groupby(keys[:-1]):
        pivot=part.pivot(index='replication',columns='method',values='excess_logloss')
        for method in pivot.columns:
            difference=(pivot[method]-pivot.human_MLE).dropna()
            paired.append(dict(zip(keys[:-1],condition))|{'method':method,'shared_finite':len(difference),
                'mean_excess_logloss_difference':difference.mean(),
                'Monte_Carlo_SE':difference.std(ddof=1)/np.sqrt(len(difference)),
                'scope':'conditional on shared estimates; availability and interval issuance reported separately'})
    pd.DataFrame(paired).to_csv(out/'finite_pool_paired_comparisons.csv',index=False)


def calibration_checks(x,truth):
    grid=x[:21];prob=expit(grid@truth);maximum=0.
    for e in [.1,.25]:
        z1=e+(1-2*e)*prob
        g1=expit(grid@truth+np.log((1-e)/e));g0=expit(grid@truth-np.log((1-e)/e))
        maximum=max(maximum,float(abs(z1*g1+(1-z1)*g0-prob).max()))
    assert maximum<1e-12
    return {'oracle_conditional_mean_max_error':maximum,'ridge_penalty':'rho=1/m on every coordinate',
            'confidence_scope':'diagnostic Wald intervals; no finite-sample guarantee'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--replications',type=int,default=500);p.add_argument('--resume',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    source,y_source,_,_,_=load_data(a.input_dir);base=source[:1200]
    truth,_=converged(Logistic(base,y_source[:1200]));checks=calibration_checks(base,truth)
    save(a.out/'finite_pool_DGP.json',{'support_rows':len(base),'dimensions':base.shape[1],'truth':truth,
        'truth_role':'Frozen simulation parameter; not claimed structural truth of observed choices',
        'min_probability':expit(base@truth).min(),'max_probability':expit(base@truth).max(),
        'sampling':'independent IID choice rows with replacement, labels and proxies simulated',**checks})
    script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();rows=[];failures=[];start=0
    checkpoint=a.out/'finite_pool_numeric_checks.json'
    if a.resume and checkpoint.exists():
        previous=json.loads(checkpoint.read_text());assert previous['script_sha256']==script_hash and previous['requested_replications']==a.replications
        start=previous['completed_replications'];rows=pd.read_csv(a.out/'finite_pool_trial_diagnostics.csv').to_dict('records')
        failures=json.loads((a.out/'finite_pool_unavailable_cases.json').read_text())
    rng=np.random.default_rng(20261112);critical=norm.ppf(.975);max_penalty_score=0.;max_root_score=0.
    for rep in range(a.replications):
        primary=base[rng.integers(0,len(base),150)];auxiliary=base[rng.integers(0,len(base),600)]
        p_primary=expit(primary@truth);p_aux=expit(auxiliary@truth)
        y_primary=(rng.random(150)<p_primary).astype(float);y_aux=(rng.random(600)<p_aux).astype(float)
        flip_primary=rng.random(150);flip_aux=rng.random(600)
        if rep<start:continue
        for m in [20,50,150]:
            # Human estimates are shared across all epsilon/auxiliary conditions.
            human_loss=WeightedLogit(primary[:m],np.full(m,1/m),primary[:m].T@y_primary[:m]/m)
            ridge_initial,initial_check=interior_fit(human_loss,np.zeros(base.shape[1]),rho=1/m)
            assert ridge_initial is not None
            human,human_check=solve(human_loss,ridge_initial)
            for N in [150,600]:
                x=np.vstack([primary[:m],auxiliary[:N]]);y=np.r_[y_primary[:m],y_aux[:N]]
                for e in [.1,.25]:
                    z=np.r_[np.logical_xor(y_primary[:m],flip_primary[:m]<e),np.logical_xor(y_aux[:N],flip_aux[:N]<e)].astype(float)
                    g=expit(x@truth+(2*z-1)*np.log((1-e)/e))
                    cases=[('human_MLE',human,human_check,0.,0.)]
                    unit_loss=objective(x,y,g,m,1.);unit,unit_check=solve(unit_loss,ridge_initial)
                    cases.append(('PPI_unit',unit,unit_check,1.,0.))
                    tuned_ridge_weight=None
                    for label,anchor in [('PPI_power_human',human),('PPI_power_ridge',ridge_initial)]:
                        if anchor is None:
                            cases.append((label,None,{'status':'initialization_unavailable'},None,0.));continue
                        lam,tuning=power_weight(x,y,g,m,anchor)
                        if lam is None:
                            cases.append((label,None,{'status':'weight_unresolved'},None,0.));continue
                        beta,check=solve(objective(x,y,g,m,lam),ridge_initial)
                        cases.append((label,beta,check,lam,0.))
                        if label=='PPI_power_ridge':tuned_ridge_weight=lam
                    if tuned_ridge_weight is not None:
                        loss=objective(x,y,g,m,tuned_ridge_weight)
                        try:
                            step=ridge_initial-np.linalg.solve(loss.hessian(ridge_initial),loss.value(ridge_initial)[1])
                            if not np.isfinite(step).all():raise np.linalg.LinAlgError()
                            cases.append(('PPI_one_step_ridge',step,{'status':'finite_one_step','score_max':float(max(abs(loss.value(step)[1])))},tuned_ridge_weight,0.))
                        except np.linalg.LinAlgError:cases.append(('PPI_one_step_ridge',None,{'status':'one_step_unresolved'},tuned_ridge_weight,0.))
                    else:cases.append(('PPI_one_step_ridge',None,{'status':'weight_unresolved'},None,0.))
                    penalized,check=interior_fit(unit_loss,ridge_initial,rho=1/m)
                    cases.append(('PPI_ridge_unit',penalized,check,1.,1/m))
                    assert np.linalg.matrix_rank(x[m:])==base.shape[1] and g[m:].min()>0 and g[m:].max()<1
                    soft=WeightedLogit(x,np.full(len(y),1/len(y)),x.T@np.r_[y[:m],g[m:]]/len(y))
                    pooled,check=interior_fit(soft,ridge_initial)
                    cases.append(('soft_oracle_pool',pooled,check,None,0.))
                    assert len(cases)==7
                    for method,beta,check,lam,rho in cases:
                        row={'replication':rep,'label_rows':m,'auxiliary_rows':N,'epsilon':e,'method':method,
                             'status':check['status'],'estimate_available':beta is not None,'interval_available':False,
                             'weight':lam,'score_max':check.get('score_max'),
                             'issued_and_marginal_coverage':0.,'issued_and_first_coverage':0.}
                        if beta is not None:
                            error=beta-truth;row.update(MSE=float(np.mean(error**2)),
                                stabilized_parameter_error=float(100*np.mean(abs(error)/(1+abs(truth)))),
                                excess_logloss=float(np.mean(np.logaddexp(0,base@beta)-np.logaddexp(0,base@truth)-expit(base@truth)*(base@error))))
                            cov=covariance(x,y,g,m,beta,lam,method,rho)
                            if cov is not None:
                                se=np.sqrt(np.diag(cov));covered=(abs(error)<=critical*se)
                                row.update(interval_available=True,marginal_coverage=float(covered.mean()),first_coverage=float(covered[0]),
                                    issued_and_marginal_coverage=float(covered.mean()),issued_and_first_coverage=float(covered[0]),
                                    mean_interval_width=float(2*critical*se.mean()),**{'cover_'+str(j):float(c) for j,c in enumerate(covered)})
                            else:failures.append({'replication':rep,'label_rows':m,'auxiliary_rows':N,'epsilon':e,'method':method,'status':'interval_unavailable'})
                        else:failures.append({'replication':rep,'label_rows':m,'auxiliary_rows':N,'epsilon':e,'method':method,**check})
                        if check.get('score_max') is not None:
                            if check['status']=='finite_penalized_root':max_penalty_score=max(max_penalty_score,check['score_max'])
                            if check['status']=='finite_root':max_root_score=max(max_root_score,check['score_max'])
                        rows.append(row)
        if (rep+1)%10==0 or rep+1==a.replications:
            summarize(rows,a.out);save(a.out/'finite_pool_unavailable_cases.json',failures)
            save(checkpoint,{**checks,'completed_replications':rep+1,'requested_replications':a.replications,
                'script_sha256':script_hash,'conditions':12,'methods':7,
                'maximum_penalized_score':max_penalty_score,'maximum_root_score':max_root_score,
                'new_human_or_AI_responses':False,'independent_novelty_proven':False})
            print('finite-pool replications complete',rep+1,'of',a.replications,flush=True)
    print('All planned finite-pool validation cells completed.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
