"""Prespecified stronger-utility companion to the prediction-bias validation."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from threadpoolctl import threadpool_limits
from public_conjoint_optimizer_audit import load_data,Logistic,converged,save
from power_tuned_choice_audit import WeightedLogit,objective,solve,power_weight
from finite_pool_choice_validation import interior_fit,covariance


def estimators(x,y,g,m,human,human_check,ridge_initial):
    cases=[('human_MLE',human,human_check,0.,0.)]
    unit_loss=objective(x,y,g,m,1.);unit,unit_check=solve(unit_loss,ridge_initial)
    cases.append(('PPI_unit',unit,unit_check,1.,0.));ridge_weight=None
    for label,anchor in [('PPI_power_human',human),('PPI_power_ridge',ridge_initial)]:
        if anchor is None:cases.append((label,None,{'status':'initialization_unavailable'},None,0.));continue
        lam,_=power_weight(x,y,g,m,anchor)
        if lam is None:cases.append((label,None,{'status':'weight_unresolved'},None,0.));continue
        beta,check=solve(objective(x,y,g,m,lam),ridge_initial);cases.append((label,beta,check,lam,0.))
        if label=='PPI_power_ridge':ridge_weight=lam
    if ridge_weight is not None:
        loss=objective(x,y,g,m,ridge_weight)
        try:
            step=ridge_initial-np.linalg.solve(loss.hessian(ridge_initial),loss.value(ridge_initial)[1])
            if not np.isfinite(step).all():raise np.linalg.LinAlgError()
            cases.append(('PPI_one_step_ridge',step,{'status':'finite_one_step','score_max':float(max(abs(loss.value(step)[1])))},ridge_weight,0.))
        except np.linalg.LinAlgError:cases.append(('PPI_one_step_ridge',None,{'status':'one_step_unresolved'},ridge_weight,0.))
    else:cases.append(('PPI_one_step_ridge',None,{'status':'weight_unresolved'},None,0.))
    penalized,check=interior_fit(unit_loss,ridge_initial,rho=1/m);cases.append(('PPI_ridge_unit',penalized,check,1.,1/m))
    assert np.linalg.matrix_rank(x[m:])==x.shape[1] and g[m:].min()>0 and g[m:].max()<1
    soft=WeightedLogit(x,np.full(len(y),1/len(y)),x.T@np.r_[y[:m],g[m:]]/len(y))
    pooled,check=interior_fit(soft,ridge_initial);cases.append(('soft_frozen_pool',pooled,check,None,0.))
    assert len(cases)==7;return cases


def summarize(rows,out):
    frame=pd.DataFrame(rows);frame.to_csv(out/'strong_signal_pool_trial_diagnostics.csv',index=False)
    keys=['prediction','label_rows','auxiliary_rows','method']
    frame.groupby(keys+['status'],as_index=False).size().to_csv(out/'strong_signal_pool_status_counts.csv',index=False)
    summary=frame.groupby(keys,as_index=False).agg(replications=('replication','size'),estimates=('estimate_available','sum'),intervals=('interval_available','sum'),
        MSE=('MSE','mean'),median_MSE=('MSE','median'),q90_MSE=('MSE',lambda s:s.quantile(.9)),
        excess_logloss=('excess_logloss','mean'),median_excess_logloss=('excess_logloss','median'),
        conditional_marginal_coverage=('marginal_coverage','mean'),interval_and_marginal_coverage=('issued_and_marginal_coverage','mean'),
        conditional_first_coverage=('first_coverage','mean'),interval_and_first_coverage=('issued_and_first_coverage','mean'),
        mean_interval_width=('mean_interval_width','mean'),weight=('weight','mean'),
        marginal_coverage_MCSE=('issued_and_marginal_coverage',lambda s:s.std(ddof=1)/np.sqrt(len(s))))
    summary.to_csv(out/'strong_signal_pool_summary.csv',index=False)
    frame.groupby(keys,as_index=False).agg(**{'coordinate'+str(j)+'_conditional_coverage':('cover_'+str(j),'mean') for j in range(11)}).to_csv(out/'strong_signal_pool_coordinate_coverage.csv',index=False)
    pairs=[]
    for condition,part in frame.groupby(keys[:-1]):
        pivot=part.pivot(index='replication',columns='method',values='excess_logloss')
        for method in pivot.columns:
            d=(pivot[method]-pivot.human_MLE).dropna();pairs.append(dict(zip(keys[:-1],condition))|{'method':method,'shared_estimates':len(d),'mean_loss_difference':d.mean(),'Monte_Carlo_SE':d.std(ddof=1)/np.sqrt(len(d)),'scope':'conditional on shared estimates; availability separately reported'})
    pd.DataFrame(pairs).to_csv(out/'strong_signal_pool_paired_comparisons.csv',index=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--replications',type=int,default=500);p.add_argument('--resume',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    source,observed,_,_,_=load_data(a.input_dir);base=source[:1200];truth,_=converged(Logistic(base,observed[:1200]));truth=3*truth
    sourcehash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();helperhash=hashlib.sha256((Path(__file__).parent/'finite_pool_choice_validation.py').read_bytes()).hexdigest()
    checkpoint=a.out/'strong_signal_pool_numeric_checks.json';rows=[];failures=[];start=0;maxroot=0.;maxpenalty=0.
    if a.resume and checkpoint.exists():
        previous=json.loads(checkpoint.read_text());assert previous['source_sha256']==sourcehash and previous['helper_sha256']==helperhash and previous['requested_replications']==a.replications
        start=previous['completed_replications'];maxroot=previous['maximum_root_score'];maxpenalty=previous['maximum_penalized_score']
        rows=pd.read_csv(a.out/'strong_signal_pool_trial_diagnostics.csv').to_dict('records');failures=json.loads((a.out/'strong_signal_pool_unavailable_cases.json').read_text())
    save(a.out/'strong_signal_pool_DGP.json',{'truth':truth,'truth_scale':3,'support_rows':len(base),'dimensions':base.shape[1],'source':'same pinned vaccine support as the earlier finite-pool validation','truth_role':'simulation parameter, not structural truth of observed choices','sampling':'IID choice rows with replacement; generated Y,Z; no new respondent or LLM responses','prediction_formula':'expit(a*correct conditional log odds+b)'})
    rng=np.random.default_rng(20261121);critical=norm.ppf(.975);prob_base=expit(base@truth)
    for rep in range(a.replications):
        primary=base[rng.integers(0,len(base),150)];aux=base[rng.integers(0,len(base),600)]
        yp=(rng.random(150)<expit(primary@truth)).astype(float);yu=(rng.random(600)<expit(aux@truth)).astype(float)
        zp=np.logical_xor(yp,rng.random(150)<.25).astype(float);zu=np.logical_xor(yu,rng.random(600)<.25).astype(float)
        if rep<start:continue
        for m in [50,150]:
            human_loss=WeightedLogit(primary[:m],np.full(m,1/m),primary[:m].T@yp[:m]/m)
            ridge_initial,_=interior_fit(human_loss,np.zeros(base.shape[1]),rho=1/m);assert ridge_initial is not None
            human,human_check=solve(human_loss,ridge_initial)
            x=np.vstack([primary[:m],aux]);y=np.r_[yp[:m],yu];z=np.r_[zp[:m],zu];logodds=x@truth+(2*z-1)*np.log(3.)
            for prediction,slope,bias in [('oracle',1.,0.),('overconfident',2.,0.),('intercept_bias',1.,1.)]:
                g=expit(slope*logodds+bias)
                for method,beta,check,lam,rho in estimators(x,y,g,m,human,human_check,ridge_initial):
                    row={'replication':rep,'prediction':prediction,'label_rows':m,'auxiliary_rows':600,'method':method,'status':check['status'],
                        'estimate_available':beta is not None,'interval_available':False,'weight':lam,'score_max':check.get('score_max'),
                        'issued_and_marginal_coverage':0.,'issued_and_first_coverage':0.}
                    if beta is not None:
                        error=beta-truth;row.update(MSE=float(np.mean(error**2)),excess_logloss=float(np.mean(np.logaddexp(0,base@beta)-np.logaddexp(0,base@truth)-prob_base*(base@error))))
                        cov=covariance(x,y,g,m,beta,lam,'soft_oracle_pool' if method=='soft_frozen_pool' else method,rho)
                        if cov is not None:
                            se=np.sqrt(np.diag(cov));covered=abs(error)<=critical*se
                            row.update(interval_available=True,marginal_coverage=float(covered.mean()),first_coverage=float(covered[0]),issued_and_marginal_coverage=float(covered.mean()),issued_and_first_coverage=float(covered[0]),mean_interval_width=float(2*critical*se.mean()),**{'cover_'+str(j):float(c) for j,c in enumerate(covered)})
                        else:failures.append({'replication':rep,'prediction':prediction,'label_rows':m,'method':method,'status':'interval_unavailable'})
                    else:failures.append({'replication':rep,'prediction':prediction,'label_rows':m,'method':method,**check})
                    if check.get('score_max') is not None:
                        if check['status']=='finite_root':maxroot=max(maxroot,check['score_max'])
                        if check['status']=='finite_penalized_root':maxpenalty=max(maxpenalty,check['score_max'])
                    rows.append(row)
        if (rep+1)%10==0 or rep+1==a.replications:
            summarize(rows,a.out);save(a.out/'strong_signal_pool_unavailable_cases.json',failures)
            save(checkpoint,{'completed_replications':rep+1,'requested_replications':a.replications,'source_sha256':sourcehash,'helper_sha256':helperhash,'conditions':6,'methods':7,'maximum_root_score':maxroot,'maximum_penalized_score':maxpenalty,'new_human_or_LLM_responses':False,'independent_novelty_proven':False})
            print('Strong-signal finite-pool replications complete',rep+1,'of',a.replications,flush=True)
    print('All strong-signal finite-pool conditions complete.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
