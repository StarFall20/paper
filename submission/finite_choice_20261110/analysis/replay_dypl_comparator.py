"""Established DY-prior comparator on two unchanged displayed simulations.

Independent implementation of published pseudo-response estimator; no external
R code executes. Original seven estimators and their trial files stay unchanged.
Raw DY estimates and diagnostic sandwiches are not Gaussian HD corrections.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.stats import norm
from threadpoolctl import threadpool_limits
from public_conjoint_optimizer_audit import load_data,Logistic,converged,save
from power_tuned_choice_audit import WeightedLogit,solve
from finite_pool_choice_validation import interior_fit


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def dy_loss(x,y,alpha):
    target=alpha*y+(1-alpha)/2
    return WeightedLogit(x,np.full(len(y),1/len(y)),x.T@target/len(y))


def dy_covariance(x,y,alpha,beta):
    loss=dy_loss(x,y,alpha);mu=expit(x@beta)
    scores=x*(mu-alpha*y-(1-alpha)/2)[:,None]
    H=loss.hessian(beta)
    try:
        if np.linalg.eigvalsh(H).min()<1e-11:return None
        inverse=np.linalg.inv(H)
        covariance=inverse@np.atleast_2d(np.cov(scores.T,ddof=1))@inverse/len(y)
        if not np.isfinite(covariance).all() or min(np.diag(covariance))<=0:return None
        return covariance
    except np.linalg.LinAlgError:return None


def numeric_checks(out):
    x=np.array([[1.,0.],[0.,1.],[-1.,1.],[1.,-1.],[.5,.8]])
    y=np.array([0.,1.,1.,0.,1.]);b=np.array([.2,-.3]);step=1e-5
    gradient_error=0.;hessian_error=0.;root_error=0.;cov_error=0.;value_error=0.
    for alpha in [.5,.8]:
        loss=dy_loss(x,y,alpha);target=alpha*y+(1-alpha)/2
        value,grad=loss.value(b)
        direct=np.mean(target*np.logaddexp(0,-x@b)+(1-target)*np.logaddexp(0,x@b))
        value_error=max(value_error,abs(value-direct))
        eye=np.eye(2)
        fd=np.array([(loss.value(b+v*step)[0]-loss.value(b-v*step)[0])/(2*step) for v in eye])
        fh=np.column_stack([(loss.value(b+v*step)[1]-loss.value(b-v*step)[1])/(2*step) for v in eye])
        gradient_error=max(gradient_error,float(max(abs(fd-grad))))
        hessian_error=max(hessian_error,float(abs(fh-loss.hessian(b)).max()))
        xx=np.ones((8,1));yy=np.r_[np.ones(3),np.zeros(5)]
        q=alpha*yy.mean()+(1-alpha)/2;expected=logit(q)
        fit,check=interior_fit(dy_loss(xx,yy,alpha),np.zeros(1));assert fit is not None
        root_error=max(root_error,abs(fit[0]-expected))
        covariance=dy_covariance(xx,yy,alpha,fit)
        exact=alpha**2*np.var(yy,ddof=1)/(len(yy)*q*q*(1-q)**2)
        cov_error=max(cov_error,abs(float(covariance[0,0])-exact))
    assert gradient_error<1e-7 and hessian_error<1e-7 and value_error<1e-12
    assert root_error<1e-7 and cov_error<1e-7
    result={'analytic_intercept_root_max_error':root_error,'analytic_intercept_covariance_max_error':cov_error,
            'gradient_finite_difference_max_error':gradient_error,'hessian_finite_difference_max_error':hessian_error,
            'independent_probability_loss_max_error':value_error,'external_research_code_executed':False,
            'HD_Gaussian_rescaling_implemented':False,'priority_proven':False}
    save(out/'dypl_comparator_numeric_checks.json',result);return result


def summarize(rows,old,out):
    f=pd.DataFrame(rows);f.to_csv(out/'dypl_comparator_trial_diagnostics.csv',index=False)
    summary=f.groupby(['condition','method'],as_index=False).agg(
        replications=('replication','size'),estimates=('estimate_available','sum'),intervals=('interval_available','sum'),
        alpha=('alpha','mean'),MSE=('MSE','mean'),median_MSE=('MSE','median'),q90_MSE=('MSE',lambda s:s.quantile(.9)),
        expected_KL=('excess_logloss','mean'),conditional_marginal_coverage=('marginal_coverage','mean'),
        interval_and_marginal_coverage=('issued_and_marginal_coverage','mean'),maximum_score=('score_max','max'))
    summary.to_csv(out/'dypl_comparator_summary.csv',index=False)
    f.groupby(['condition','method','status'],as_index=False).size().to_csv(out/'dypl_comparator_status_counts.csv',index=False)
    pair=[]
    for (condition,method),part in f.groupby(['condition','method']):
      for existing,ref in old[old.condition==condition].groupby('method'):
        joined=part.merge(ref,on='replication',suffixes=('_DY','_old'))
        for metric in ['MSE','excess_logloss']:
            shared=joined[[metric+'_DY',metric+'_old']].dropna()
            d=shared[metric+'_DY']-shared[metric+'_old']
            pair.append({'condition':condition,'method':method,'comparator':existing,'metric':metric,
                         'shared_available_fits':len(shared),'mean_DY':shared[metric+'_DY'].mean(),
                         'mean_comparator':shared[metric+'_old'].mean(),'mean_difference':d.mean(),
                         'paired_MCSE':d.std(ddof=1)/np.sqrt(len(d)),
                         'scope':'conditional on shared available fits; no new independent draws'})
    pd.DataFrame(pair).to_csv(out/'dypl_comparator_paired.csv',index=False)


def replay(input_dir,references,out):
    weak=references/'finite_pool_trial_diagnostics.csv';strong=references/'strong_signal_pool_trial_diagnostics.csv'
    old_weak=pd.read_csv(weak).query('label_rows==20 and auxiliary_rows==600 and epsilon==0.25').copy()
    old_strong=pd.read_csv(strong).query('label_rows==150 and prediction=="overconfident"').copy()
    old_weak['condition']='20_labels_oracle';old_strong['condition']='150_labels_overconfident'
    old=pd.concat([old_weak,old_strong],ignore_index=True)
    assert len(old)==7000 and old.groupby(['condition','method']).size().eq(500).all()
    source,observed,_,_,_=load_data(input_dir);base=source[:1200]
    base_truth,_=converged(Logistic(base,observed[:1200]));p=base.shape[1]
    rows=[];matches=0;max_relative=0.;max_score=0.;critical=norm.ppf(.975)
    source_hash=digest(Path(__file__));reference_hashes={weak.name:digest(weak),strong.name:digest(strong)}
    originals={name:digest(Path(__file__).with_name(name)) for name in ['finite_pool_choice_validation.py','imperfect_strong_signal_validation.py']}
    population=[]
    for condition,m,multiplier,seed in [('20_labels_oracle',20,1,20261112),('150_labels_overconfident',150,3,20261121)]:
        truth=multiplier*base_truth;prob=expit(base@truth);rng=np.random.default_rng(seed)
        human_ref=old[(old.condition==condition)&(old.method=='human_MLE')].set_index('replication')
        for method,alpha in [('DY_dimension_power',m/(m+p)),('DY_half_power',.5)]:
            expected=alpha*prob+(1-alpha)/2
            target=WeightedLogit(base,np.full(len(base),1/len(base)),base.T@expected/len(base))
            beta,check=interior_fit(target,np.zeros(p));assert beta is not None
            population.append({'condition':condition,'method':method,'alpha':alpha,
                               'human_parameter_target_MSE':float(np.mean((beta-truth)**2)),
                               'softened_population_score_max':check['score_max']})
        for rep in range(500):
            primary=base[rng.integers(0,len(base),150)];aux=base[rng.integers(0,len(base),600)]
            yp=(rng.random(150)<expit(primary@truth)).astype(float)
            yu=(rng.random(600)<expit(aux@truth)).astype(float)
            fp=rng.random(150);fu=rng.random(600)  # consume the exact original stream
            x=primary[:m];y=yp[:m]
            human_loss=WeightedLogit(x,np.full(m,1/m),x.T@y/m)
            initial,_=interior_fit(human_loss,np.zeros(p),rho=1/m);assert initial is not None
            human,check=solve(human_loss,initial);reference=human_ref.loc[rep]
            assert check['status']==reference.status,(condition,rep,check,reference.status)
            assert (human is not None)==bool(reference.estimate_available)
            if human is not None:
                mse=float(np.mean((human-truth)**2));err=abs(mse-reference.MSE)/(1+reference.MSE)
                max_relative=max(max_relative,err);assert err<1e-5,(condition,rep,err)
            matches+=1
            for method,alpha in [('DY_dimension_power',m/(m+p)),('DY_half_power',.5)]:
                row={'condition':condition,'replication':rep,'label_rows':m,'auxiliary_rows_used':0,
                     'method':method,'alpha':alpha,'estimate_available':False,'interval_available':False,
                     'issued_and_marginal_coverage':0.,'MSE':np.nan,'excess_logloss':np.nan,
                     'marginal_coverage':np.nan,'score_max':np.nan}
                if np.linalg.matrix_rank(x)<p:row['status']='rank_deficient'
                else:
                    beta,fit=interior_fit(dy_loss(x,y,alpha),np.zeros(p));row.update(fit)
                    if beta is not None:
                        error=beta-truth;u=base@beta;u0=base@truth
                        KL=float(np.mean(np.logaddexp(0,u)-np.logaddexp(0,u0)-prob*(u-u0)))
                        assert KL>=-1e-12
                        row.update(estimate_available=True,MSE=float(np.mean(error**2)),excess_logloss=KL)
                        cov=dy_covariance(x,y,alpha,beta)
                        if cov is not None:
                            coverage=float(np.mean(abs(error)<=critical*np.sqrt(np.diag(cov))))
                            row.update(interval_available=True,marginal_coverage=coverage,issued_and_marginal_coverage=coverage)
                    if fit.get('score_max') is not None:max_score=max(max_score,fit['score_max'])
                rows.append(row)
            if (rep+1)%100==0:
                summarize(rows,old,out)
                save(out/'dypl_comparator_status.json',{'completed_original_draws':matches,'requested_original_draws':1000,
                    'completed_DY_fits':len(rows),'requested_DY_fits':2000,'source_sha256':source_hash,
                    'original_source_sha256':originals,'reference_sha256':reference_hashes,
                    'human_reconstruction_status_matches':matches,'maximum_human_MSE_relative_difference':max_relative,
                    'maximum_DY_score':max_score,'original_reference_files_unchanged':all(digest(references/n)==h for n,h in reference_hashes.items()),
                    'new_independent_responses':False,'new_random_experiment':False,
                    'HD_Gaussian_rescaling_implemented':False,'submission_readiness_proven':False})
                print(condition,rep+1,'of500',flush=True)
    pd.DataFrame(population).to_csv(out/'dypl_comparator_population_targets.csv',index=False)
    assert digest(Path(__file__))==source_hash
    assert all(digest(references/n)==h for n,h in reference_hashes.items())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path);p.add_argument('--references',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--checks-only',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        print(json.dumps(numeric_checks(a.out)),flush=True)
        if not a.checks_only:
            assert a.input_dir is not None and a.references is not None
            replay(a.input_dir,a.references,a.out)
