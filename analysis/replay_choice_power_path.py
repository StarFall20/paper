"""Refit the prespecified m20,N600,epsilon.25 draws from the completed study.

The original seed and draw order are replayed and checked against stored results.
No new respondent,LLM response or independent stochastic experiment is added.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits
from public_conjoint_optimizer_audit import load_data,Logistic,converged,save
from power_tuned_choice_audit import objective,support,WeightedLogit,solve
from finite_pool_choice_validation import interior_fit
from vanishing_power_choice import rescaled_support


class ScaledCorrected:
 def __init__(self,x,y,g,m,lam):self.x=x;self.y=y;self.g=g;self.m=m;self.lam=lam;self.s=2*y[:m]-1
 def value(self,b):
  u=self.x@b;m=self.m;lam=self.lam
  H=np.mean(np.logaddexp(0,-self.s*u[:m]))
  A=np.mean(np.logaddexp(0,u[m:])-self.g[m:]*u[m:])+np.mean((self.g[:m]-self.y[:m])*u[:m])
  hgrad=self.x[:m].T@(-self.s*expit(-self.s*u[:m]))/m
  agrad=self.x[m:].T@(expit(u[m:])-self.g[m:])/(len(u)-m)+self.x[:m].T@(self.g[:m]-self.y[:m])/m
  return float((1/lam-1)*H+A),(1/lam-1)*hgrad+agrad
 def hessian(self,b):
  p=expit(self.x@b);m=self.m;N=len(p)-m
  return (1/self.lam-1)*(self.x[:m].T@((p[:m]*(1-p[:m]))[:,None]*self.x[:m]))/m+(self.x[m:].T@((p[m:]*(1-p[m:]))[:,None]*self.x[m:]))/N


def fit_scaled(loss,initial):
 fit=minimize(loss.value,initial,jac=True,method='BFGS',options={'gtol':1e-10,'maxiter':2500});b=fit.x.copy()
 for _ in range(12):
  val,g=loss.value(b)
  if max(abs(g))<1e-10:break
  try:delta=np.linalg.solve(loss.hessian(b),g)
  except np.linalg.LinAlgError:break
  step=1.
  while step>1e-10 and loss.value(b-step*delta)[0]>val+1e-12:step/=2
  if step<=1e-10:break
  b-=step*delta
 score=float(max(abs(loss.value(b)[1])));eig=float(np.linalg.eigvalsh(loss.hessian(b)).min())
 if not np.isfinite(b).all() or score>1e-7 or eig<=1e-12:return None,{'fit_status':'fit_unresolved','scaled_score':score,'scaled_minimum_information':eig}
 return b,{'fit_status':'finite_root','scaled_score':score,'scaled_minimum_information':eig}


def summarize(rows,out):
 f=pd.DataFrame(rows);f.to_csv(out/'replayed_power_trial_diagnostics.csv',index=False)
 f.groupby(['power','human_support','support_status','fit_status'],dropna=False,as_index=False).size().to_csv(out/'replayed_power_status_counts.csv',index=False)
 f.groupby(['power','human_support'],dropna=False,as_index=False).agg(replications=('replication','size'),finite_support=('support_status',lambda v:int(v.eq('interior').sum())),estimates=('MSE','count'),MSE=('MSE','mean'),median_MSE=('MSE','median'),q90_MSE=('MSE',lambda s:s.quantile(.9)),mean_parameter_norm=('parameter_norm','mean'),median_parameter_norm=('parameter_norm','median'),mean_excess_logloss=('excess_logloss','mean'),maximum_scaled_score=('scaled_score','max')).to_csv(out/'replayed_power_summary.csv',index=False)
 f.groupby('power',dropna=False,as_index=False).agg(replications=('replication','size'),finite_support=('support_status',lambda v:int(v.eq('interior').sum())),estimates=('MSE','count'),MSE=('MSE','mean'),median_MSE=('MSE','median'),q90_MSE=('MSE',lambda s:s.quantile(.9)),mean_parameter_norm=('parameter_norm','mean'),mean_excess_logloss=('excess_logloss','mean'),maximum_scaled_score=('scaled_score','max')).to_csv(out/'replayed_power_overall_summary.csv',index=False)


def main():
 p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 reference=pd.read_csv(a.reference).query('label_rows==20 and auxiliary_rows==600 and epsilon==0.25 and method in ["human_MLE","PPI_unit"]').set_index(['replication','method']);assert len(reference)==1000
 source,observed,_,_,_=load_data(a.input_dir);base=source[:1200];truth,_=converged(Logistic(base,observed[:1200]));pb=expit(base@truth)
 # Independent value,gradient and Hessian checks for the scaled objective.
 x=np.array([[1.,0.],[0.,1.],[-1.,1.],[1.,-1.],[.5,.8]]);y=np.array([0.,1.,1.,0.,0.]);g=np.array([.2,.8,.6,.4,.7]);b=np.array([.2,-.3]);step=1e-5;deriv=0.
 for lam in [.01,.1,.5,1.]:
  loss=ScaledCorrected(x,y,g,3,lam);val,grad=loss.value(b);v0,g0=objective(x,y,g,3,lam).value(b)
  assert abs(val-v0/lam)<1e-10 and max(abs(grad-g0/lam))<1e-10
  eye=np.eye(2);fd=np.array([(loss.value(b+e*step)[0]-loss.value(b-e*step)[0])/(2*step) for e in eye]);fh=np.column_stack([(loss.value(b+e*step)[1]-loss.value(b-e*step)[1])/(2*step) for e in eye])
  deriv=max(deriv,float(max(abs(fd-grad))),float(abs(fh-loss.hessian(b)).max()))
 assert deriv<1e-7
 rng=np.random.default_rng(20261112);rows=[];maxdiff=0.;maxrelative=0.;matches=0;limitcounts={}
 for rep in range(500):
  primary=base[rng.integers(0,len(base),150)];aux=base[rng.integers(0,len(base),600)]
  yp=(rng.random(150)<expit(primary@truth)).astype(float);yu=(rng.random(600)<expit(aux@truth)).astype(float);fp=rng.random(150);fu=rng.random(600)
  x=np.vstack([primary[:20],aux]);y=np.r_[yp[:20],yu];z=np.r_[np.logical_xor(yp[:20],fp[:20]<.25),np.logical_xor(yu,fu<.25)].astype(float);g=expit(x@truth+(2*z-1)*np.log(3.))
  hloss=WeightedLogit(primary[:20],np.full(20,1/20),primary[:20].T@yp[:20]/20);initial,_=interior_fit(hloss,np.zeros(11),rho=1/20);human,hcheck=solve(hloss,initial)
  hs='interior' if human is not None else hcheck['status'];assert hcheck['status']==reference.loc[(rep,'human_MLE'),'status']
  lim=rescaled_support(x,y,g,20);limitcounts[lim['status']]=limitcounts.get(lim['status'],0)+1
  for lam in [0.,.01,.1,.5,1.]:
   if lam==0.:beta=human;sc={'status':hs};fit={'fit_status':hcheck['status'],'scaled_score':hcheck.get('score_max')}
   else:
    sc=rescaled_support(x,y,g,20,lam);beta=None;fit={'fit_status':'not_attempted','scaled_score':None}
    if sc['status']=='interior':beta,fit=fit_scaled(ScaledCorrected(x,y,g,20,lam),initial)
   row={'replication':rep,'power':lam,'human_support':hs,'support_status':sc['status'],**fit}
   if beta is not None:
    error=beta-truth;row.update(MSE=float(np.mean(error**2)),parameter_norm=float(np.linalg.norm(beta)),excess_logloss=float(np.mean(np.logaddexp(0,base@beta)-np.logaddexp(0,base@truth)-pb*(base@error))))
   if lam in [0.,1.]:
    old=reference.loc[(rep,'human_MLE' if lam==0 else 'PPI_unit')]
    expected='interior' if old['estimate_available'] else old['status'];assert sc['status']==expected,(rep,lam,sc,old['status'])
    if old['estimate_available']:
     assert beta is not None
     diff=abs(row['MSE']-old.MSE);maxdiff=max(maxdiff,diff);maxrelative=max(maxrelative,diff/(1+old.MSE));assert diff/(1+old.MSE)<1e-5,(rep,lam,diff,old.MSE)
    matches+=1
   assert sc['status']!='interior' or lim['status']=='interior'
   rows.append(row)
  if (rep+1)%25==0 or rep==499:
   summarize(rows,a.out);save(a.out/'replayed_power_status.json',{'completed_repetitions':rep+1,'requested_repetitions':500,'original_human_unit_reconstruction_status_matches':matches,'maximum_reconstructed_MSE_absolute_difference':maxdiff,'maximum_reconstructed_MSE_relative_difference':maxrelative,'scaled_derivative_check_error':deriv,'limiting_support_statuses':limitcounts,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'reference_trial_file_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest(),'replayed_original_draws':True,'new_independent_responses':False});print('Original support draws refitted',rep+1,'of500',flush=True)


if __name__=='__main__':
 with threadpool_limits(limits=1):main()
