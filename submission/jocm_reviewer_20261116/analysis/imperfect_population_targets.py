"""Exact expected targets on the finite released attribute support."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.special import expit
from public_conjoint_optimizer_audit import load_data,Logistic,converged,save


def main():
 p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--truth-scale',type=float,default=1.);p.add_argument('--prefix',default='imperfect_pool');a=p.parse_args()
 x,y,_,_,_=load_data(a.input_dir);x=x[:1200];truth,_=converged(Logistic(x,y[:1200]));truth=a.truth_scale*truth;mu=expit(x@truth);pz=.25+.5*mu
 rows=[];maxscore=0.;oracleerror=0.
 for name,slope,bias in [('oracle',1.,0.),('overconfident',2.,0.),('intercept_bias',1.,1.)]:
  g1=expit(slope*(x@truth+np.log(3))+bias);g0=expit(slope*(x@truth-np.log(3))+bias);gbar=pz*g1+(1-pz)*g0
  if name=='oracle':oracleerror=float(max(abs(gbar-mu)))
  for lam in [0.,.25,.5,1.]:
   primary=x.T@((1-lam)*mu-mu+lam*gbar)/len(x)
   aux=lam*x.T@(mu-gbar)/len(x);maxscore=max(maxscore,float(max(abs(primary+aux))))
  for m in [50,150]:
   expected_target=(m*mu+600*gbar)/(m+600);beta,check=converged(Logistic(x,expected_target));error=beta-truth
   rows.append({'prediction':name,'truth_scale':a.truth_scale,'label_rows':m,'auxiliary_rows':600,'prediction_mean_error_max':float(max(abs(gbar-mu))),
    'prediction_target_score_max':float(max(abs(x.T@(gbar-mu)/len(x)))),'soft_population_parameter_MSE':float(np.mean(error**2)),
    'soft_population_coefficient_difference_max':float(max(abs(error))),
    'soft_population_excess_logloss':float(np.mean(np.logaddexp(0,x@beta)-np.logaddexp(0,x@truth)-mu*(x@error))),
    'soft_target_converged_score_max':float(max(abs(Logistic(x,expected_target).value(beta)[1]))),
    'scope':'exact empirical-support expectation under the frozen simulation truth; not observed human structural truth'})
 assert maxscore<1e-12 and oracleerror<1e-12
 df=pd.DataFrame(rows);assert df.soft_target_converged_score_max.max()<1e-7
 df.to_csv(a.out/(a.prefix+'_population_targets.csv'),index=False)
 save(a.out/(a.prefix+'_population_checks.json'),{'truth_scale':a.truth_scale,'fixed_power_corrected_expected_score_max':maxscore,'oracle_conditional_mean_identity_error':oracleerror,'expected_soft_target_fits':len(rows),'fixed_power_target_identity_is_prior_work':True,'data_selected_weight_unbiasedness_asserted':False})
 print(df[['prediction','label_rows','soft_population_parameter_MSE','soft_population_excess_logloss']].to_string(index=False))


if __name__=='__main__':main()
