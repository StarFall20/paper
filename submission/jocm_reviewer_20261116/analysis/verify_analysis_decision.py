"""Closed-form audit checks and all recorded exterior-case decision examples."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.special import expit
from audit_binary_choice import audit
from public_conjoint_optimizer_audit import load_data,Logistic,converged,save
from public_conjoint_matched_audit import nuisance
from power_tuned_choice_audit import power_weight,WeightedLogit,objective,solve
from finite_pool_choice_validation import interior_fit


def main():
 p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 one=np.ones((4,1));y=np.array([0.,1.,np.nan,np.nan]);g=np.array([.5,.5,.3,.3])
 inside=audit(one,y,g,2);assert inside['fit_status']=='finite_root';root=inside['coefficient'][0];assert abs(root-np.log(.3/.7))<1e-7
 yy=np.array([0.,0.,np.nan,np.nan]);boundary=audit(one,yy,np.full(4,.1),2);assert boundary['support']['status']=='boundary_or_near_boundary' and boundary['coefficient'] is None
 outside=audit(one,yy,np.array([.9,.9,.1,.1]),2);assert outside['support']['status']=='outside_mean_support' and outside['support']['separation_gap']>0
 rank=audit(np.ones((4,2)),y,g,2);assert rank['support']['status']=='rank_deficient' and rank['coefficient'] is None
 cases=json.loads((a.out/'power_nonfinite_cases.json').read_text());cases=[v for v in cases if v['status']=='outside_mean_support'];assert len(cases)==8
 x,y,z,_,_=load_data(a.input_dir);x=x[:1200];y=y[:1200];rng=np.random.RandomState(20261109);rows=[]
 for trial in range(max(c['trial'] for c in cases)+1):
  order=(rng.permutation(240)[:,None]*5+np.arange(5)[None]).reshape(-1)
  for c in [v for v in cases if v['trial']==trial]:
   xx=x[order];yy=y[order];zz=z[order];m=c['label_rows'];g,_=nuisance(xx,yy,zz,m,c['prediction']=='AI')
   fixed=audit(xx,yy,g,m);assert fixed['support']['status']=='outside_mean_support'
   human,_=converged(Logistic(xx[:m],yy[:m]));lam,_=power_weight(xx,yy,g,m,human);assert lam is not None
   tuned=audit(xx,yy,g,m,lam);assert tuned['fit_status']=='finite_root'
   n=len(yy);soft=WeightedLogit(xx,np.full(n,1/n),xx.T@np.r_[yy[:m],g[m:]]/n);beta,check=interior_fit(soft,human);assert beta is not None
   ownscore=float(max(abs(soft.value(beta)[1])));correctedscore=float(max(abs(objective(xx,yy,g,m,1.).value(beta)[1])))
   assert ownscore<1e-7 and correctedscore>1e-7
   rows.append({'trial':trial,'prediction':c['prediction'],'label_rows':m,'unit_support':fixed['support']['status'],'unit_gap':fixed['support']['separation_gap'],
    'existing_tuned_power':lam,'tuned_status':tuned['fit_status'],'tuned_score':tuned['fit_check']['score_max'],'soft_own_score':ownscore,'soft_corrected_score':correctedscore,
    'interpretation':'existing tuning solves its stated objective; pooling solves a different stated objective; no CI comparison or new estimator claim'})
 pd.DataFrame(rows).to_csv(a.out/'analysis_decision_examples.csv',index=False)
 save(a.out/'analysis_decision_checks.json',{'analytic_root_max_error':abs(root-np.log(.3/.7)),'boundary_and_rank_checks':True,'all_recorded_exterior_cases':len(rows),
  'all_existing_tuned_roots_verified':all(v['tuned_status']=='finite_root' for v in rows),'minimum_exterior_gap':min(v['unit_gap'] for v in rows),
  'minimum_soft_corrected_score':min(v['soft_corrected_score'] for v in rows),'selection_scope':'all8 recorded exterior examples,not a new prevalence or performance sample','classical_LP_claimed_new':False,'new_estimator_claimed':False})
 print(json.dumps({'recorded_cases':len(rows),'minimum_gap':min(v['unit_gap'] for v in rows),'minimum_soft_corrected_score':min(v['soft_corrected_score'] for v in rows)}))


if __name__=='__main__':main()
