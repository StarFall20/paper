"""Independent analytic benchmarks and aggregate/source consistency checks.

Run without public input downloads. This is not an originality or acceptance test.
"""
from pathlib import Path
import argparse,json,subprocess,sys,hashlib
import numpy as np
import pandas as pd
from audit_binary_choice import audit
from finite_ratio_choice_boundary import numeric_checks

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('regenerated/checks'));p.add_argument('--full-math',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 root=Path(__file__).resolve().parents[1]
 x=np.ones((4,1));y=np.array([0,1,np.nan,np.nan]);g=np.array([.5,.5,.3,.3])
 result=audit(x,y,g,2);assert result['fit_status']=='finite_root';err=abs(result['coefficient'][0]-np.log(.3/.7));assert err<1e-7
 boundary=audit(x,np.array([0,0,np.nan,np.nan]),np.full(4,.1),2);assert boundary['support']['status']=='boundary_or_near_boundary'
 exterior=audit(x,np.array([0,0,np.nan,np.nan]),np.array([.9,.9,.1,.1]),2);assert exterior['support']['status']=='outside_mean_support'
 rank=audit(np.ones((4,2)),y,g,2);assert rank['support']['status']=='rank_deficient'
 balanced=numeric_checks()
 for script in ['verify_finite_pool_covariance.py','verify_matched_moment_proposition.py','verify_proper_augmentation_invariance.py']:
  subprocess.run([sys.executable,str(root/'analysis'/script),'--out',str(a.out)],check=True,stdout=subprocess.DEVNULL)
 if a.full_math:
  for script in ['finite_ratio_signal_boundary.py','imperfect_predictor_choice_boundary.py','vanishing_power_choice.py']:
   subprocess.run([sys.executable,str(root/'analysis'/script),'--out',str(a.out),'--checks-only'],check=True,stdout=subprocess.DEVNULL)
 grids={}
 for file,expected,reps in [('finite_ratio_phase_checks.csv',80,500),('finite_ratio_signal_phase.csv',80,300),('imperfect_prediction_phase.csv',80,300)]:
  f=pd.read_csv(root/'results'/file);assert len(f)==expected and (f.replications==reps).all() and f.unresolved.sum()==0
  assert ((f.feasible+f.outside+f.unresolved)==f.replications).all();grids[file]={'cells':len(f),'draws_per_cell':reps,'unresolved':int(f.unresolved.sum())}
 for file,expected in [('finite_pool_summary.csv',84),('imperfect_pool_summary.csv',42),('strong_signal_pool_summary.csv',42)]:
  f=pd.read_csv(root/'results'/file);assert len(f)==expected and (f.replications==500).all();assert (f.intervals<=f.estimates).all() and (f.estimates<=f.replications).all()
 t2=pd.read_csv(root/'manuscript/table2.csv',dtype=str)
 assert len(t2)==14
 labels={'Human MLE':'human_MLE','Unit PPI':'PPI_unit','Power human start':'PPI_power_human','Power ridge start':'PPI_power_ridge','One step':'PPI_one_step_ridge','Ridge PPI':'PPI_ridge_unit','Soft pooling':'soft_oracle_pool'}
 for condition,name in [('20 labels oracle','finite_pool_summary.csv'),('150 labels overconfident','strong_signal_pool_summary.csv')]:
  f=pd.read_csv(root/'results'/name);f=f[(f.label_rows==(20 if '20 labels' in condition else 150))&(f.auxiliary_rows==600)]
  f=f[f.epsilon==.25] if 'epsilon' in f else f[f.prediction=='overconfident']
  for label,method in labels.items():
   if name.startswith('strong') and method=='soft_oracle_pool':method='soft_frozen_pool'
   r=f[f.method==method].iloc[0];t=t2[(t2.Condition==condition)&(t2.Estimator==label)].iloc[0]
   assert t.Available==f'{int(r.estimates)}/500'
   for shown,raw,tol in [('MSE','MSE',.00005),('Coverage','conditional_marginal_coverage',.0005),('Issued and covers','interval_and_marginal_coverage',.0005)]:assert abs(float(t[shown])-r[raw])<=tol
 t1=pd.read_csv(root/'manuscript/table1.csv',dtype=str);b=pd.read_csv(root/'results/imperfect_prediction_boundaries.csv')
 for row in t1.to_dict('records'):
  pred={'Oracle':'oracle','Overconfident':'overconfident','Shifted log odds':'intercept_bias'}[row['Prediction']]
  sub=b[(b.tau==float(row['Auxiliary to label ratio']))&(b.prediction==pred)];r=sub.loc[(sub.power-float(row['Power'])).abs().idxmin()]
  assert abs(float(row['Power'])-r.power)<=.00005 and abs(float(row['Critical slope to label ratio'])-r.critical_ratio)<=.00005
 # Preserve the original numerical failure and require its independent resolution.
 power_status=json.loads((root/'results/vanishing_power_status.json').read_text())
 assert power_status['completed_DGP_blocks']==12 and power_status['replications_per_block']==500
 assert power_status['positive_power_nesting_violations']==0
 assert power_status['source_sha256']==hashlib.sha256((root/'analysis/vanishing_power_choice.py').read_bytes()).hexdigest()
 raw_phase=root/'results/vanishing_power_phase.csv';raw=pd.read_csv(raw_phase)
 verified=pd.read_csv(root/'results/vanishing_power_phase_verified.csv')
 resolution=json.loads((root/'results/vanishing_power_resolution.json').read_text())
 assert resolution['raw_phase_sha256']==hashlib.sha256(raw_phase.read_bytes()).hexdigest()
 assert raw.unresolved.sum()==resolution['raw_unresolved']==1
 assert resolution['remaining_unresolved']==verified.unresolved.sum()==0
 assert len(verified)==72 and ((verified.interior+verified.outside+verified.boundary_or_near_boundary+verified.unresolved)==500).all()
 assert len(resolution['cases'])==1 and resolution['cases'][0]['independent_original_moment_check']['separation_gap']>0
 replay=json.loads((root/'results/replayed_power_status.json').read_text())
 assert replay['completed_repetitions']==500 and replay['original_human_unit_reconstruction_status_matches']==1000
 assert replay['source_sha256']==hashlib.sha256((root/'analysis/replay_choice_power_path.py').read_bytes()).hexdigest()
 assert replay['maximum_reconstructed_MSE_relative_difference']<1e-5
 overall=pd.read_csv(root/'results/replayed_power_overall_summary.csv')
 t3=pd.read_csv(root/'manuscript/table3.csv',dtype=str)
 assert len(overall)==len(t3)==5 and (overall.estimates==overall.finite_support).all()
 for row in overall.to_dict('records'):
  label='0 human' if row['power']==0 else f"{row['power']:g}";t=t3[t3.Power==label].iloc[0]
  assert t.Available==f"{int(row['estimates'])}/500"
  for shown,field in [('Mean MSE','MSE'),('Median MSE','median_MSE'),('90th percentile MSE','q90_MSE')]:assert abs(float(t[shown])-row[field])<=.0005
 pairs=pd.read_csv(root/'results/replayed_power_paired_comparisons.csv')
 unit=pairs[pairs.power_reference==1];assert len(unit)==3 and (unit.shared_finite_estimates==412).all()
 assert (abs(pairs.MSE_comparator-pairs.MSE_reference-pairs.mean_MSE_difference)<1e-10).all()
 report={'analytic_intercept_root_error':err,'boundary_exterior_rank_checks':True,'balanced_LP_and_quadrature':balanced,'completed_grid_consistency':grids,'full_math_checks_run':a.full_math,'reported_table_rounding_checked':True,'positive_power_paired_draws':6000,'raw_unknown_preserved_and_resolved':True,'original_draws_replayed':500,'old_support_status_matches':1000,'all_three_tables_reconciled':True,'requires_raw_questionnaires':False,'proves_novelty_or_submission_readiness':False}
 (a.out/'quick_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'analytic_root_error':err,'original_interval_LP_matches':balanced['original_interval_LP_matches'],'full_math_checks':a.full_math,'aggregate_tables_consistent':True}))

if __name__=='__main__':main()
