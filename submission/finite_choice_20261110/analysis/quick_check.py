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
 for script in ['verify_finite_pool_covariance.py','verify_matched_moment_proposition.py','verify_proper_augmentation_invariance.py','verify_zonotope_reduction.py','verify_general_hard_endpoint.py','power_domain_choice.py']:
  subprocess.run([sys.executable,str(root/'analysis'/script),'--out',str(a.out)],check=True,stdout=subprocess.DEVNULL)
 if a.full_math:
  for script in ['finite_ratio_signal_boundary.py','imperfect_predictor_choice_boundary.py','vanishing_power_choice.py','joint_power_choice.py','general_joint_power_boundary.py']:
   subprocess.run([sys.executable,str(root/'analysis'/script),'--out',str(a.out),'--checks-only'],check=True,stdout=subprocess.DEVNULL)
  subprocess.run([sys.executable,str(root/'analysis/verify_general_joint_geometry.py'),'--out',str(a.out)],check=True,stdout=subprocess.DEVNULL)
 grids={}
 for file,expected,reps in [('finite_ratio_phase_checks.csv',80,500),('finite_ratio_signal_phase.csv',80,300),('imperfect_prediction_phase.csv',80,300)]:
  f=pd.read_csv(root/'results'/file);assert len(f)==expected and (f.replications==reps).all() and f.unresolved.sum()==0
  assert ((f.feasible+f.outside+f.unresolved)==f.replications).all();grids[file]={'cells':len(f),'draws_per_cell':reps,'unresolved':int(f.unresolved.sum())}
 for file,expected in [('finite_pool_summary.csv',84),('imperfect_pool_summary.csv',42),('strong_signal_pool_summary.csv',42)]:
  f=pd.read_csv(root/'results'/file);assert len(f)==expected and (f.replications==500).all();assert (f.intervals<=f.estimates).all() and (f.estimates<=f.replications).all()
 t2=pd.read_csv(root/'manuscript/table2.csv',dtype=str)
 assert len(t2)==18
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
 dy=pd.read_csv(root/'results/dypl_comparator_summary.csv')
 ds=json.loads((root/'results/dypl_comparator_status.json').read_text())
 assert ds['completed_original_draws']==ds['human_reconstruction_status_matches']==1000 and ds['completed_DY_fits']==2000
 assert ds['maximum_human_MSE_relative_difference']<1e-5 and ds['original_reference_files_unchanged']
 assert ds['source_sha256']==hashlib.sha256((root/'analysis/replay_dypl_comparator.py').read_bytes()).hexdigest()
 assert len(dy)==4 and (dy.replications==500).all() and (dy.estimates==500).all()
 for condition,shown in [('20_labels_oracle','20 labels oracle'),('150_labels_overconfident','150 labels overconfident')]:
  for method,label in [('DY_dimension_power','DY dimension rule'),('DY_half_power','DY alpha 0.5')]:
   r=dy[(dy.condition==condition)&(dy.method==method)].iloc[0];t=t2[(t2.Condition==shown)&(t2.Estimator==label)].iloc[0]
   assert t.Available==f'{int(r.estimates)}/500'
   for field,raw,tol in [('MSE','MSE',.00005),('Coverage','conditional_marginal_coverage',.0005),('Issued and covers','interval_and_marginal_coverage',.0005)]:assert abs(float(t[field])-r[raw])<=tol
 dp=pd.read_csv(root/'results/dypl_comparator_paired.csv');assert len(dp)==56
 assert (abs(dp.mean_DY-dp.mean_comparator-dp.mean_difference)<1e-12).all()
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
 joint=json.loads((root/'results/joint_power_status.json').read_text())
 assert joint['completed_DGP_blocks']==joint['original_limit_aggregate_matches']==12
 assert joint['original_draws_replayed']==6000 and joint['new_positive_power_fits']==12000
 assert joint['source_sha256']==hashlib.sha256((root/'analysis/joint_power_choice.py').read_bytes()).hexdigest()
 assert joint['reference_sha256']==hashlib.sha256((root/'results/vanishing_power_phase_verified.csv').read_bytes()).hexdigest()
 assert joint['raw_unresolved']==joint['remaining_unresolved']==0
 jf=pd.read_csv(root/'results/joint_power_phase_verified.csv');assert len(jf)==36
 assert ((jf.interior+jf.outside+jf.boundary_or_near_boundary+jf.unresolved)==500).all()
 kl=pd.read_csv(root/'results/replayed_choice_loss_pairs.csv');assert len(kl)==4
 assert (abs(kl.mean_KL_comparator-kl.mean_KL_reference-kl.mean_KL_difference)<1e-12).all()
 assert (kl[kl.power_reference==1].shared_finite_estimates==412).all()
 general=json.loads((root/'results/general_joint_power_phase_verification.json').read_text())
 gs=json.loads((root/'results/general_joint_power_status.json').read_text())
 gr=pd.read_csv(root/'results/general_joint_power_phase_raw.csv');gv=pd.read_csv(root/'results/general_joint_power_phase_verified.csv')
 assert gs['completed_DGP_blocks']==12 and len(gr)==len(gv)==180
 assert general['complete_feasibility_calls']==90000 and general['independent_Gaussian_draws']==6000
 assert general['initial_unknowns']==general['resolved_by_independent_certificates']==gr.unresolved.sum()
 assert general['remaining_unresolved']==gv.unresolved.sum()==0
 assert general['raw_aggregate_sha256']==hashlib.sha256((root/'results/general_joint_power_phase_raw.csv').read_bytes()).hexdigest()
 assert general['source_sha256']==gs['source_sha256']==hashlib.sha256((root/'analysis/general_joint_power_boundary.py').read_bytes()).hexdigest()
 assert (gv[['interior','outside','boundary_or_near_boundary','rank_deficient','unresolved']].sum(axis=1)==500).all()
 for case in general['resolutions']:
  cert=case['independent_verification']
  if cert['status']=='outside_mean_support':assert cert['separation_gap']>1e-8
  else:assert cert['status']=='interior' and cert['interior_margin']>1e-8 and cert['equality_residual']<1e-7
 gp=json.loads((root/'results/general_joint_power_population_checks.json').read_text())
 assert gp['population_conditions']==30 and gp['finite_power_comparisons']==120
 assert gp['200_400_node_max_error']<1e-7 and gp['zero_scale_derivative_identity_error']<1e-9
 gg=json.loads((root/'results/general_joint_power_geometry_checks.json').read_text())
 assert gg['original_rescaled_LP_matches']==120 and gg['independent_limiting_projection_cases']==24
 assert gg['maximum_projection_norm_squared_difference']<1e-6
 domain=json.loads((root/'results/power_domain_replay_checks.json').read_text())
 domain_summary=pd.read_csv(root/'results/power_domain_summary.csv')
 domain_cases=json.loads((root/'results/power_domain_exterior_certificates.json').read_text())
 assert domain['completed_cases']==domain['interior_witnesses_independently_checked']==domain['existing_selected_power_support_interior']==1000
 assert domain['issues']==domain['original_unit_status_mismatches']==0
 assert len(domain_cases)==domain['interior_cutoff_certificates']==8
 assert domain_summary.allocations.eq(500).all() and domain_summary.positive_intervals.eq(500).all()
 assert domain_summary.unit_exterior.sum()==domain_summary.endpoint_below_one.sum()==8
 assert domain_summary.existing_selected_power_zero.sum()==domain['existing_selected_zero_powers']
 for name,digest in domain['source_hashes'].items():
  if name.endswith('.py'):assert hashlib.sha256((root/'analysis'/name).read_bytes()).hexdigest()==digest
 for case in domain_cases:
  cut=case['closed_endpoint'];h=case['human_recession'];cost=case['unit_recession']
  assert 0<cut<1 and h>0 and cost<0 and abs(h/(h-cost)-cut)<1e-7
  assert 0<=case['existing_selected_power']<cut and case['above_endpoint_direct_recession']<0
  assert case['maximum_independent_identity_error']<1e-7
  brackets=case['fixed_power_brackets']
  assert brackets['half_endpoint']['status']==brackets['below_endpoint']['status']=='interior'
  assert brackets['at_endpoint']['status']=='boundary_or_near_boundary' and brackets['above_endpoint']['status']=='outside_mean_support'
 report={'analytic_intercept_root_error':err,'boundary_exterior_rank_checks':True,'balanced_LP_and_quadrature':balanced,'completed_grid_consistency':grids,'full_math_checks_run':a.full_math,'reported_table_rounding_checked':True,'positive_power_paired_draws':6000,'raw_unknown_preserved_and_resolved':True,'original_draws_replayed':500,'old_support_status_matches':1000,'all_three_tables_reconciled':True,'joint_power_existing_draws_replayed':6000,'joint_power_new_feasibility_checks':12000,'classical_zonotope_reduction_checked':True,'paired_expected_choice_loss_checked':True,'DY_comparator_original_draws_replayed':1000,'DY_comparator_fits':2000,'general_all_power_boundary_checked':True,'excluded_hard_endpoint_scope_checked':True,'general_phase_calls':90000,'general_initial_unknowns':general['initial_unknowns'],'finite_design_power_domain_cases':1000,'independent_endpoint_recession_certificates':8,'requires_raw_questionnaires':False,'proves_novelty_or_submission_readiness':False}
 (a.out/'quick_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'analytic_root_error':err,'original_interval_LP_matches':balanced['original_interval_LP_matches'],'full_math_checks':a.full_math,'aggregate_tables_consistent':True}))

if __name__=='__main__':main()
