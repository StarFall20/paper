"""Independent analytic benchmarks and aggregate/source consistency checks.

Run without public input downloads. This is not an originality or acceptance test.
"""
from pathlib import Path
import argparse,json,subprocess,sys,hashlib,shutil
from fractions import Fraction
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
 for script in ['verify_finite_pool_covariance.py','verify_matched_moment_proposition.py','verify_proper_augmentation_invariance.py','verify_zonotope_reduction.py','verify_general_hard_endpoint.py','power_domain_choice.py','attribute_aligned_prediction.py','calibrated_aligned_prediction.py','verify_trained_prediction_rotation.py']:
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
 ap=json.loads((root/'results/aligned_prediction_numeric_checks.json').read_text())
 cp=json.loads((root/'results/calibrated_aligned_numeric_checks.json').read_text())
 phase=json.loads((root/'results/aligned_phase_status.json').read_text())
 rotation=json.loads((root/'results/trained_prediction_rotation_checks.json').read_text())
 assert ap['population_conditions']==18 and cp['population_conditions']==12
 assert ap['original_oriented_LP_matches']==234 and len(ap['separately_verified_rank_deficient_cases'])==6
 assert rotation['corrected_objective_support_rotation_matches']==180
 for name,record in [('attribute_aligned_prediction.py',ap),('calibrated_aligned_prediction.py',cp),('aligned_prediction_phase.py',phase),('verify_trained_prediction_rotation.py',rotation)]:
  assert hashlib.sha256((root/'analysis'/name).read_bytes()).hexdigest()==record['source_sha256']
 assert phase['paired_draws']==1000 and phase['completed_cells']==16 and phase['feasibility_calls']==8000 and phase['unknowns']==0
 assert phase['population_sha256']==hashlib.sha256((root/'results/calibrated_aligned_boundaries.csv').read_bytes()).hexdigest()
 table=pd.read_csv(root/'manuscript/tableE1.csv',dtype=str);paired=pd.read_csv(root/'results/aligned_phase_paired_summary.csv')
 assert len(table)==len(paired)==8 and paired.paired_difference.gt(0).sum()==1
 for row in paired.to_dict('records'):
  shown=table[(table.Labels==str(int(row['labels'])))&(table.Slopes==str(int(row['total_slopes'])))].iloc[0]
  for column,field,decimals in [('External','external_frequency',3),('Aligned','aligned_frequency',3),('Difference','paired_difference',3),('MCSE','paired_MCSE',4)]:assert shown[column]==f'{row[field]:.{decimals}f}'
  assert abs(row['aligned_frequency']-row['external_frequency']-row['paired_difference'])<1e-12
 initial=json.loads((root/'results/aligned_prediction_initial_diagnostic_attempt.json').read_text())
 assert initial['initial_source_sha256']==hashlib.sha256((root/'reports/public_conjoint_audit_20261109/aligned_initial_diagnostic_source.py').read_bytes()).hexdigest()
 budget=pd.read_csv(root/'results/training_budget_summary.csv')
 budget_status=pd.read_csv(root/'results/training_budget_status.csv')
 budget_pairs=pd.read_csv(root/'results/training_budget_paired.csv')
 budget_check=json.loads((root/'results/training_budget_verification.json').read_text())
 budget_run=json.loads((root/'results/training_budget_status.json').read_text())
 adaptation=json.loads((root/'results/training_budget_source_adaptation.json').read_text())
 assert budget_run['state']=='complete' and budget_run['returned_method_rows']==4800 and budget_run['completed_model_draws']==800
 assert len(budget)==48 and len(budget_pairs)==80 and budget.draws.eq(100).all()
 assert budget_status['size'].sum()==4800 and budget_run['unresolved_method_rows']==0
 assert budget_run['source_sha256']==adaptation['original_run_source_sha256']
 assert adaptation['packaged_source_sha256']==hashlib.sha256((root/'analysis/training_budget_diagnostic_20261111.py').read_bytes()).hexdigest()
 assert budget_run['plan_sha256']==hashlib.sha256((root/'analysis/training_budget_plan_20261111.json').read_bytes()).hexdigest()
 counts=budget.groupby('method').verified_fits.sum().to_dict()
 assert counts=={'all_labels_mle':403,'all_labels_ridge_0.1':800,'corrected_power_0.1':223,'corrected_power_0.5':169,'corrected_power_1.0':28,'training_ridge_only':800}
 assert budget_check['ordinary_boundary_cases_separately_certified_strictly_separated']==397
 separators=pd.read_csv(root/'results/training_budget_mle_separation.csv')
 assert len(separators)==397 and separators.strict_separating_margin.gt(1e-9).all()
 table4=pd.read_csv(root/'manuscript/table4.csv',dtype=str)
 assert len(table4)==8
 for row in table4.to_dict('records'):
  part=budget[(budget.total_labels==int(row['Total labels']))&(budget.slope_dimension==int(row['Slopes']))&(budget.training_labels==int(row['Training labels']))].set_index('method')
  for shown,method in [('Human MLE','all_labels_mle'),('PPI $\\lambda=0.1$','corrected_power_0.1'),('Human ridge','all_labels_ridge_0.1')]:assert row[shown]==str(int(part.loc[method,'verified_fits']))+'/100'
 negative=budget_pairs[budget_pairs.method.str.startswith('corrected')&(budget_pairs.baseline=='all_labels_ridge_0.1')&budget_pairs.shared_verified_fits.gt(0)]
 assert len(negative)==16 and negative.excess_loss_difference.gt(0).all()
 report={'analytic_intercept_root_error':err,'boundary_exterior_rank_checks':True,'balanced_LP_and_quadrature':balanced,'completed_grid_consistency':grids,'full_math_checks_run':a.full_math,'reported_table_rounding_checked':True,'positive_power_paired_draws':6000,'raw_unknown_preserved_and_resolved':True,'original_draws_replayed':500,'old_support_status_matches':1000,'all_three_tables_reconciled':True,'joint_power_existing_draws_replayed':6000,'joint_power_new_feasibility_checks':12000,'classical_zonotope_reduction_checked':True,'paired_expected_choice_loss_checked':True,'DY_comparator_original_draws_replayed':1000,'DY_comparator_fits':2000,'general_all_power_boundary_checked':True,'excluded_hard_endpoint_scope_checked':True,'general_phase_calls':90000,'general_initial_unknowns':general['initial_unknowns'],'finite_design_power_domain_cases':1000,'independent_endpoint_recession_certificates':8,'prediction_index_population_conditions':30,'prediction_index_paired_draws':1000,'prediction_index_phase_calls':8000,'appendix_tableE1_reconciled':True,'trained_prediction_rotation_support_matches':180,'initial_rank_classification_failure_and_source_retained':True,'requires_raw_questionnaires':False,'proves_novelty_or_submission_readiness':False}
 report.update(all_four_main_tables_reconciled=True,independent_training_budget_draws=800,training_budget_method_records=4800,training_budget_strict_separation_certificates=397,training_cost_adverse_results_retained=True)
 score=json.loads((root/'results/score/source_hashes.json').read_text())
 for path,digest in score['sha256'].items():assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest,path
 for key in ['population','phase']:assert hashlib.sha256((root/score[key+'_plan']).read_bytes()).hexdigest()==score[key+'_plan_sha256']
 for script in ['verify_variance_score_identity.py','certify_variance_reference_counterexample.py']:
  subprocess.run([sys.executable,str(root/'analysis'/script),'--out',str(a.out/'score_identity')],check=True,stdout=subprocess.DEVNULL)
 score_phase=a.out/'score_phase';shutil.copytree(root/'results/score/phase',score_phase,dirs_exist_ok=True)
 subprocess.run([sys.executable,str(root/'analysis/verify_variance_reference_phase.py'),'--plan',str(root/score['phase_plan']),'--out',str(score_phase),'--population',str(root/'results/score/population_clean'),'--certificate',str(root/'results/score/certified_bounds/rational_counterexample_certificate.json')],check=True,stdout=subprocess.DEVNULL)
 certificate=json.loads((a.out/'score_identity/rational_counterexample_certificate.json').read_text())
 assert Fraction(certificate['unit_criterion_at_one']['lower_rational'])>Fraction(57,100)
 derivative=certificate['unit_criterion_derivative_at_one'];assert -Fraction(11,100)<Fraction(derivative['lower_rational'])<=Fraction(derivative['upper_rational'])<Fraction(11,100)
 assert Fraction(certificate['oracle_criterion_at_one']['upper_rational'])<Fraction(32,25)
 summary=pd.read_csv(root/'results/score/phase/variance_reference_phase_summary.csv')
 expected=summary.pivot(index=['label_rows','dimension_ratio'],columns='method',values='finite_frequency').reset_index()
 table=pd.read_csv(root/'manuscript/tableB1.csv');assert list(table.columns)==list(expected.columns) and np.allclose(table.to_numpy(float),expected.to_numpy(float),rtol=0,atol=1e-14)
 assert (root/'manuscript/tableB1.md').read_text().strip() in (root/'manuscript/paper.md').read_text()
 menu_publication=a.out/'menu_publication';shutil.copytree(root/'manuscript',menu_publication,dirs_exist_ok=True)
 subprocess.run([sys.executable,str(root/'analysis/build_menu_publication_assets.py'),'--results',str(root/'results/menu'),'--manuscript',str(menu_publication)],check=True,stdout=subprocess.DEVNULL)
 report.update(score_identity_source_closure_files=len(score['sha256']),identical_score_phase_records_reconciled=16000,identical_truth_score_groups=4000,independent_score_path_separator_directions=2355,score_path_remaining_unknowns=0,oracle_counterexample_rational_bounds_checked=True,appendix_tableB1_reconciled=True,appendix_tablesF1F2_regenerated_and_reconciled=True,all_eight_manuscript_tables_reconciled=True,all_original_Monte_Carlo_LPs_rerun=False)
 observed=json.loads((root/'results/observable/source_hashes.json').read_text())
 for path,digest in observed['sha256'].items():assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest,path
 subprocess.run([sys.executable,str(root/'analysis/certify_observable_score_path.py'),'--out',str(a.out/'observable_exact')],check=True,stdout=subprocess.DEVNULL)
 exact=json.loads((a.out/'observable_exact/observable_rational_example.json').read_text())
 assert exact['base_variance_power']=='15/88' and exact['finite_power_cutoff']=='9/44'
 assert exact['all_computed_score_rows_identical'] and exact['all_empirical_score_middles_identical'] and exact['all_recomputed_variance_powers_equal_path_powers']
 observed_pairs=pd.read_csv(root/'results/observable/high/verification/power_stratified_pairs.csv')
 checked_records=0
 for budget,folder,statusname in [(50,'high','observable_score_path_status.json'),(20,'low','observable_low_label_status.json')]:
  status=json.loads((root/f'results/observable/{folder}'/statusname).read_text());assert status['state']=='complete' and status['completed_allocations']==500
  frame=pd.read_csv(root/f'results/observable/{folder}/observable_score_path_records.csv');assert len(frame)==5000 and frame.groupby(['trial','prediction']).size().eq(5).all()
  assert frame.status.isin(['interior','outside_mean_support','boundary_or_near_boundary']).all()
  frame['base_power_stratum']=frame.base_power.map(lambda v:'zero' if v==0 else ('one' if v==1 else 'interior'))
  for (pred,stratum),part in frame.groupby(['prediction','base_power_stratum']):
   pivot=part.assign(finite=part.status.eq('interior')).pivot(index='trial',columns='fraction',values='finite')
   base=pivot[0.].to_numpy(int);end=pivot[1.].to_numpy(int)
   r=observed_pairs[(observed_pairs.label_rows==budget)&(observed_pairs.prediction==pred)&(observed_pairs.base_power_stratum==stratum)].iloc[0]
   assert (len(base),int(((base==1)&(end==1)).sum()),int(((base==1)&(end==0)).sum()),int(((base==0)&(end==1)).sum()),int(((base==0)&(end==0)).sum()))==(r.allocations,r.both_finite,r.base_only,r.endpoint_only,r.neither_finite)
  checked_records+=len(frame)
 tB2=pd.read_csv(root/'manuscript/tableB2.csv',dtype=str);assert len(tB2)==4
 for r in tB2.to_dict('records'):
  pred='AI' if r['Prediction']=='AI aided' else 'X_only';v=observed_pairs[(observed_pairs.label_rows==int(r['Labels']))&(observed_pairs.prediction==pred)&(observed_pairs.base_power_stratum=='interior')].iloc[0]
  assert [int(r[k]) for k in ['Allocations','Base finite','End finite','Base only','End only']]==[v.allocations,v.both_finite+v.base_only,v.both_finite+v.endpoint_only,v.base_only,v.endpoint_only]
 assert (root/'manuscript/tableB2.md').read_text().strip() in (root/'manuscript/paper.md').read_text()
 observed_verification=json.loads((root/'results/observable/high/verification/observable_path_complete_verification.json').read_text())
 assert observed_verification['retained_path_records']==10000 and observed_verification['replayed_support_classifications']==observed_verification['independent_residual_interior_LPs']==870
 assert observed_verification['all_low_label_exterior_directions_verified']==490 and observed_verification['all_ordinary_weak_separators_verified']==391 and observed_verification['all_unresolved']==0
 assert observed_verification['source_sha256']==hashlib.sha256((root/'analysis/verify_observable_score_path.py').read_bytes()).hexdigest()
 report.update(observed_reference_source_closure_files=len(observed['sha256']),observable_reference_records_reconciled=checked_records,observable_exact_scalar_example_checked=True,appendix_tableB2_reconciled=True,all_nine_manuscript_tables_reconciled=True,original_public_path_support_programs_all_rerun=False)
 complete_dir=a.out/'complete_gmm'
 subprocess.run([sys.executable,str(root/'analysis/certify_complete_gmm_comparison.py'),'--out',str(complete_dir)],check=True,stdout=subprocess.DEVNULL)
 subprocess.run([sys.executable,str(root/'analysis/verify_complete_gmm_comparison.py'),'--certificate',str(complete_dir/'complete_gmm_exact_comparison.json'),'--out',str(complete_dir/'verification')],check=True,stdout=subprocess.DEVNULL)
 complete=json.loads((complete_dir/'complete_gmm_exact_comparison.json').read_text())
 assert complete['complete_criterion_has_finite_global_minimizer'] and complete['unit_corrected_loss_unbounded_below'] and not complete['supervised_MLE_has_finite_minimizer']
 report['exact_complete_GMM_and_compensated_loss_comparison_checked']=True
 report['complete_GMM_comparison_is_not_production_CC_PPI']=True
 subprocess.run([sys.executable,str(root/'analysis/reconcile_probability_risk_tables.py'),'--out',str(a.out/'probability_risk_tables')],check=True,stdout=subprocess.DEVNULL)
 import re
 assert len(re.findall(r'(?:^\|.*\n)+',(root/'manuscript/paper.md').read_text(),re.M))==12
 report.update(probability_risk_path_records_reconciled=1920,probability_risk_numerical_fits=1401,new_probability_loss_tables_reconciled=['5','D1'],all_eleven_manuscript_tables_reconciled=True)
 subprocess.run([sys.executable,str(root/'analysis/build_radial_publication_assets.py'),'--out',str(a.out/'radial_publication')],check=True,stdout=subprocess.DEVNULL)
 report.update(all_twelve_manuscript_tables_reconciled=True,radial_nested_records=12000,radial_distinct_objectives=10000,radial_paired_transition_rows=24,radial_table6_reconciled=True)
 (a.out/'quick_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'analytic_root_error':err,'original_interval_LP_matches':balanced['original_interval_LP_matches'],'full_math_checks':a.full_math,'aggregate_tables_consistent':True,'score_identity_phase_records_reconciled':16000,'observed_reference_records_reconciled':checked_records,'all_nine_tables_reconciled':True}))

if __name__=='__main__':main()
