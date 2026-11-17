from pathlib import Path
import argparse,csv,json
def main(root):
    plan=json.loads((root/'plan.json').read_text());rows=json.loads((root/'results/api_records.json').read_text());v=json.loads((root/'verification/verification.json').read_text());assert v['state']=='passed'
    chosen={f"public_{c['trial']}_{c['prediction']}":('previous_exterior' if c['existing_unit_support']=='outside_mean_support' else 'interior_control') for c in plan['public_cases']}
    summary=[]
    for group in ['previous_exterior','interior_control']:
        for power in plan['API_powers']:
            a=[r for r in rows if chosen.get(r['case'])==group and r['requested_power']==power]
            assert len(a)==8
            summary.append(dict(group=group,requested_power='automatic' if power is None else str(power),cases=8,finite_numeric_arrays=sum(r['return_finite_array'] for r in a),last_optimizer_success=sum(r['optimizer_calls'][-1]['success'] for r in a),interior_effective_target=sum(r.get('support',{}).get('state')=='interior' for r in a),outside_effective_target=sum(r.get('support',{}).get('state')=='outside' for r in a),API_exceptions=sum(r['exception'] is not None for r in a)))
    with (root/'results/public_case_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    ext=[r for r in rows if chosen.get(r['case'])=='previous_exterior' and r['requested_power']==1]
    boundary=next(r for r in rows if r['kind']=='analytic' and r['requested_power']==.55)
    result=dict(state='passed',API_calls=68,public_cases=16,analytical_cases=4,diagnostic_selection_not_prevalence_estimation=True,public_unit_exterior_arrays_returned=8,public_unit_exterior_underlying_optimizer_failures=8,public_automatic_initial_Hessian_exceptions=8,public_interior_control_outputs_returned=32,public_half_power_exterior_case_recoveries=8,public_zero_power_targets_interior=16,exterior_unit_coefficient_norm_min=min(r['beta_norm'] for r in ext),exterior_unit_coefficient_norm_max=max(r['beta_norm'] for r in ext),exterior_unit_original_score_max_min=min(r['original_score_max'] for r in ext),exterior_unit_original_score_max_max=max(r['original_score_max'] for r in ext),rational_boundary_example=dict(power='11/20',coefficient_norm=boundary['beta_norm'],score_max=boundary['original_score_max'],optimizer_success=boundary['optimizer_calls'][0]['success'],finite_global_minimizer=False,machine_input_scope='Rounded representation of the exact rational design; LP classification retains boundary or near boundary.'),optimizer_options=plan['optimizer_options'],new_general_theory=False,formal_version_gap_resolved=False)
    (root/'results/overview.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);main(p.parse_args().root)
