"""Replay the complete power domain on unchanged public-choice allocations.

Classical support LPs audit the existing corrected objective. No new estimator,
response generation, or private-questionnaire observation enters this replay.
The individual diagnostic CSV stays local; aggregate checks and the eight
previously reported public-allocation certificates may be shared.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from public_conjoint_optimizer_audit import load_data, Logistic, converged, save
from public_conjoint_matched_audit import nuisance
from power_tuned_choice_audit import objective, support, power_weight
from power_domain_choice import power_domain, recession_costs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows, certificates, issues, out, frozen, completed):
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'power_domain_trial_diagnostics.csv', index=False)
    summaries = []
    for prediction, part in frame.groupby('prediction'):
        cut = part.closed_endpoint.dropna()
        summaries.append({
            'prediction': prediction, 'allocations': len(part),
            'positive_intervals': int(part.domain_state.eq('positive_interval').sum()),
            'unit_interior': int(part.unit_status.eq('interior').sum()),
            'unit_exterior': int(part.unit_status.eq('outside_mean_support').sum()),
            'endpoint_below_one': int((cut < 1-1e-8).sum()),
            'endpoint_minimum': float(cut.min()),
            'endpoint_median': float(cut.median()),
            'endpoint_maximum': float(cut.max()),
            'existing_selected_power_zero': int(part.selected_power.eq(0).sum()),
            'existing_selected_power_one': int(part.selected_power.eq(1).sum()),
            'existing_selected_support_interior': int(part.selected_support.eq('interior').sum()),
            'maximum_selected_power_replay_error': float(part.selected_power_replay_error.max()),
        })
    pd.DataFrame(summaries).to_csv(out/'power_domain_summary.csv', index=False)
    save(out/'power_domain_exterior_certificates.json', certificates)
    save(out/'power_domain_replay_issues.json', issues)
    save(out/'power_domain_replay_checks.json', {
        'requested_allocations': 500, 'completed_allocations': completed,
        'requested_cases': 1000, 'completed_cases': len(rows),
        'source_hashes': frozen, 'label_rows': 50, 'fixed_public_pool_rows': 1200,
        'prediction_sources': ['AI', 'X_only'],
        'domain_states': frame.domain_state.value_counts().to_dict(),
        'unit_states': frame.unit_status.value_counts().to_dict(),
        'interior_witnesses_independently_checked': int(frame.witness_support.eq('interior').sum()),
        'original_unit_status_mismatches': int(frame.unit_status_matches_original.eq(False).sum()),
        'existing_selected_power_support_interior': int(frame.selected_support.eq('interior').sum()),
        'existing_selected_zero_powers': int(frame.selected_power.eq(0).sum()),
        'maximum_selected_power_replay_error': float(frame.selected_power_replay_error.max()),
        'maximum_primal_residual_or_violation': float(frame.maximum_primal_residual.max()),
        'maximum_endpoint_dual_identity_error': float(frame.endpoint_dual_error.max()),
        'interior_cutoff_certificates': len(certificates),
        'issues': len(issues), 'all_issues_retained': True,
        'new_weight_tuning_rule': False, 'new_human_records': 0,
        'new_AI_responses': 0, 'private_questionnaire_records_used': 0,
        'population_prevalence_or_interval_validity_tested': False,
        'existing_tuning_used_unchanged': True,
        'scope': 'Same 500 source-block allocations, two existing prediction routines, m=50; source blocks do not establish respondent identities.',
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent
    frozen = {name: sha(source/name) for name in [
        'audit_public_power_domain.py', 'power_domain_choice.py',
        'power_tuned_choice_audit.py', 'public_conjoint_matched_audit.py',
        'public_conjoint_optimizer_audit.py']}
    frozen['reference_power_trial_diagnostics.csv'] = sha(args.reference)
    save(args.out/'power_domain_replay_frozen_sources.json', frozen)
    reference = pd.read_csv(args.reference)
    reference = reference[reference.label_rows.eq(50)].set_index(['trial', 'prediction', 'method'])
    assert reference.index.is_unique
    x, y, z, _, _ = load_data(args.input_dir)
    x, y = x[:1200], y[:1200]
    rng = np.random.RandomState(20261109)
    rows, certificates, issues = [], [], []
    for trial in range(500):
        order = (rng.permutation(240)[:, None]*5 + np.arange(5)[None]).reshape(-1)
        xx, yy, zz = x[order], y[order], z[order]
        human, _ = converged(Logistic(xx[:50], yy[:50]))
        for prediction, use_ai in [('AI', True), ('X_only', False)]:
            g, fallback = nuisance(xx, yy, zz, 50, use_ai)
            domain = power_domain(xx, yy, g, 50)
            unit_ref = reference.loc[(trial, prediction, 'ppi_lambda1')]
            tuned_ref = reference.loc[(trial, prediction, 'power_human_init')]
            selected, tuning = power_weight(xx, yy, g, 50, human)
            selected_check = support(objective(xx, yy, g, 50, selected)) if selected is not None else {'status': 'weight_unresolved'}
            expected = {'finite_root': 'interior', 'outside_mean_support': 'outside_mean_support'}.get(unit_ref.status, unit_ref.status)
            row = {
                'trial': trial, 'prediction': prediction, 'label_rows': 50,
                'domain_state': domain['state'],
                'closed_endpoint': domain.get('closed_endpoint'),
                'witness_power': domain.get('interior_witness_power'),
                'witness_margin': domain.get('interior_witness_margin'),
                'unit_status': domain.get('unit_status', 'unresolved'),
                'original_unit_fit_status': unit_ref.status,
                'unit_status_matches_original': domain.get('unit_status') == expected,
                'selected_power': selected,
                'original_selected_power': float(tuned_ref.weight),
                'original_selected_fit_status': tuned_ref.status,
                'selected_support': selected_check['status'],
                'selected_power_replay_error': abs(selected-tuned_ref.weight) if selected is not None else None,
                'maximum_primal_residual': max([v for k, v in domain.items() if 'residual' in k or 'violation' in k], default=0.),
                'endpoint_dual_error': domain.get('endpoint_dual_identity_error', 0.),
                'witness_support': 'not_checked',
            }
            checks = []
            if fallback != 0: checks.append('nuisance_fallback')
            if domain['state'] != 'positive_interval': checks.append('domain_not_verified_positive_interval')
            else:
                witness_check = support(objective(xx, yy, g, 50, domain['interior_witness_power']))
                row['witness_support'] = witness_check['status']
                if witness_check['status'] != 'interior': checks.append('independent_witness_not_interior')
                cut = domain['closed_endpoint']
                if cut < 1-1e-8:
                    bracket = {}
                    for label, power in [('half_endpoint', cut/2), ('below_endpoint', .99*cut), ('at_endpoint', cut), ('above_endpoint', (1+cut)/2)]:
                        check = support(objective(xx, yy, g, 50, power))
                        bracket[label] = {'power': power, 'status': check['status'], 'interior_margin': check.get('interior_margin'), 'separation_gap': check.get('separation_gap')}
                    if bracket['half_endpoint']['status'] != 'interior' or bracket['below_endpoint']['status'] != 'interior': checks.append('endpoint_lower_bracket')
                    if bracket['above_endpoint']['status'] != 'outside_mean_support': checks.append('endpoint_upper_bracket')
                    if bracket['at_endpoint']['status'] != 'boundary_or_near_boundary': checks.append('endpoint_numerical_classification')
                    direction = np.asarray(domain.get('endpoint_direction', []))
                    if direction.shape != (xx.shape[1],): checks.append('missing_endpoint_direction')
                    else:
                        h, a = recession_costs(xx, yy, g, 50, direction)
                        power = (1+cut)/2
                        loss = objective(xx, yy, g, 50, power)
                        recession = float(loss.weights@np.maximum(xx@direction, 0)-loss.moment@direction)
                        identity_error = max(abs(recession-((1-power)*h+power*a)), abs(h/(h-a)-cut), abs(50*h-cut), abs(50*(a-h)+1))
                        if recession >= -1e-8 or identity_error > 1e-7: checks.append('independent_recession_certificate')
                        certificates.append({
                            'trial': trial, 'prediction': prediction,
                            'closed_endpoint': cut, 'finite_interior_power_domain': '(0, closed_endpoint)',
                            'original_unit_fit_status': unit_ref.status,
                            'existing_selected_power': selected, 'existing_selected_fit_status': tuned_ref.status,
                            'endpoint_direction': direction.tolist(), 'human_recession': h, 'unit_recession': a,
                            'above_endpoint_direct_recession': recession,
                            'maximum_independent_identity_error': identity_error,
                            'fixed_power_brackets': bracket,
                            'interpretation': 'Allocation-level public-design certificate; classical parametric LP, not a tuning rule or new respondent sample.',
                        })
            if not row['unit_status_matches_original']: checks.append('original_unit_status_mismatch')
            if selected is None or row['selected_power_replay_error'] > 1e-8: checks.append('selected_power_replay_mismatch')
            if selected_check['status'] != 'interior': checks.append('existing_selected_support_not_interior')
            if checks: issues.append({'trial': trial, 'prediction': prediction, 'checks': checks, 'domain': domain, 'selected_check': selected_check, 'tuning': tuning})
            rows.append(row)
        if (trial+1) % 25 == 0:
            summarize(rows, certificates, issues, args.out, frozen, trial+1)
            print(f'Power-domain allocations complete {trial+1}/500; cases {len(rows)}; retained issues {len(issues)}', flush=True)
    assert sha(args.reference) == frozen['reference_power_trial_diagnostics.csv']
    for name, digest in frozen.items():
        if name.endswith('.py'): assert sha(source/name) == digest
    print('Completed all planned 1000 public-design cases; retained every observed numerical state.', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1): main()
