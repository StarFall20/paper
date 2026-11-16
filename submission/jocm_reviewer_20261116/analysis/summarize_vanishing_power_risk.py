"""Summarize independently verified probability-loss paths without new fits.

Raw statuses and metrics remain in the original reports. This aggregation uses
the resolved support states and corrected Gaussian quadrature. Public repeated
allocations receive no Monte Carlo standard errors or participant interpretation.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from vanishing_power_risk_audit import tables


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root, out):
    out.mkdir(parents=True, exist_ok=True)
    all_summary, all_pairs, inputs = [], [], {}
    total_roots = total_rows = 0
    for mode in ['public', 'gaussian']:
        source = root / 'verification_final' / mode / 'path_records_verified.csv'
        frame = pd.read_csv(source)
        inputs[str(source.relative_to(root))] = sha(source)
        frame['initial_fit_status'] = frame['fit_status']
        frame['initial_support_status'] = frame['support_status']
        exterior = frame.verified_support_status.eq('outside_mean_support')
        assert not (exterior & frame.fit_status.eq('numerical_finite_root')).any()
        frame.loc[exterior, 'fit_status'] = 'not_fitted_outside_mean_support'
        frame['support_status'] = frame['verified_support_status']
        assert frame.support_status.isin(['interior', 'outside_mean_support']).all()
        assert frame.loc[~exterior, 'fit_status'].eq('numerical_finite_root').all()
        assert len(frame) == (720 if mode == 'public' else 1200)
        assert not frame.duplicated(['case', 'prediction', 'power']).any()
        folder = out / mode
        folder.mkdir(parents=True, exist_ok=True)
        tables(frame.to_dict('records'), folder)
        summary = pd.read_csv(folder / 'path_summary.csv')
        pairs = pd.read_csv(folder / 'paired_to_half_power.csv')
        for idx, row in summary.iterrows():
            part = frame[(frame.label_rows == row.label_rows) &
                         (frame.prediction == row.prediction) &
                         (frame.power == row.power)]
            finite = part[part.fit_status.eq('numerical_finite_root')]
            summary.loc[idx, 'outside'] = int(part.support_status.eq('outside_mean_support').sum())
            summary.loc[idx, 'finite_fraction'] = len(finite) / len(part)
            for field in ['test_log_loss', 'test_Brier_loss', 'coefficient_norm',
                          'ridge_test_log_loss', 'ridge_test_Brier_loss']:
                summary.loc[idx, 'median_' + field] = finite[field].median()
                if field.startswith('ridge_'):
                    summary.loc[idx, 'mean_' + field] = finite[field].mean()
        for idx, row in pairs.iterrows():
            group = frame[(frame.label_rows == row.label_rows) &
                          (frame.prediction == row.prediction)]
            base = group[group.power == .5].set_index('case')
            end = group[group.power == row.power].set_index('case')
            shared = [case for case in base.index.intersection(end.index)
                      if base.loc[case, 'fit_status'] == 'numerical_finite_root'
                      and end.loc[case, 'fit_status'] == 'numerical_finite_root']
            assert len(shared) == int(row.shared_finite)
            for field in ['test_log_loss', 'test_Brier_loss', 'coefficient_norm']:
                a = base.loc[shared, field].to_numpy(float)
                b = end.loc[shared, field].to_numpy(float)
                delta = b - a
                for name, values in [('reference', a), ('endpoint', b)]:
                    pairs.loc[idx, f'mean_{name}_{field}'] = values.mean()
                    pairs.loc[idx, f'median_{name}_{field}'] = np.median(values)
                pairs.loc[idx, 'median_difference_' + field] = np.median(delta)
                pairs.loc[idx, 'increased_' + field] = int((delta > 1e-10).sum())
                pairs.loc[idx, 'decreased_' + field] = int((delta < -1e-10).sum())
                pairs.loc[idx, 'unchanged_' + field] = int((abs(delta) <= 1e-10).sum())
            pairs.loc[idx, 'mean_shared_ridge_log_loss'] = base.loc[shared, 'ridge_test_log_loss'].mean()
            pairs.loc[idx, 'median_shared_ridge_log_loss'] = base.loc[shared, 'ridge_test_log_loss'].median()
        summary.to_csv(folder / 'path_summary_verified.csv', index=False)
        pairs.to_csv(folder / 'paired_summary_verified.csv', index=False)
        all_summary.append(summary)
        all_pairs.append(pairs)
        total_roots += int(frame.fit_status.eq('numerical_finite_root').sum())
        total_rows += len(frame)
    combined_summary = pd.concat(all_summary, ignore_index=True)
    combined_pairs = pd.concat(all_pairs, ignore_index=True)
    combined_summary.to_csv(out / 'all_path_summary_verified.csv', index=False)
    combined_pairs.to_csv(out / 'all_paired_summary_verified.csv', index=False)
    status = dict(date='2026-11-12', planned_records=total_rows,
                  finite_root_records=total_roots, outside_records=total_rows-total_roots,
                  unresolved=0, verified_inputs_sha256=inputs,
                  source_sha256=sha(Path(__file__)), original_records_modified=False,
                  public_MCSE_or_participant_inference=False,
                  paired_denominators_explicit=True, corrected_Gaussian_metrics_used=True)
    (out / 'summary_checks.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.out)
