"""Reconcile paired denominators and prepare the whole-menu quality table.

Availability strata are a secondary descriptive analysis after the fixed full
follow-up. All strata and all six existing conditions are retained.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


def run(results, out):
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(results/'quality_records.csv', float_precision='round_trip')
    pairs = pd.read_csv(results/'paired_with_ridge.csv', float_precision='round_trip')
    fallback = pd.read_csv(results/'ridge_fallback.csv', float_precision='round_trip')
    assert len(data) == 15000
    measures = ['squared_coefficient_error', 'coefficient_norm', 'human_log_loss',
                'excess_human_log_loss', 'human_Brier_loss', 'human_classification_error']
    table, strata = [], []
    max_error = 0.
    for (m, q), group in data.groupby(['label_menus', 'orthogonal_slopes']):
        by = {method: part.set_index('trial') for method, part in group.groupby('method')}
        ridge, ref, human, pool = (by[name] for name in ['human_ridge', 'oracle_null_slope_variance', 'ordinary', 'hard_pool'])
        assert ridge.fit_status.eq('numerical_finite_root').all() and len(ridge) == 500
        ref_ok = ref.fit_status.eq('numerical_finite_root')
        human_ok = human.fit_status.eq('numerical_finite_root')
        pool_ok = pool.fit_status.eq('numerical_finite_root')
        common = ref.index[ref_ok & pool_ok]
        table.append(dict(Labels=int(m), Slopes=int(q), Shared=f'{len(common)}/500',
                          Corrected=f'{ref.loc[common,"human_log_loss"].median():.4f}',
                          Ridge=f'{ridge.loc[common,"human_log_loss"].median():.4f}',
                          Pooling=f'{pool.loc[common,"human_log_loss"].median():.4f}'))
        for name, mask in [('both_finite', ref_ok & human_ok),
                           ('corrected_only', ref_ok & ~human_ok),
                           ('human_only', ~ref_ok & human_ok),
                           ('neither_finite', ~ref_ok & ~human_ok)]:
            ids = ref.index[mask]
            row = dict(label_menus=int(m), orthogonal_slopes=int(q), stratum=name, allocations=len(ids))
            for method in ['oracle_null_slope_variance', 'ordinary', 'human_ridge', 'hard_pool']:
                good = [i for i in ids if by[method].loc[i, 'fit_status'] == 'numerical_finite_root']
                row[method+'_available'] = len(good)
                row[method+'_mean_log_loss'] = float(by[method].loc[good, 'human_log_loss'].mean()) if good else None
            strata.append(row)
        for method, part in by.items():
            ids = part.index[part.fit_status.eq('numerical_finite_root')]
            saved = pairs[(pairs.label_menus==m)&(pairs.orthogonal_slopes==q)&pairs.method.eq(method)].iloc[0]
            assert saved.paired_allocations == 500 and saved.shared_numerical_finite == len(ids)
            for measure in measures:
                if len(ids):
                    difference = part.loc[ids, measure].to_numpy()-ridge.loc[ids, measure].to_numpy()
                    max_error = max(max_error, abs(float(saved['mean_difference_'+measure])-float(difference.mean())))
                    if len(ids)>1:
                        se = difference.std(ddof=1)/np.sqrt(len(difference))
                        max_error = max(max_error, abs(float(saved['paired_MCSE_'+measure])-float(se)))
            if method in ['ordinary', 'oracle_null_slope_variance', 'unit']:
                risk = ridge.human_log_loss.copy()
                risk.loc[ids] = part.loc[ids, 'human_log_loss']
                changes = risk-ridge.human_log_loss
                saved = fallback[(fallback.label_menus==m)&(fallback.orthogonal_slopes==q)&fallback.method.eq(method)].iloc[0]
                assert saved.evaluated == 500 and saved.direct_method_used == len(ids) and saved.ridge_fallbacks == 500-len(ids)
                max_error = max(max_error, abs(float(saved.mean_human_log_loss)-float(risk.mean())),
                                abs(float(saved.mean_difference_from_ridge)-float(changes.mean())),
                                abs(float(saved.paired_MCSE)-float(changes.std(ddof=1)/np.sqrt(500))))
    assert max_error < 1e-10
    pd.DataFrame(table).to_csv(out/'tableF3.csv', index=False)
    headers = list(table[0])
    block = '\n'.join(['|'+'|'.join(headers)+'|', '|'+'|'.join(['---']*len(headers))+'|']+
                       ['|'+'|'.join(str(row[h]) for h in headers)+'|' for row in table])+'\n'
    (out/'tableF3.md').write_text(block)
    pd.DataFrame(strata).to_csv(out/'all_availability_strata.csv', index=False)
    report = dict(state='passed', original_conditions=6, table_rows=len(table), paired_rows=len(pairs),
                  fallback_rows=len(fallback), all_availability_strata=len(strata),
                  maximum_paired_or_fallback_error=max_error,
                  stratum_analysis='Secondary descriptive analysis after observing primary risk outputs; every stratum retained.',
                  table_denominator='Shared numerical finite fits for corrected, ridge and pooling in each cell.',
                  original_primary_records_changed=False)
    (out/'summary_verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a=p.parse_args()
    run(a.results, a.out)
