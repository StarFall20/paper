"""Reconcile new manuscript loss tables to independently verified path records.

No fitting, new outcomes, or hypothesis tests. Paired public denominators and
conditional simulation denominators are checked separately against all records.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd


def outputs(root):
    folder = root / 'results/probability_risk/verified_summary'
    pairs = pd.read_csv(folder / 'all_paired_summary_verified.csv')
    summary = pd.read_csv(folder / 'all_path_summary_verified.csv')
    public = pd.read_csv(folder / 'public/path_records.csv')
    gaussian = pd.read_csv(folder / 'gaussian/path_records.csv')
    assert len(public) == 720 and len(gaussian) == 1200
    assert public.fit_status.eq('numerical_finite_root').sum() == 574
    assert gaussian.fit_status.eq('numerical_finite_root').sum() == 827
    public_rows = []
    for prediction, label in [('AI', 'AI aided'), ('X_only', 'Attributes only')]:
        part = public[public.prediction.eq(prediction)]
        base = part[part.power.eq(.5)].set_index('case')
        end = part[part.power.eq(.0002)].set_index('case')
        shared = [k for k in base.index.intersection(end.index)
                  if base.loc[k, 'fit_status'] == end.loc[k, 'fit_status'] == 'numerical_finite_root']
        assert len(base) == len(end) == 60 and len(shared) == 47
        row = pairs[(pairs['mode'] == 'public') & pairs.prediction.eq(prediction) & pairs.power.eq(.0002)].iloc[0]
        assert row.shared_finite == 47
        a = base.loc[shared, 'test_log_loss'].to_numpy(float)
        b = end.loc[shared, 'test_log_loss'].to_numpy(float)
        for computed, field in [(a.mean(), 'mean_reference_test_log_loss'),
                                (b.mean(), 'mean_endpoint_test_log_loss'),
                                (np.median(a), 'median_reference_test_log_loss'),
                                (np.median(b), 'median_endpoint_test_log_loss')]:
            assert abs(computed-row[field]) < 1e-11
        assert int((b > a).sum()) == 46 and int((b < a).sum()) == 1
        public_rows.append([label, '47/60', f'{a.mean():.4f}', f'{b.mean():.4f}',
                            f'{np.median(a):.4f}', f'{np.median(b):.4f}'])
    table5 = pd.DataFrame(public_rows, columns=['Prediction', 'Shared fits',
        'Mean at 0.5', 'Mean at 0.0002', 'Median at 0.5', 'Median at 0.0002'])
    simulation_rows = []
    for m in [80, 160]:
        for power in [.5, .1, .0002]:
            part = gaussian[gaussian.label_rows.eq(m) & gaussian.power.eq(power)]
            finite = part[part.fit_status.eq('numerical_finite_root')]
            assert len(part) == 100
            row = summary[(summary['mode'] == 'gaussian') & summary.label_rows.eq(m) & summary.power.eq(power)].iloc[0]
            for computed, field in [(finite.test_log_loss.mean(), 'mean_test_log_loss'),
                                    (finite.test_log_loss.median(), 'median_test_log_loss'),
                                    (finite.test_Brier_loss.mean(), 'mean_test_Brier_loss')]:
                assert abs(computed-row[field]) < 1e-11
            assert finite.test_classification_error.eq(.5).all()
            simulation_rows.append([str(m), f'{power:g}', f'{len(finite)}/100',
                f'{finite.test_log_loss.mean():.4f}', f'{finite.test_log_loss.median():.4f}',
                f'{finite.test_Brier_loss.mean():.6f}'])
    tableD1 = pd.DataFrame(simulation_rows, columns=['Labels', 'Power', 'Finite',
        'Mean log loss', 'Median log loss', 'Mean Brier'])
    return {'table5': table5, 'tableD1': tableD1}


def markdown(frame):
    values = [list(frame.columns), ['---']*len(frame.columns)] + frame.values.tolist()
    return '\n'.join('|'+'|'.join(map(str, row))+'|' for row in values)+'\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--package-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--write-manuscript-tables', action='store_true')
    args = parser.parse_args()
    root = args.package_root
    args.out.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs(root).items():
        text = markdown(frame)
        frame.to_csv(args.out/(name+'.csv'), index=False)
        (args.out/(name+'.md')).write_text(text)
        if args.write_manuscript_tables:
            frame.to_csv(root/'manuscript'/(name+'.csv'), index=False)
            (root/'manuscript'/(name+'.md')).write_text(text)
        else:
            retained = pd.read_csv(root/'manuscript'/(name+'.csv'), dtype=str)
            assert retained.equals(frame)
            assert (root/'manuscript'/(name+'.md')).read_text() == text
            assert text.strip() in (root/'manuscript/paper.md').read_text()
    record = dict(date='2026-11-12',planned_path_records=1920,finite_numerical_fits=1401,
        paired_public_shared_fits_per_routine=47,public_allocations_per_routine=60,
        simulation_draws_per_label_budget=100,corrected_Gaussian_metrics_used=True,
        raw_records_or_coefficients_changed=False,public_MCSE_claimed=False,
        tables_reconciled=['5','D1'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out/'probability_risk_tables_checked.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
