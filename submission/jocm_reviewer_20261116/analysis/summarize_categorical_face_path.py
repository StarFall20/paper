"""Summaries of all categorical face paths and a paired descriptive figure."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(args):
    assert not args.out.exists(), 'Use a fresh summary directory.'
    args.out.mkdir(parents=True)
    verification = json.loads((args.verification/'verification.json').read_text())
    assert verification['state']=='passed'
    for name, expected in verification['input_results_sha256'].items():
        assert hashlib.sha256((args.results/name).read_bytes()).hexdigest()==expected
    faces = pd.read_csv(args.results/'face_summary.csv')
    selectors = pd.read_csv(args.results/'selector_summary.csv')
    paths = pd.read_csv(args.results/'path_records.csv')
    assert len(faces)==60 and len(selectors)==120 and len(paths)==720
    assert paths.distance_inf.notna().sum()==verification['finite_roots']==574
    faces.groupby('type', as_index=False).agg(allocations=('trial','size'),
        min_exposed=('exposed_rows','min'), max_exposed=('exposed_rows','max')).to_csv(args.out/'face_counts.csv', index=False)
    summaries = []
    for key, group in paths.groupby(['face_type', 'prediction', 'power']):
        finite = group[group.distance_inf.notna()]
        row = dict(zip(['face_type','prediction','power'], key))
        row.update(planned=len(group), finite=len(finite))
        for field in ['distance_inf', 'normalized_distance_inf', 'neutral_margin_max',
                      'exposed_margin_deficit','scaled_test_logloss','scaled_test_hinge']:
            row['median_'+field] = float(finite[field].median()) if len(finite) else None
            row['q25_'+field] = float(finite[field].quantile(.25)) if len(finite) else None
            row['q75_'+field] = float(finite[field].quantile(.75)) if len(finite) else None
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(args.out/'all_path_summary.csv', index=False)
    paired = []; common = []; curves = []
    for (face, pred), group in paths.groupby(['face_type','prediction']):
        pivot = group.pivot(index='trial', columns='power', values='distance_inf')
        complete = pivot.dropna().index
        part = group[group.trial.isin(complete)]
        common.append(part)
        for power, g in part.groupby('power'):
            curves.append(dict(face_type=face, prediction=pred, power=power, shared=len(complete),
                median=float(g.distance_inf.median()), q25=float(g.distance_inf.quantile(.25)),
                q75=float(g.distance_inf.quantile(.75))))
        delta = pivot.loc[complete, .0002]-pivot.loc[complete, .5]
        paired.append(dict(face_type=face, prediction=pred, planned=len(pivot),
            shared_all_six=len(complete), mean_initial=float(pivot.loc[complete,.5].mean()),
            mean_endpoint=float(pivot.loc[complete,.0002].mean()),
            median_initial=float(pivot.loc[complete,.5].median()),
            median_endpoint=float(pivot.loc[complete,.0002].median()),
            decreased=int((delta < -1e-9).sum()), increased=int((delta > 1e-9).sum()),
            unchanged=int((abs(delta)<=1e-9).sum())))
    pd.DataFrame(paired).to_csv(args.out/'paired_distance_endpoints.csv', index=False)
    curves = pd.DataFrame(curves); curves.to_csv(args.out/'common_cohort_plot_data.csv', index=False)
    pd.concat(common).to_csv(args.out/'common_cohort_records.csv', index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'svg.hashsalt':'categorical_face_20261113'})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True)
    for ax, (face, title) in zip(axes, [('complete','Complete separation'),('quasi','Quasi-complete separation'),('overlap','Overlap')]):
        for pred, color, marker in [('AI','#22649b','o'),('X_only','#b86324','s')]:
            data = curves[(curves.face_type==face)&(curves.prediction==pred)].sort_values('power', ascending=False)
            xx = np.log(1/data.power.to_numpy())
            ax.plot(xx, data['median'], marker=marker, color=color,
                    label=('AI aided' if pred=='AI' else 'Attributes only')+f" (n={data.shared.iloc[0]})")
            ax.fill_between(xx, data.q25, data.q75, color=color, alpha=.10)
        ax.set(title=title, xlabel='log(1 / correction power)', ylabel='Distance to optimal selector set')
        ax.set_ylim(bottom=0); ax.grid(axis='y', alpha=.18)
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].legend(fontsize=8, frameon=False)
    axes[2].legend(fontsize=8, frameon=False)
    fig.savefig(args.out/'categorical_face_paths.png', dpi=180)
    fig.savefig(args.out/'categorical_face_paths.svg', metadata={'Date':None})
    plt.close(fig)
    checks = dict(state='passed', allocations=60, predictor_allocations=120, paths=720,
        finite=574, unavailable=146, rows_in_common_cohorts=sum(len(p) for p in common),
        figure='Medians and interquartile ranges on each routine/type common six-power cohort; descriptive intervals, not uncertainty intervals.',
        raw_distance='Infinity distance to the whole LP face with objective slack 1e-8.',
        normalized_distance='Optional recorded distance divided by 1+the infinity norm of a nearest selector returned by the program; raw distance is primary.',
        complete_or_quasi_face_is_new=False, generic_margin_limit_is_new=False,
        asymptotic_limit_established_by_finite_grid=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out/'summary_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--verification', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
