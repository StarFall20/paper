"""Reconcile every nested-budget state and build the manuscript table/figure.

No new fitting or model outcomes. This reads primary records, independent
original-objective classifications and paired transitions separately.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def markdown(frame):
    rows=[list(frame.columns),['---']*len(frame.columns)]+frame.values.tolist()
    return '\n'.join('|'+'|'.join(map(str,r))+'|' for r in rows)+'\n'


def reconcile(root):
    folder=root/'results/radial_budget'
    trials=pd.read_csv(folder/'results/nested_auxiliary_trials.csv')
    independent=pd.read_csv(folder/'verification/independent_support_replays.csv').set_index('row_index')
    record=json.loads((folder/'verification/verification.json').read_text())
    assert len(trials)==len(independent)==12000
    assert independent.index.is_unique
    assert np.array_equal(trials.status.to_numpy(),independent.loc[trials.index,'verified_status'].to_numpy())
    assert independent.distinct_program_first_row.nunique()==10000
    assert trials.status.isin(['feasible','outside']).all()
    assert record['distinct_original_objective_programs']==10000 and record['unknowns']==0
    rays=json.loads((folder/'verification/negative_recession_directions.json').read_text())
    assert len(rays)==1923 and all(r['recession_cost']<-1e-8 for r in rays)
    plan=json.loads((root/'analysis/radial_auxiliary_budget_plan_20261112.json').read_text())
    for rel in ['radial_auxiliary_budget_audit.py','verify_radial_auxiliary_budget.py']:
        reported='source_sha256'
        status=folder/('results/run_status.json' if rel.startswith('radial_auxiliary') else 'verification/verification.json')
        assert hashlib.sha256((root/'analysis'/rel).read_bytes()).hexdigest()==json.loads(status.read_text())[reported]
    assert hashlib.sha256((root/'analysis/radial_auxiliary_budget_plan_20261112.json').read_bytes()).hexdigest()==record['plan_sha256']
    assert hashlib.sha256((folder/'results/nested_auxiliary_trials.csv').read_bytes()).hexdigest()==record['original_trials_sha256']
    counts=trials.groupby(['labels','model','auxiliary_ratio','rule']).status.agg(lambda s:int(s.eq('feasible').sum()))
    sizes=trials.groupby(['labels','model','auxiliary_ratio','rule']).size()
    assert sizes.eq(500).all() and len(sizes)==24
    rows=[]
    for m in plan['label_budgets']:
        for ratio,power_label in zip(plan['auxiliary_ratios'],['1/41','1/2','4/5']):
            vals=[str(m),str(int(round(m*ratio))),power_label]
            vals += [str(counts.loc[m,model,ratio,rule])+'/500' for model,rule in [
                ('constant_scale','variance_rule'),('two_point_scale','variance_rule'),('two_point_scale','held_power')]]
            assert counts.loc[m,'constant_scale',ratio,'held_power']==500
            rows.append(vals)
    expected=pd.DataFrame(rows,columns=['Labels','Auxiliary rows','Variance power','Constant scale','Variable scale','Variable held power'])
    paired=pd.read_csv(folder/'verification/paired_auxiliary_transitions.csv')
    for r in paired.itertuples(index=False):
        part=trials[(trials.labels==r.labels)&(trials.model==r.model)&(trials.rule==r.rule)]
        pivot=part.pivot(index='replication',columns='auxiliary_ratio',values='status')
        a=pivot[r.lower_ratio].eq('feasible').astype(int);b=pivot[r.higher_ratio].eq('feasible').astype(int)
        assert r.finite_to_outside==int(((a==1)&(b==0)).sum())
        assert r.outside_to_finite==int(((a==0)&(b==1)).sum())
        assert abs(r.finite_frequency_change-(b-a).mean())<1e-14
        assert abs(r.paired_Monte_Carlo_SE-(b-a).std(ddof=1)/np.sqrt(len(a)))<1e-14
    return expected,dict(all_reported_states_reconciled=12000,distinct_original_objectives=10000,
        low_ratio_duplicates=2000,negative_recession_directions=1923,full_cells=24,
        independent_cohorts=1000,paired_transition_rows=len(paired),unknowns=0)


def plot(root,destination):
    frame=pd.read_csv(root/'results/radial_budget/results/population_budget_boundaries.csv')
    assert len(frame)==52
    fig,axes=plt.subplots(1,2,figsize=(6.5,3.45),layout='constrained')
    for model,rule,color,style,label in [
        ('constant_scale','variance_rule','#1f5d92','-','Constant scale'),
        ('two_point_scale','variance_rule','#b84034','-','Variable scale'),
        ('two_point_scale','held_power','#518543','--','Variable scale fixed power')]:
        part=frame[(frame.model==model)&(frame.rule==rule)].sort_values('auxiliary_ratio')
        assert len(part)==13
        axes[0].plot(part.auxiliary_ratio,part.critical_ratio,color=color,ls=style,lw=1.6,label=label)
    axes[0].axhline(.25,color='#777777',ls=':',lw=1.1,label='Dimension ratio 0.25')
    axes[0].set(xscale='log',xlabel='Auxiliary / labelled rows',ylabel='Critical noise / labels',
                title='A  Finite estimate boundary',ylim=(0,1.2))
    axes[0].legend(loc='upper left',fontsize=8,frameon=False,handlelength=1.7)
    part=frame[(frame.model=='two_point_scale')&(frame.rule=='variance_rule')].sort_values('auxiliary_ratio')
    same=frame[(frame.model=='constant_scale')&(frame.rule=='variance_rule')].sort_values('auxiliary_ratio')
    assert np.allclose(part.regular_covariance_times_labels,same.regular_covariance_times_labels)
    axes[1].plot(part.auxiliary_ratio,part.regular_covariance_times_labels,color='#6d5196',lw=1.8,label='Both scale laws')
    axes[1].axhline(4,color='#777777',ls=':',lw=1.1,label='Labelled only')
    axes[1].set(xscale='log',xlabel='Auxiliary / labelled rows',ylabel='Oracle covariance factor',
                title='B  Regular local covariance',ylim=(2.85,4.12))
    axes[1].legend(loc='lower left',fontsize=8,frameon=False)
    for ax in axes:
        ax.grid(axis='y',alpha=.16);ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.xaxis.label.set_size(9);ax.yaxis.label.set_size(9);ax.title.set_size(9.5)
    destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,dpi=220);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--package-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--write-assets',action='store_true');args=parser.parse_args()
    root=args.package_root;args.out.mkdir(parents=True,exist_ok=True)
    table,report=reconcile(root);text=markdown(table)
    table.to_csv(args.out/'table6.csv',index=False);(args.out/'table6.md').write_text(text)
    if args.write_assets:
        table.to_csv(root/'manuscript/table6.csv',index=False)
        (root/'manuscript/table6.md').write_text(text)
        plot(root,root/'manuscript/figures/figure_radial_auxiliary.png')
    else:
        assert table.equals(pd.read_csv(root/'manuscript/table6.csv',dtype=str))
        assert (root/'manuscript/table6.md').read_text()==text
    assert text.strip() in (root/'manuscript/paper.md').read_text()
    report.update(date='2026-11-13',table6_reconciled=True,priority_or_readiness_certified=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out/'radial_publication_checked.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
