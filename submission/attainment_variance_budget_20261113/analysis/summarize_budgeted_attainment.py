"""Reconcile population budgets and verified illustration, including controls."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(a.root/'results/budget_population.csv');checked=pd.read_csv(a.root/'verification/population_checks.csv').set_index('cell_id');assert len(frame)==len(checked)==195
    assert max(abs(frame.optimized_capacity.to_numpy()-checked.loc[frame.cell_id,'integrated_capacity'].to_numpy()))<1e-8
    phase=pd.read_csv(a.root/'phase_verification/phase_trials_verified.csv');original=pd.read_csv(a.root/'phase/phase_trials.csv');independent=pd.read_csv(a.root/'phase_verification/original_loss_checks.csv').set_index('row_index')
    assert len(phase)==len(original)==len(independent)==5000
    assert np.array_equal(phase.status,independent.loc[phase.index,'verified_status'])
    known=original.status!='unknown';assert np.array_equal(original.loc[known,'status'],phase.loc[known,'status'])
    unresolved_count=int((~known).sum())
    assert phase.loc[~known,'status'].isin(['human_finite','human_separated','feasible','outside']).all()
    resolved_human_interior=int(((~known)&phase.status.eq('human_finite')).sum())
    assert not phase.status.eq('unknown').any()
    summary=pd.read_csv(a.root/'phase_verification/phase_summary_verified.csv');counts=phase.groupby(['labels','method']).status.agg(lambda s:int(s.isin(['feasible','human_finite']).sum()));assert len(counts)==10
    for row in summary.itertuples(index=False):assert row.replications==500 and row.finite==counts.loc[row.labels,row.method]
    paired=pd.read_csv(a.root/'phase_verification/paired_transitions_verified.csv')
    for row in paired.itertuples(index=False):
        pivot=phase[phase.labels==row.labels].pivot(index='replication',columns='method',values='status').isin(['feasible','human_finite']).astype(int);before=pivot[row.first_method];after=pivot[row.second_method];diff=after-before
        assert row.finite_to_outside==int(((before==1)&(after==0)).sum()) and row.outside_to_finite==int(((before==0)&(after==1)).sum())
        assert abs(row.finite_frequency_change-diff.mean())<1e-14 and abs(row.paired_Monte_Carlo_SE-diff.std(ddof=1)/np.sqrt(len(diff)))<1e-14
    selected=frame.loc[frame.cell_id=='five_level_tau0.25_C1.05'].iloc[0];covariance=4*(1-selected.power*.25)
    values={'human_only':4.,'unweighted_compensation':covariance,'power_at_budget':covariance*selected.power_variance,'optimized_at_budget':covariance*selected.optimized_variance,'inverse_scale':covariance*selected.inverse_scale_cost}
    order=['human_only','unweighted_compensation','power_at_budget','optimized_at_budget','inverse_scale'];table=[]
    for method in order:table.append(dict(method=method,finite_m80=f'{counts.loc[80,method]}/500',finite_m160=f'{counts.loc[160,method]}/500',regular_covariance_factor=values[method]))
    pd.DataFrame(table).to_csv(a.out/'verified_illustration_table.csv',index=False)
    gains=frame[frame.scale_levels>=3].copy();gains.to_csv(a.out/'all_multilevel_budget_gains.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.1),layout='constrained')
    for model,color,label in [('previous_two_point','#777777','Two levels'),('four_level_mild','#0072B2','Mild four levels'),('three_level','#009E73','Three levels'),('five_level','#D55E00','Five levels')]:
        part=frame[(frame.model==model)&(frame.auxiliary_ratio==.25)&frame.variance_budget.isin([1,1.01,1.05,1.1,1.25,1.5,2])].sort_values('variance_budget')
        axes[0].plot(part.variance_budget,part.capacity_gain,marker='o',markersize=3,color=color,label=label)
    axes[0].set(xlabel='Regular covariance budget C',ylabel='Boundary gain over the power path');axes[0].set_title('A  Same-budget population comparison',fontsize=10);axes[0].grid(alpha=.2);axes[0].legend(fontsize=8)
    x=np.arange(5);width=.35
    for offset,m,color in [(-width/2,80,'#0072B2'),(width/2,160,'#D55E00')]:
        freq=np.array([counts.loc[m,method]/500 for method in order]);z=1.959963984540054;n=500;center=(freq+z*z/(2*n))/(1+z*z/n);radius=z*np.sqrt(freq*(1-freq)/n+z*z/(4*n*n))/(1+z*z/n);lower=np.maximum(0,center-radius);upper=np.minimum(1,center+radius)
        axes[1].bar(x+offset,freq,width,color=color,label=f'{m} labels',yerr=np.maximum(0,np.vstack([freq-lower,upper-freq])),capsize=2)
    axes[1].set(xticks=x,xticklabels=['Labelled\nonly','Unweighted','Power','Optimized','Inverse\nscale'],ylabel='Finite-estimate frequency',ylim=(0,1.15));axes[1].set_title('B  Selected illustration, all controls',fontsize=10);axes[1].legend(fontsize=8,loc='upper right');axes[1].tick_params(axis='x',labelsize=8)
    fig.savefig(a.out/'budgeted_attainment.png',dpi=220);fig.savefig(a.out/'budgeted_attainment.svg');plt.close(fig)
    result=dict(date='2026-11-13',population_conditions=195,raw_phase_records=5000,verified_phase_records=5000,previous_cohorts=1000,raw_undetermined_states_preserved=unresolved_count,undetermined_human_states_resolved_as_interior=resolved_human_interior,paired_transition_rows_reconciled=len(paired),selected_cell='five_level_tau0.25_C1.05',population_capacity_gain=float(selected.capacity_gain),table_rows=table,
        figure_scope='PanelA uses all seven standard budgets at auxiliary ratio0.25; the complete195conditions and model-specific inverse endpoints remain archived. PanelB uses all five methods and both label budgets, with marginal95%Wilson Monte Carlo intervals.',
        ordinary_baseline_more_finite_in_this_illustration=True,estimator_superiority_claimed=False,high_dimensional_risk_gain_claimed=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.out/'summary_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
