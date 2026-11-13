"""Reconcile the row-weight comparison, check local cost, and plot evidence."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(a.root/'results/weight_path_trials.csv');verified=pd.read_csv(a.root/'verification/weight_path_trials_verified.csv');replay=pd.read_csv(a.root/'verification/independent_support_replays.csv').set_index('row_index')
    assert len(raw)==len(verified)==len(replay)==5000 and np.array_equal(raw.status,verified.status)
    assert np.array_equal(raw.status,replay.loc[raw.index,'verified_status'])
    summary=pd.read_csv(a.root/'results/weight_path_summary.csv')
    counts=verified.groupby(['labels','row_exponent']).status.agg(lambda s:int(s.eq('feasible').sum()))
    sizes=verified.groupby(['labels','row_exponent']).size();assert sizes.eq(500).all()
    low=.1;high=np.sqrt(9.91);d=np.array([low,high]);mass=np.array([.9,.1])
    def variance(r):return float((mass@d**(2-2*r))/(mass@d**(2-r))**2)
    table=[]
    for r in sorted(raw.row_exponent.unique()):
        table.append(dict(row_exponent=float(r),finite_m80=f'{counts.loc[80,r]}/500',finite_m160=f'{counts.loc[160,r]}/500',regular_covariance_multiplier=variance(r)))
    pd.DataFrame(table).to_csv(a.out/'weight_tradeoff_table.csv',index=False)
    transitions=pd.read_csv(a.root/'results/paired_weight_transitions.csv')
    for row in transitions.itertuples(index=False):
        part=verified[verified.labels==row.labels].pivot(index='replication',columns='row_exponent',values='status')
        before=part[row.lower_exponent].eq('feasible').astype(int);after=part[row.higher_exponent].eq('feasible').astype(int);diff=after-before
        assert row.finite_to_outside==int(((before==1)&(after==0)).sum())
        assert row.outside_to_finite==int(((before==0)&(after==1)).sum())
        assert abs(row.finite_frequency_change-diff.mean())<1e-14
        assert abs(row.paired_Monte_Carlo_SE-diff.std(ddof=1)/np.sqrt(len(diff)))<1e-14
    population=pd.read_csv(a.root/'results/population_weight_path.csv');cost_checks=[]
    for key,part in population.groupby(['model','auxiliary_ratio']):
        model,tau=key;first=part[part.row_exponent==0].iloc[0]
        if model=='constant':ds=np.ones(1);ps=np.ones(1)
        else:
            small=first.small_scale;p=first.large_probability;ds=np.array([small,np.sqrt((1-(1-p)*small**2)/p)]);ps=np.array([1-p,p])
        f=lambda r:float((ps@ds**(2-2*r))/(ps@ds**(2-r))**2)
        h=.001;quadratic=(f(h)+f(-h)-2*f(0))/(2*h*h)
        error=abs(quadratic-first.small_exponent_quadratic_cost);assert error<1e-5
        for row in part.itertuples(index=False):
            step=1e-5;fd=(f(row.row_exponent+step)-f(row.row_exponent-step))/(2*step)
            assert abs(fd-row.covariance_multiplier_derivative)<1e-6
        cost_checks.append(dict(model=model,auxiliary_ratio=tau,quadratic_coefficient=first.small_exponent_quadratic_cost,independent_symmetric_difference=quadratic,discrepancy=error))
    pd.DataFrame(cost_checks).to_csv(a.out/'local_variance_cost_checks.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.8),layout='constrained')
    for model,color,label in [('mild','#0072B2','Mild scale variation'),('moderate','#009E73','Moderate variation'),('previous_two_point','#D55E00','Previous two-point scale')]:
        part=population[(population.model==model)&(population.auxiliary_ratio==1)].sort_values('row_exponent')
        axes[0].plot(part.regular_covariance_multiplier,part.critical_ratio,color=color,label=label)
        axes[0].scatter(part.iloc[[0,-1]].regular_covariance_multiplier,part.iloc[[0,-1]].critical_ratio,color=color,s=20)
    axes[0].axhline(.25,color='#777777',linestyle=':',linewidth=1)
    axes[0].set(xlabel='Regular covariance multiplier',ylabel='Critical noise-to-label ratio')
    axes[0].set_title('A  Population boundary and covariance',fontsize=10)
    axes[0].legend(fontsize=7.5,loc='lower right');axes[0].grid(alpha=.2)
    for m,color in [(80,'#0072B2'),(160,'#D55E00')]:
        part=summary[summary.labels==m].sort_values('row_exponent');freq=part.finite_frequency.to_numpy();n=500;z=1.959963984540054
        center=(freq+z*z/(2*n))/(1+z*z/n);radius=z*np.sqrt(freq*(1-freq)/n+z*z/(4*n*n))/(1+z*z/n)
        lower=np.maximum(0,center-radius);upper=np.minimum(1,center+radius)
        axes[1].errorbar(part.row_exponent,freq,yerr=np.maximum(0,np.vstack([freq-lower,upper-freq])),color=color,marker='o',markersize=4,capsize=3,label=f'{m} labelled rows')
    axes[1].set(xlabel='Fixed row-weight exponent r',ylabel='Finite-estimate frequency',ylim=(-.05,1.08))
    axes[1].set_title('B  Finite estimates on previous cohorts',fontsize=10)
    axes[1].set_xticks([0,.25,.5,.75,1]);axes[1].legend(fontsize=8,loc='lower right');axes[1].grid(alpha=.2)
    fig.savefig(a.out/'weight_tradeoff.png',dpi=220);fig.savefig(a.out/'weight_tradeoff.svg');plt.close(fig)
    result=dict(date='2026-11-13',retained_records=5000,source_cohorts=1000,paired_transition_rows_checked=len(transitions),covariance_local_cost_checks=len(cost_checks),maximum_quadratic_cost_discrepancy=max(x['discrepancy'] for x in cost_checks),table_rows=table,figure_intervals='Marginal 95% Wilson Monte Carlo intervals over 500 independent cohorts per label budget; paired transition uncertainty is reported separately.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),population_curves_not_finite_sample_risk_guarantees=True)
    (a.out/'summary_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
