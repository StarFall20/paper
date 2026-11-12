"""Publication-sized figure and complete table from the matched phase audit."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from variance_reference_existence_audit import ShrunkPopulation,ordinary_boundary


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',type=Path,required=True);p.add_argument('--population',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    state=json.loads((a.phase/'variance_reference_phase_status.json').read_text());assert state['state']=='complete'
    data=pd.read_csv(a.phase/'variance_reference_phase_summary.csv');pop=pd.read_csv(a.population/'variance_reference_population_primary.csv');assert len(data)==32 and data.replications.eq(500).all()
    part=pop[(pop.gamma==1.5)&(pop.epsilon==.1)&(pop.tau==1)]
    boundaries={'ordinary':ordinary_boundary(1.5,200),
        'oracle_variance_reference':float(part[part.prediction_strength==1].critical_ratio.iloc[0]),
        'shrunk_variance_reference':float(part[part.prediction_strength==.5].critical_ratio.iloc[0]),
        'shrunk_fixed_half':ShrunkPopulation(1.5,.1,1.,.5,.5,200).critical_general()['critical_ratio']}
    methods=[('ordinary','Human logit','#555555','o'),('oracle_variance_reference','Oracle prediction, variance power','#0072B2','s'),
        ('shrunk_variance_reference','Shrunk prediction, variance power','#D55E00','^'),('shrunk_fixed_half','Shrunk prediction, fixed power 0.5','#009E73','D')]
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(6.5,3.65),sharey=True)
    for ax,m in zip(axes,[80,160]):
        for name,label,color,marker in methods:
            rows=data[(data.label_rows==m)&data.method.eq(name)].sort_values('dimension_ratio');f=rows.finite_frequency.to_numpy();n=rows.replications.to_numpy()
            z=1.959963984540054;denom=1+z*z/n;center=(f+z*z/(2*n))/denom;half=z*np.sqrt(f*(1-f)/n+z*z/(4*n*n))/denom
            low=np.maximum(0,center-half);high=np.minimum(1,center+half)
            ax.errorbar(rows.dimension_ratio,f,yerr=np.maximum(0,np.array([f-low,high-f])),color=color,marker=marker,markersize=3.5,capsize=2,linewidth=1,label=label)
            ax.axvline(boundaries[name],color=color,linestyle=':',linewidth=.8,alpha=.9)
        ax.set(xlim=(.20,.70),ylim=(-.025,1.025),xticks=[.25,.3625,.45,.65],xlabel='Noise slopes per labelled choice',title=f'{m} labelled choices')
        ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Finite root frequency')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=7.6,bbox_to_anchor=(.5,-.005))
    fig.tight_layout(rect=(0,.17,1,1),w_pad=1.)
    a.out.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.out/'identical_score_phase.png',dpi=300,bbox_inches='tight');fig.savefig(a.out/'identical_score_phase.svg',bbox_inches='tight',metadata={'Date':None});plt.close(fig)
    wide=data.pivot(index=['label_rows','dimension_ratio'],columns='method',values='finite_frequency').reset_index()
    wide.to_csv(a.out/'identical_score_phase_table.csv',index=False)
    lines=['| Labels | Noise slopes per label | Human | Oracle variance | Shrunk variance | Shrunk fixed half |','|---|---|---|---|---|---|']
    for r in wide.itertuples():lines.append(f'| {r.label_rows} | {r.dimension_ratio:.4f} | {r.ordinary:.3f} | {r.oracle_variance_reference:.3f} | {r.shrunk_variance_reference:.3f} | {r.shrunk_fixed_half:.3f} |')
    (a.out/'identical_score_phase_table.md').write_text('\n'.join(lines)+'\n')
    meta=dict(boundaries=boundaries,replications_per_point=500,intervals='95percent Wilson binomial Monte Carlo intervals',
        phase_summary_sha256=hashlib.sha256((a.phase/'variance_reference_phase_summary.csv').read_bytes()).hexdigest(),
        population_sha256=hashlib.sha256((a.population/'variance_reference_population_primary.csv').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        caption='Matched Gaussian draws at gamma1.5,proxy error0.1 andequal sample sizes. The two variance-reference objectives have exactly the same truth score and full fixed-dimensional covariance. Dotted lines are their theoretical boundaries. Connecting lines show measured frequencies and do not estimate a transition rate. Predictions and powers are known population references,not fitted tuning procedures.')
    (a.out/'identical_score_figure_metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)


if __name__=='__main__':main()
