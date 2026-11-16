"""Build the diagnostic figure from retained tables, without model generation."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def wilson(k,n):
    p=k/n;z=1.959963984540054
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return center-half,center+half


def build(base,out):
    out.mkdir(parents=True,exist_ok=True)
    envelope=pd.read_csv(base/'envelope/all_power_envelopes.csv')
    risk=pd.read_csv(base/'envelope/learned_prediction_risks.csv')
    audit=pd.read_csv(base/'all_power/summary.csv')
    direct=pd.read_csv(base/'phase/phase_summary.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(10.2,7.8))
    colors={.3:'#245a81',.45:'#a64a24'}
    for rho in [.3,.45]:
        sub=envelope[envelope.total_slope_ratio.eq(rho)&envelope.auxiliary_per_total_label.eq(1)].sort_values('training_fraction')
        axes[0,0].plot(sub.training_fraction,(1-sub.training_fraction)*sub.envelope-rho,'o-',color=colors[rho],label=f'Slopes / total labels = {rho:g}')
        rr=risk[risk.total_slope_ratio.eq(rho)].sort_values('training_fraction')
        axes[0,1].plot(rr.training_fraction,rr.classification_error,'o-',color=colors[rho])
    axes[0,0].axhline(0,color='#555555',ls=':',lw=1)
    axes[0,0].set(title='a  All-power budget slack',ylabel='Existence slack per total label')
    axes[0,1].set(title='b  Trained predictor error',ylabel='Limiting classification error')
    axes[0,0].legend(fontsize=8,loc='lower left',frameon=False)
    for ax,rho in zip(axes[1],[.3,.45]):
        for B,color in [(80,'#245a81'),(160,'#27724b')]:
            sub=audit[audit.total_labels.eq(B)&audit.slopes.eq(round(rho*B))].sort_values('training_labels')
            x=sub.training_labels/B;k=sub.some_positive_power_finite.to_numpy();n=sub.draws.to_numpy();p=k/n
            lo,hi=wilson(k,n)
            ax.errorbar(x,p,yerr=np.maximum(0,np.vstack([p-lo,hi-p])),fmt='o-',color=color,capsize=3,label=f'Any positive power, {B} labels')
            ref=direct[direct.total_labels.eq(B)&direct.slopes.eq(round(rho*B))&direct.method.eq('all_labels_human')].sort_values('training_labels')
            k=ref.finite_events.to_numpy();n=ref.draws.to_numpy();p=k/n;lo,hi=wilson(k,n)
            ax.errorbar(ref.training_labels/B,p,yerr=np.maximum(0,np.vstack([p-lo,hi-p])),fmt='s--',mfc='white',color=color,capsize=3,label=f'Direct human, {B} labels')
        ax.set(title=f'{"c" if rho==.3 else "d"}  Finite events at slope ratio {rho:g}',ylabel='Verified event frequency',ylim=(-.03,1.03))
    for ax in axes.ravel():
        ax.set_xticks([.25,.5],['25%','50%']);ax.set_xlabel('Fraction of human labels used for training')
        ax.grid(axis='y',alpha=.18)
    handles,labels=axes[1,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=8.5)
    fig.tight_layout(pad=1.4,rect=(0,.085,1,1))
    fig.savefig(out/'learned_training_budget.png',dpi=180)
    fig.savefig(out/'learned_training_budget.pdf')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.base,a.out)
