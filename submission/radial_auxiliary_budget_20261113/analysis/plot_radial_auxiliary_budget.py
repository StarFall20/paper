"""Plot retained population calculations; no new model fitting."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(args.root/'results/population_budget_boundaries.csv')
    assert len(frame)==52
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    styles=[('constant_scale','variance_rule','#1f5d92','-','Constant scale; variance rule'),
            ('two_point_scale','variance_rule','#b84034','-','Variable scale; variance rule'),
            ('two_point_scale','held_power','#518543','--','Variable scale; fixed power')]
    for model,rule,color,style,label in styles:
        part=frame[(frame.model==model)&(frame.rule==rule)].sort_values('auxiliary_ratio')
        assert len(part)==13
        axes[0].plot(part.auxiliary_ratio,part.critical_ratio,color=color,ls=style,lw=1.8,label=label)
    axes[0].axhline(.25,color='#777777',ls=':',lw=1.2,label='Simulated dimension ratio 0.25')
    axes[0].set(xscale='log',xlabel='Auxiliary / labelled rows',ylabel='Critical noise / label ratio',
                title='A  Boundary for a finite estimate',ylim=(0,.92))
    axes[0].legend(loc='upper left',fontsize=7.3,frameon=False)
    part=frame[(frame.model=='two_point_scale')&(frame.rule=='variance_rule')].sort_values('auxiliary_ratio')
    same=frame[(frame.model=='constant_scale')&(frame.rule=='variance_rule')].sort_values('auxiliary_ratio')
    assert np.allclose(part.regular_covariance_times_labels,same.regular_covariance_times_labels)
    axes[1].plot(part.auxiliary_ratio,part.regular_covariance_times_labels,color='#6d5196',lw=2,
                 label='Both scale laws; variance rule')
    axes[1].axhline(4,color='#777777',ls=':',lw=1.2,label='Labelled-only reference')
    axes[1].set(xscale='log',xlabel='Auxiliary / labelled rows',ylabel='Oracle covariance factor (mV)',
                title='B  Regular local covariance',ylim=(2.85,4.12))
    axes[1].legend(loc='lower left',fontsize=8,frameon=False)
    for ax in axes:
        ax.grid(axis='y',alpha=.16)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.xaxis.label.set_size(9);ax.yaxis.label.set_size(9);ax.title.set_size(10)
    for suffix in ['png','pdf','svg']:
        fig.savefig(args.out/('auxiliary_budget.'+suffix),dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
