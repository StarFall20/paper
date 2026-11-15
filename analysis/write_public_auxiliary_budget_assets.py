"""Plot the entire prespecified public auxiliary-budget grid."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    f=pd.read_csv(a.base/'results/summary.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(9.8,7),sharex=True,sharey=True)
    methods=[('source_tuned_each_N','Retuned source power','#0072B2','o'),('fixed_first_N_power','Power fixed at N = 50','#D55E00','D'),('unit','Unit power','#009E73','s')]
    for i,m in enumerate([20,50]):
        for j,pred in enumerate(['AI','X_only']):
            ax=axes[i,j]
            for method,label,color,marker in methods:
                s=f[(f.label_rows==m)&(f.prediction==pred)&(f.method==method)].sort_values('auxiliary_rows')
                n=s.allocations.to_numpy();q=s.interior.to_numpy()/n;z=1.959963984540054
                den=1+z*z/n;center=(q+z*z/(2*n))/den
                half=z*np.sqrt(q*(1-q)/n+z*z/(4*n*n))/den
                ax.errorbar(s.auxiliary_rows,q,yerr=np.r_[[(q-center+half)],[(center+half-q)]].reshape(2,-1),
                    color=color,marker=marker,markersize=6 if marker=='D' else 4.5,
                    markerfacecolor='none' if marker=='D' else color,linewidth=1.3,capsize=2,label=label)
            base=s.iloc[0];h=1-base.human_separated/base.allocations
            ax.axhline(h,ls='--',color='0.4',lw=1.2,label='Human logit with the same labels')
            ax.set_title(f'{m} human labels; '+('AI-aided prediction' if pred=='AI' else 'attributes-only prediction'),fontsize=10.5)
            ax.set_xscale('log',base=2);ax.set_ylim(-.025,1.06);ax.set_xlim(45,1350)
            ax.set_xticks([50,100,200,400,800,1200]);ax.set_xticklabels(['50','100','200','400','800','1200'])
            ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
            if j==0:ax.set_ylabel('Fraction with a finite unique estimate')
            if i==1:ax.set_xlabel('Auxiliary profiles N')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.015),ncol=2,frameon=False,fontsize=9)
    fig.subplots_adjust(left=.085,right=.985,top=.94,bottom=.18,hspace=.20,wspace=.12)
    for ext in ['png','pdf']:fig.savefig(a.out/f'public_auxiliary_budget.{ext}',dpi=190)
    plt.close(fig)
    metadata=dict(source='results/summary.csv',allocations_per_point=500,
        panels=4,prespecified_cells=72,intervals='95% Wilson binomial Monte Carlo intervals conditional on the fixed file and allocation scheme.',
        notes=['The 20-label final pool has 1180 rows; the 50-label pool has 1150.',
               'Retuned and fixed-first-N curves coincide at 50 human labels.',
               'Budgets and methods share each allocation. Lines connect measured points.',
               'The human baseline repeats the same fit across N and across prediction routines.'],
        finite_event='Strict moment interior with full curvature rank; zero-power separated boundaries are excluded.')
    (a.out/'figure_notes.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':main()
