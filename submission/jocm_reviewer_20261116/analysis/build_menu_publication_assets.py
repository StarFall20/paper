"""Build publication-sized menu figure and appendix tables from frozen results."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def markdown(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,row))+'|' for row in rows])


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--manuscript',type=Path,required=True);a=p.parse_args()
    population=pd.read_csv(a.results/'hard_menu_primary_values.csv');phase=pd.read_csv(a.results/'hard_menu_matched_phase_summary.csv')
    assert len(phase)==18 and phase.replications.eq(500).all() and phase.unresolved.sum()==0
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(6.5,3.1),sharex=True,sharey=True)
    methods=[('ordinary','Ordinary','#444444'),('unit','Unit power','#CC6677'),('oracle_null_slope_variance','Variance reference','#0072B2')]
    for ax,(method,title,color) in zip(axes,methods):
        part=phase[phase.method.eq(method)]
        for m,marker,style in [(40,'o','--'),(80,'s','-')]:
            cell=part[part.label_menus==m].sort_values('dimension_ratio');n=cell.replications.to_numpy();freq=cell.finite_frequency.to_numpy();z=1.959963984540054
            center=(freq+z*z/(2*n))/(1+z*z/n);radius=z*np.sqrt(freq*(1-freq)/n+z*z/(4*n*n))/(1+z*z/n)
            errors=np.maximum(np.vstack([freq-center+radius,center+radius-freq]),0.)
            ax.errorbar(cell.dimension_ratio,freq,yerr=errors,color=color,marker=marker,ms=4,ls=style,lw=1.4,capsize=2,label=f'{m} labelled menus')
        ax.axvline(float(part.iloc[0].critical_ratio),color=color,ls=':',lw=1.3,label='Boundary')
        ax.set(title=title,ylim=(-.02,1.02),xlim=(.1,.78));ax.set_xticks([.2,.4,.6]);ax.grid(alpha=.14);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Finite fit frequency')
    fig.supxlabel('Orthogonal attributes per labelled menu',fontsize=9,y=.12)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=8,bbox_to_anchor=(.5,-.005))
    fig.tight_layout(rect=(0,.16,1,1),w_pad=.8)
    target=a.manuscript/'figures';target.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','svg']:fig.savefig(target/f'figure_menu_phase.{suffix}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    rows=[]
    for J in [2,3]:
      for gamma in [0.,1.5]:
       for tau in [1.,4.]:
        part=population[(population.alternatives==J)&(population.gamma==gamma)&(population.tau==tau)&(population.epsilon==1)]
        ordinary=float(part[part.event.eq('ordinary')].iloc[0].critical_ratio);limit=float(part[part.event.eq('positive_limit')].iloc[0].critical_ratio)
        rows.append([J,f'{gamma:g}',f'{tau:g}',f'{ordinary:.6f}',f'{limit:.6f}',f'{limit-ordinary:.6f}'])
    h1=['Alternatives','Signal','Auxiliary ratio','Ordinary','Positive limit','Difference']
    wide=phase.pivot(index=['label_menus','dimension_ratio'],columns='method',values='finite_frequency').reset_index()
    h2=['Labelled menus','Slopes per menu','Ordinary','Unit','Variance reference']
    r2=[[int(r.label_menus),f'{r.dimension_ratio:.2f}',f'{r.ordinary:.3f}',f'{r.unit:.3f}',f'{r.oracle_null_slope_variance:.3f}'] for r in wide.itertuples()]
    manuscript=(a.manuscript/'paper.md').read_text()
    for name,headers,data in [('tableF1',h1,rows),('tableF2',h2,r2)]:
        shown=markdown(headers,data);assert shown in manuscript,name
        (a.manuscript/f'{name}.md').write_text(shown+'\n');pd.DataFrame(data,columns=headers).to_csv(a.manuscript/f'{name}.csv',index=False)
    print(json.dumps(dict(figure_width_inches=6.5,formats=['png','svg'],appendix_tables_verified=2,all_values_from_completed_results=True)))


if __name__=='__main__':main()
