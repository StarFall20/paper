"""Publication figures/tables from completed aggregate experiments only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path('results'));p.add_argument('--manuscript',type=Path,default=Path('manuscript'));args=p.parse_args()
SRC=args.results
OUT=args.manuscript
FIG=OUT/'figures'
FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
phase=pd.read_csv(SRC/'imperfect_prediction_phase.csv')
fig,axs=plt.subplots(1,2,figsize=(8,3.6),sharey=True)
for ax,pred,title in zip(axs,['overconfident','intercept_bias'],['a  Overconfident predictions','b  Shifted prediction log odds']):
 for power,color,label in [('unit','#333333','Unit power'),('orthogonal_optimal','#0072B2','Variance reference power')]:
  for m,mark,ls in [(40,'o','--'),(80,'s','-')]:
   p=phase[(phase.tau==1)&(phase.prediction==pred)&(phase.power_name==power)&(phase.labels==m)].sort_values('orthogonal_dimension_label_ratio')
   assert len(p)==5 and (p.replications==300).all() and p.unresolved.sum()==0
   ax.errorbar(p.orthogonal_dimension_label_ratio,p.finite_frequency,yerr=1.96*np.sqrt(p.finite_frequency*(1-p.finite_frequency)/300),color=color,marker=mark,ls=ls,lw=1,ms=4,capsize=2,label=f'{label}, m={m}')
  ax.axvline(p.critical_ratio.iloc[0],color=color,lw=1,ls=':')
 ax.set_title(title,loc='left',fontsize=10);ax.set_xlabel('Orthogonal slopes per labelled row');ax.set_ylim(-.04,1.04);ax.set_xlim(0,.7);ax.grid(alpha=.15)
axs[0].set_ylabel('Finite fit frequency')
handles,legendlabels=axs[0].get_legend_handles_labels()
fig.legend(handles,legendlabels,frameon=False,fontsize=9,loc='lower center',ncol=2,bbox_to_anchor=(.5,-.005))
fig.tight_layout(rect=(0,.17,1,1));fig.subplots_adjust(wspace=.25);fig.savefig(FIG/'figure1_phase.png',dpi=220);fig.savefig(FIG/'figure1_phase.svg');plt.close(fig)

order=['human_MLE','PPI_unit','PPI_power_human','PPI_power_ridge','PPI_one_step_ridge','PPI_ridge_unit','soft_oracle_pool']
labels=['Human MLE','Unit PPI','Power human start','Power ridge start','One step','Ridge PPI','Soft pooling']
low=pd.read_csv(SRC/'finite_pool_summary.csv').query('label_rows==20 and auxiliary_rows==600 and epsilon==0.25').set_index('method').loc[order].reset_index()
high=pd.read_csv(SRC/'strong_signal_pool_summary.csv').query('label_rows==150 and prediction=="overconfident"').copy()
high['method']=high.method.replace({'soft_frozen_pool':'soft_oracle_pool'});high=high.set_index('method').loc[order].reset_index()
fig,axs=plt.subplots(2,2,figsize=(9,6.0),sharey='row')
for col,(part,title) in enumerate([(low,'a  20 labels and oracle predictions'),(high,'b  150 labels and overconfident predictions')]):
 y=np.arange(7);colors=['#555555']*6+['#0072B2']
 axs[0,col].barh(y,part.MSE,color=colors,height=.6);axs[0,col].set_xscale('log');axs[0,col].set_xlabel('Mean parameter squared error');axs[0,col].set_title(title,loc='left',fontsize=10)
 for j,row in part.iterrows():
  axs[0,col].annotate(f'{row.estimates}/500',(row.MSE,j),xytext=(4,0),textcoords='offset points',va='center',fontsize=8)
 axs[0,col].set_xlim(.003,80);axs[0,col].set_yticks(y,labels);axs[0,col].set_ylim(6.6,-.6)
 axs[1,col].plot(part.conditional_marginal_coverage,y,'o',color='#333333',label='Coverage given an interval')
 axs[1,col].plot(part.interval_and_marginal_coverage,y,'s',mfc='none',color='#D55E00',label='Issued and covers')
 axs[1,col].axvline(.95,color='#777777',ls=':',lw=1);axs[1,col].set_xlabel('Average marginal coverage');axs[1,col].set_xlim(.15,1.02);axs[1,col].set_yticks(y,labels);axs[1,col].set_ylim(6.6,-.6);axs[1,col].grid(axis='x',alpha=.15)
axs[1,0].legend(frameon=False,fontsize=8,loc='lower left')
fig.tight_layout();fig.savefig(FIG/'figure2_estimates_intervals.png',dpi=220);fig.savefig(FIG/'figure2_estimates_intervals.svg');plt.close(fig)

def mdtable(frame):
 return '\n'.join(['| '+' | '.join(frame.columns)+' |','|'+'|'.join(['---']*len(frame.columns))+'|']+['| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None)])
rows=[]
for name,part in [('20 labels oracle',low),('150 labels overconfident',high)]:
 for label,(_,row) in zip(labels,part.iterrows()):
  rows.append([name,label,f'{row.estimates}/500',f'{row.MSE:.4f}',f'{row.conditional_marginal_coverage:.3f}',f'{row.interval_and_marginal_coverage:.3f}'])
table=pd.DataFrame(rows,columns=['Condition','Estimator','Available','MSE','Coverage','Issued and covers'])
(OUT/'table2.md').write_text(mdtable(table)+'\n')
table.to_csv(OUT/'table2.csv',index=False)
bound=pd.read_csv(SRC/'imperfect_prediction_boundaries.csv')
rows=[]
for tau in [1,4]:
 for pred in ['oracle','overconfident','intercept_bias']:
  for power in ['unit','orthogonal_optimal']:
   row=bound[(bound.tau==tau)&(bound.prediction==pred)&(bound.power_name==power)].iloc[0]
   rows.append([str(tau),{'oracle':'Oracle','overconfident':'Overconfident','intercept_bias':'Shifted log odds'}[pred],f'{row.power:.4f}',f'{row.critical_ratio:.4f}'])
table=pd.DataFrame(rows,columns=['Auxiliary to label ratio','Prediction','Power','Critical slope to label ratio'])
(OUT/'table1.md').write_text(mdtable(table)+'\n');table.to_csv(OUT/'table1.csv',index=False)
(OUT/'asset_provenance.json').write_text(json.dumps({'new_estimates_or_simulations':False,'sources':['imperfect_prediction_phase.csv','finite_pool_summary.csv','strong_signal_pool_summary.csv','imperfect_prediction_boundaries.csv'],'all_selected_cells_completed':True,'figure2_availability_is_point_estimate_availability':'interval availability remains in underlying complete aggregate tables'},indent=2)+'\n')
print('Created two publication figures and two tables from completed aggregates.')
