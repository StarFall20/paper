"""Report the completed positive-power checks and original-draw replay."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from finite_ratio_choice_boundary import critical_ratio
from vanishing_power_choice import positive_limit
from public_conjoint_optimizer_audit import save

def table(frame):return '\n'.join(['| '+' | '.join(frame.columns)+' |','|'+'|'.join(['---']*len(frame.columns))+'|']+['| '+' | '.join(map(str,r))+' |' for r in frame.itertuples(index=False,name=None)])

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--manuscript',type=Path,required=True);a=p.parse_args();r=a.results;md=a.manuscript;figdir=md/'figures';figdir.mkdir(parents=True,exist_ok=True)
 status=json.loads((r/'vanishing_power_status.json').read_text());replay=json.loads((r/'replayed_power_status.json').read_text());resolution=json.loads((r/'vanishing_power_resolution.json').read_text())
 assert status['completed_DGP_blocks']==12 and status['replications_per_block']==500 and replay['completed_repetitions']==500 and resolution['remaining_unresolved']==0
 q=pd.read_csv(r/'vanishing_power_phase_verified.csv');assert len(q)==72 and ((q.interior+q.outside+q.boundary_or_near_boundary+q.unresolved)==500).all()
 summary=pd.read_csv(r/'replayed_power_overall_summary.csv');paired=pd.read_csv(r/'replayed_power_paired_comparisons.csv');assert len(summary)==5 and (summary.estimates==summary.finite_support).all()
 display=[]
 for _,row in summary.iterrows():display.append(['0 human' if row.power==0 else f'{row.power:g}',f'{int(row.estimates)}/500',f'{row.MSE:.3f}',f'{row.median_MSE:.3f}',f'{row.q90_MSE:.3f}'])
 tab=pd.DataFrame(display,columns=['Power','Available','Mean MSE','Median MSE','90th percentile MSE']);tab.to_csv(md/'table3.csv',index=False);(md/'table3.md').write_text(table(tab)+'\n')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
 fig,axs=plt.subplots(1,2,figsize=(8,3.9))
 powers=np.linspace(.005,1,120)
 for e,color,label in [(.25,'#0072B2','Oracle proxy error 0.25'),(.5,'#D55E00','Uninformative prediction')]:
  values=[critical_ratio(1.,e,float(l))[0] for l in powers];limit,_=positive_limit(1.,e)
  axs[0].plot(powers,values,color=color,label=label);axs[0].plot(0.,limit,'o',mfc='white',mec=color,ms=6)
 axs[0].plot(0.,.5,'o',color='black',ms=5,label='Exactly zero power');axs[0].set_xlim(-.03,1.03);axs[0].set_ylim(.3,.7);axs[0].set_xlabel('Correction power');axs[0].set_ylabel('Critical slopes per labelled row');axs[0].set_title('a  The zero endpoint',loc='left',fontsize=10);axs[0].grid(alpha=.15)
 vals={1.:float(paired.query('power_reference==1 and power_comparator==0.01').iloc[0].MSE_reference)}
 for lam in [.01,.1,.5]:
  row=paired[(paired.power_reference==1)&(paired.power_comparator==lam)].iloc[0];assert row.shared_finite_estimates==412;vals[lam]=float(row.MSE_comparator)
 xs=sorted(vals);axs[1].plot(xs,[vals[v] for v in xs],'o-',color='#333333');axs[1].set_xscale('log');axs[1].set_yscale('log');axs[1].set_xlabel('Correction power');axs[1].set_ylabel('Parameter MSE on shared fits');axs[1].set_title('b  The same 412 available unit fits',loc='left',fontsize=10);axs[1].grid(alpha=.15)
 for lam in xs:axs[1].annotate(f'{vals[lam]:.2f}',(lam,vals[lam]),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
 axs[1].set_ylim(.8,300);handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=9);fig.tight_layout(rect=(0,.17,1,1));fig.subplots_adjust(wspace=.35)
 fig.savefig(figdir/'figure3_power_endpoint.png',dpi=220);fig.savefig(figdir/'figure3_power_endpoint.svg');plt.close(fig)
 report=f'''# 正权重趋零与有限解的代价

这一轮在原修正logit问题内增加了可证明的权重结构。对冻结预测器和同一数据，正权重降低时，已有有限解不会因这一改变消失。能否在某个正权重下获得有限解，取决于单位权重衰退函数在真人弱分离方向上是否严格为正。普通正权重伪标签合并的存在事件对权重大小不变；修正目标包含被减去的标签预测损失，其存在事件依赖补偿关系。一般分离、伪数据正则化和消失惩罚的间隔极限均按既有文献归属。

平衡Gaussian基准给出正权重右极限公式。在辅助/标签比例1、代理错误率0.25时，单位权重阈值0.355439，权重0.5的阈值0.602855，正权重趋零的阈值0.664488；权重恰好为零的普通logit阈值为0.5。这是先取每个固定正权重的比例维度极限，再令权重趋零；没有声称任意同步趋零序列都具有同一边界。

12个Gaussian数据生成条件各完成500次，共6000个成对样本。每个样本比较零权重、四个正权重和正权重极限可行性。3854个极限内部案例还构造并验证了实际正权重。正权重内部未发现违反嵌套关系的案例。原LP有1个未知状态，重建该样本后独立原矩LP给出正分离间隙0.000498，验证为外部矩；原表和独立复核表同时保留。不能把原求解器未知状态直接改写成非存在。

本轮还重放原支撑模拟的完整500次抽样：m20、N600、代理错误率0.25、原效用参数。1000个人类MLE/单位权重状态与已完成结果一致，可用点的最大MSE相对差异低于4.8e-8。这是原样本复算，没有新增受访者、AI答卷或独立响应实验。

{table(tab)}

上表MSE分别条件于各行可用点，分母不同。限制到单位权重可用的同412个样本，权重0.01的MSE为153.487，单位权重为4.836，成对差值均值148.650，MonteCarlo标准误10.180；权重0.01在94.66%的这些样本上有更大MSE。权重0.5在相同412个样本上的MSE为1.615。完整配对表与真人分离/非分离分层同时保留。

小权重可以使更多目标存在有限根，也可以沿分离方向产生很大的系数。对固定严格分离样本，本文推导出除以log(1/权重)后的系数集合极限，成本是修正目标的辅助衰退函数。Rosset–Zhu–Hastie的消失正则化/间隔结果是这个解释的直接前驱，不能把一般间隔收敛包装成全新理论。解析一维例子及二维独立LP给出相应核查。所有正权重复算根采用缩放得分，最大缩放得分误差低于1e-9，避免小权重下绝对梯度阈值过松。

新增结论加强了论文对“如何理解修正权重”的解释。它没有产生新的最优推断权重或置信区间保证。已有一步估计、正则化和软合并在原七方法比较中的成功继续保留。核心投稿增量仍需根据显式补偿边界、零端点结果及其适用范围审视，不能由这些数值通过率推出。
'''
 (r/'正权重零端点与创新深化_20261110.md').write_text(report)
 save(r/'vanishing_power_report_checks.json',{'all12Gaussian_blocks_complete':True,'raw_unknown_retained_and_independently_resolved':True,'replay_original_status_matches':replay['original_human_unit_reconstruction_status_matches'],'new_respondents_or_AI_samples':False,'table3_rows':len(tab),'figure3_MSE_common_repetitions':412,'fixed_power_then_power_limit_distinguished_from_joint_limit':True,'novelty_or_submission_readiness_proven':False})
 print(json.dumps({'Gaussian_DGP_blocks':12,'Gaussian_paired_draws':6000,'original_draws_replayed':500,'table3_rows':len(tab),'figure3_shared_available':412}))

if __name__=='__main__':main()
