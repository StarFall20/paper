"""Report completed biased-predictor theory and finite-support validation."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'human_MLE':'Labelled MLE','PPI_unit':'PPI, unit power','PPI_power_human':'Power, labelled start','PPI_power_ridge':'Power, ridge start','PPI_one_step_ridge':'One step, ridge start','PPI_ridge_unit':'Ridge PPI','soft_frozen_pool':'Frozen soft pooling'}
PREDICTIONS={'oracle':'Oracle','overconfident':'Overconfident','intercept_bias':'Intercept bias'}


def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out
 phase=pd.read_csv(out/'imperfect_prediction_phase.csv');boundaries=pd.read_csv(out/'imperfect_prediction_boundaries.csv')
 assert len(phase)==80 and (phase.replications==300).all() and int(phase.unresolved.sum())==0
 fig,axes=plt.subplots(2,2,figsize=(10.6,7.2),sharey=True,layout='constrained')
 for i,tau in enumerate([1,4]):
  for j,prediction in enumerate(['overconfident','intercept_bias']):
   ax=axes[i,j]
   for power,color in [('unit','#c26939'),('orthogonal_optimal','#267e9c')]:
    for m,marker in [(40,'o'),(80,'s')]:
     s=phase[(phase.tau==tau)&(phase.prediction==prediction)&(phase.power_name==power)&(phase.labels==m)].sort_values('orthogonal_dimension_label_ratio')
     se=np.sqrt(s.finite_frequency*(1-s.finite_frequency)/300)
     ax.errorbar(s.orthogonal_dimension_label_ratio,s.finite_frequency,yerr=1.96*se,color=color,marker=marker,ms=3,capsize=2,ls='--' if m==40 else '-',alpha=.65 if m==40 else 1.,label=f'{"Unit power" if power=="unit" else "Score-optimal power"}; m={m}')
    ax.axvline(s.critical_ratio.iloc[0],color=color,ls=':',lw=1)
   ax.set_title(f'{PREDICTIONS[prediction]}; N/m={tau}');ax.set_ylim(-.04,1.04);ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False);ax.set_xlabel('Orthogonal dimensions / labels (q/m)')
 axes[0,0].set_ylabel('Finite-moment frequency');axes[1,0].set_ylabel('Finite-moment frequency')
 handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False,fontsize=7)
 fig.savefig(out/'imperfect_prediction_phase.svg');fig.savefig(out/'imperfect_prediction_phase.png',dpi=180);plt.close(fig)
 summaries={};all_trials=0
 for prefix,scale in [('imperfect_pool',1),('strong_signal_pool',3)]:
  status=json.loads((out/(prefix+'_numeric_checks.json')).read_text());assert status['completed_replications']==status['requested_replications']==500
  trial=pd.read_csv(out/(prefix+'_trial_diagnostics.csv'));assert len(trial)==21000
  assert trial.excess_logloss.min()>-1e-12
  # The labelled estimator is exactly shared across the three prediction cases.
  human=trial[trial.method=='human_MLE'].pivot(index=['replication','label_rows'],columns='prediction',values='MSE')
  assert (human.max(axis=1)-human.min(axis=1)).max()<1e-12
  summary=pd.read_csv(out/(prefix+'_summary.csv'));assert len(summary)==42 and (summary.replications==500).all();summaries[scale]=summary;all_trials+=len(trial)
  fig,axes=plt.subplots(2,3,figsize=(14,8),sharey=True,layout='constrained')
  for i,m in enumerate([50,150]):
   for j,prediction in enumerate(PREDICTIONS):
    ax=axes[i,j];s=summary[(summary.label_rows==m)&(summary.prediction==prediction)].set_index('method').loc[list(LABELS)]
    y=np.arange(len(LABELS));ax.barh(y-.16,s.conditional_marginal_coverage,height=.28,color='#267e9c',label='Conditional coverage')
    ax.barh(y+.16,s.interval_and_marginal_coverage,height=.28,color='#c26939',label='Interval issued and covers')
    ax.axvline(.95,color='black',ls=':',lw=1);ax.set_xlim(0,1.03);ax.set_yticks(y,list(LABELS.values()));ax.set_title(f'{PREDICTIONS[prediction]}; m={m}')
    ax.set_xlabel('Average coverage over 11 fixed coordinates');ax.grid(axis='x',alpha=.15);ax.spines[['top','right']].set_visible(False)
  axes[0,0].invert_yaxis();handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False,fontsize=8)
  fig.savefig(out/(prefix+'_coverage.svg'));fig.savefig(out/(prefix+'_coverage.png'),dpi=180);plt.close(fig)
 report='''# 预测偏差、一般相变先例与实际属性支撑验证

2026年11月10日。本轮完成有偏冻结预测的理论核对，以及较弱、较强效用下的七方法对照。论文仍需完成优先权与独立证明审视、实际分析决策和投稿整合。

## 与最接近通用定理的关系

新找到的[Koriyama与Bellec2025](https://arxiv.org/html/2501.03163v3)已研究一般高斯单指标M估计器的存在相变。它的假设要求逐行损失的下确界具有有限绝对期望。当前校正目标的标签行损失为

(1−λ)log(1+exp(u))−(Y−λg)u。

当λ>0、g严格位于0和1之间时，Y=0的行在u趋向负无穷时趋向负无穷，Y=1的行在u趋向正无穷时趋向负无穷；λ=1时还成为线性函数。该定理不能直接覆盖这些行。总目标能否有有限根取决于行之间的补偿，具体审查见[近邻定理对照](general_M_estimator_prior_audit.md)。这说明一个明确的适用差别，其他等价先例仍需核对。

## 有偏预测的边界核对

固定真实效用信号γ=1.5、代理翻转率0.25，将正确条件对数几率乘以a、再加b得到冻结预测。过度自信取a=2,b=0；截距偏差取a=1,b=1。后者使用两个自由乘子求边界，检验了此前对称性简化以外的情形。

| 辅助/标签比 | 预测 | Brier误差 | 单位权重边界 | 既有协方差公式给出的权重 | 该权重下边界 |
|---:|---|---:|---:|---:|---:|
'''
 for tau in [1,4]:
  for pred in PREDICTIONS:
   s=boundaries[(boundaries.tau==tau)&(boundaries.prediction==pred)].set_index('power_name');u=s.loc['unit'];v=s.loc['orthogonal_optimal']
   report+=f'| {tau} | {PREDICTIONS[pred]} | {u.Brier_risk:.4f} | {u.critical_ratio:.4f} | {v.power:.4f} | {v.critical_ratio:.4f} |\n'
 report+='''
权重针对正交斜率的固定维度渐近方差，来自已有PPI协方差公式；它在实验前按总体分布固定。本文没有声称它在比例维度增长下同时最优，也没有将其等同于数据估计出来的权重。反向预测的该公式选择零权重，保留为单独的普通逻辑回归端点。

![有偏预测的有限根实验](imperfect_prediction_phase.svg)

80个条件、每条件300次直接生成高斯选择行的实验已完成，没有数值未决情形。另有120组原始矩LP与定向区间LP一致，12组独立圆锥投影一致。100/200节点积分、完整乘子梯度和包络Hessian分别核对。所有图点和误差棒使用全部300次，有限m曲线不等于渐近阶跃。

## 更严格的理论结果及其量级

[匹配误差与协方差的命题](matched_moment_existence_proposition.md)构造了预测均值、Brier误差和整个固定维度渐近协方差都相同的两种预测。辅助置信度平方更分散时，单位权重的有限估计边界严格下降。其推导仍针对同一个校正目标，没有增加另一套框架。

独立有限状态求和核对的协方差差小于2.3e-16；直接正态积分与边界公式差小于1e-15。当前构造在N/m=0.05时边界相对差约5.98%，在N/m=1时约0.12%，N/m=4时更小。它说明既有误差和方差摘要不能完全确定有限估计条件，但当前量级尚不足以建立普遍的实际收益。

## 真实属性支撑上的模型对照

模拟保留公开疫苗数据1200个属性组合，独立抽取选择行，生成真实选择及代理。第一组固定此前收敛参数；第二组将该参数乘以3。两组都不是对真人结构参数的重新认定。每组m=50、150，辅助行600，三种冻结预测，每条件500次。七方法使用同一条件下相同的属性、选择和代理。

两个信号组共12个条件、每条件500次，共42000个方法—条件结果；它们不代表42000名受访者。软标签合并使用与其他方法相同的预测，只有理想预测组具有正确条件均值。

'''
 for scale,summary in summaries.items():
  report+=f'### 效用参数倍数 {scale}\n\n'
  report+='下表固定150个标签行。条件覆盖率在已发出区间中计算；“发出且覆盖”以全部500次为分母。参数MSE保留全部可用估计，失败数单列。\n\n| 预测 | 方法 | 有限估计 | 条件覆盖率 | 发出且覆盖 | 参数MSE |\n|---|---|---:|---:|---:|---:|\n'
  for pred in PREDICTIONS:
   s=summary[(summary.label_rows==150)&(summary.prediction==pred)].set_index('method')
   for method,label in LABELS.items():
    r=s.loc[method];report+=f'| {PREDICTIONS[pred]} | {label} | {int(r.estimates)}/500 | {r.conditional_marginal_coverage:.3f} | {r.interval_and_marginal_coverage:.3f} | {r.MSE:.4f} |\n'
  prefix='imperfect_pool' if scale==1 else 'strong_signal_pool'
  report+=f'\n![效用倍数{scale}的区间任务]({prefix}_coverage.svg)\n\n'
 report+='''加惩罚方法优化了改变后的目标，区间没有校正惩罚偏差。一步估计可返回有限参数，但覆盖率仍由实验检验。完整结果保留50个标签行、11个预先固定坐标、尾部误差、每个失败状态及配对Monte Carlo误差。较弱效用下软标签合并的强表现保留在表中，不能据选择性的失败案例否定它。

### 总体目标的独立核对

在同一个有限属性分布上，精确对两个代理取值求期望，再拟合期望软标签目标。固定权重的校正总体得分在真实模拟参数处为零，这一校正恒等式属于已有PPI原理。数据依赖权重的有限样本无偏性没有由此得到证明。

| 效用倍数 | 预测 | 150标签/600辅助时软合并总体参数MSE | 总体额外选择损失 |
|---:|---|---:|---:|
'''
 for prefix,scale in [('imperfect_pool',1),('strong_signal_pool',3)]:
  target=pd.read_csv(out/(prefix+'_population_targets.csv'))
  for pred in PREDICTIONS:
   r=target[(target.label_rows==150)&(target.prediction==pred)].iloc[0]
   report+=f'| {scale} | {PREDICTIONS[pred]} | {r.soft_population_parameter_MSE:.6f} | {max(0,r.soft_population_excess_logloss):.6f} |\n'
 report+='''
有偏预测会改变软合并的总体目标。当前较弱效用下偏差较小，增加效用强度后过度自信预测的总体目标偏离更明显。表中同时给出偏差与实际有限样本表现；存在目标偏差不意味着它在每个有限样本条件下都比校正估计差。

## 期刊与进度判断

[Elsevier官方期刊说明](https://shop.elsevier.com/journals/journal-of-choice-modelling/1755-5345)确认JOCM接受选择建模的理论、方法贡献和创新应用。当前二元选择估计问题在主题范围内；稿件仍需证明方法增量和读者的实际分析收益。最新投稿指南入口本轮返回403。

本轮Scopus页面无法通过检索工具读取；Google Scholar浏览器重定向至Google访问检查地址，未读取检索结果。没有完成数据库系统综述。主要依据是直接核对的作者全文、正式论文记录和期刊官网。复现检查和自评不作为投稿达标证明。

下一步将理论候选、预测偏差实验和公开选择重分析收束到一条清楚的论证，核对近邻优先权，并检验一个可执行的实际分析决策。原300份真人招募问卷的探索性应用与60份AI预测试继续分列。当前目标仍未完成。
'''
 (out/'预测偏差检验与创新边界深化.md').write_text(report)
 checks={'imperfect_phase_cells':80,'phase_replications_per_cell':300,'finite_support_conditions':12,'finite_support_replications_per_cell':500,'finite_support_methods':7,'method_condition_outcomes':all_trials,'prediction_cases':3,'truth_scales':[1,3],'independent_novelty_proven':False,'submission_goal_complete':False}
 (out/'imperfect_prediction_report_checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))


if __name__=='__main__':main()
