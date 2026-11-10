"""Report complete replay and repetition-level precision; no new fits or draws."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();r=a.results
f=pd.read_csv(r/'dypl_comparator_trial_diagnostics.csv');s=pd.read_csv(r/'dypl_comparator_summary.csv');status=json.loads((r/'dypl_comparator_status.json').read_text())
assert len(f)==2000 and status['completed_DY_fits']==2000 and status['human_reconstruction_status_matches']==1000
precision=[]
for (condition,method),part in f.groupby(['condition','method']):
 row={'condition':condition,'method':method,'original_draws':len(part)}
 for name in ['MSE','excess_logloss','marginal_coverage']:
  values=part[name].dropna();row[name+'_n']=len(values);row[name+'_mean']=values.mean();row[name+'_MCSE']=values.std(ddof=1)/np.sqrt(len(values))
 ref=s[(s.condition==condition)&(s.method==method)].iloc[0]
 assert abs(ref.MSE-row['MSE_mean'])<1e-12 and abs(ref.expected_KL-row['excess_logloss_mean'])<1e-12
 precision.append(row)
pd.DataFrame(precision).to_csv(r/'dypl_comparator_precision.csv',index=False)
rows=[]
for x in s.itertuples():
 rows.append(f'| {x.condition} | {x.method} | {x.alpha:.4f} | {x.estimates}/500 | {x.MSE:.6f} | {x.expected_KL:.6f} | {x.conditional_marginal_coverage:.6f} |')
report='''# 最近前驱补查、DY比较与投稿进展

2026年11月10日。上一轮完成一般权重边界推广，本轮核对其相对最近文献的增量，并补上遗漏的成熟比较方法。目标仍未完成，不用数值通过率代替原创性或投稿判断。

## 这次发现改变了哪些表述

带噪标签文献早已研究无偏损失校正和凸性。DY惩罚logit已有有限估计保证及高维理论，2026年还有联合估计截距的工作；作者的公开代码亦包含Bradley–Terry选择比较。因此，保证有限估计、损失校正或加入截距均不能单独算本文创新。正文已补引这些前驱；完整来源和阅读范围见最近前驱审计。

本文保留的候选增量是同一未惩罚校正目标的显式双样本存在边界、全部权重同时上界和正权重趋零的结论。另核查了一种直接推论捷径：只依据被条件化的低维变量重新分配各行线性项，若保持原目标，剩余高斯设计会强制所有修正为零。这排除一种具体修复，未宣称排除所有等价定理。

## 同样本补充比较

两组正文原有条件各500组抽样、原七种方法及其结果均保留。新增DY设置只用相同真人标签，取作者代码使用的维数规则alpha=m/(m+11)，另固定alpha=0.5作敏感性。未用真值选择强度或初始化；未生成新问卷或独立模拟抽样。1000个原human拟合的状态全部重建一致，可用MSE最大相对差异1.73e-16；原试验文件哈希未变。

| 条件 | DY设置 | alpha | 可用估计 | 参数MSE | 已知真值期望选择KL | 诊断性覆盖 |
|---|---|---|---|---|---|---|
'''+ '\n'.join(rows)+'''

所有设置各有500个点估计及诊断区间。150标签时，维数规则DY无需辅助预测即得到MSE0.101103、覆盖0.945091，与一步PPI的0.099168、0.945091接近；相同500个样本上的MSE差异0.001935，MCSE0.001399。维数规则DY的期望选择KL比一步PPI高0.001862，MCSE0.000398。两种度量均保留，不用单一指标宣布胜出。

半强度DY在强效用条件下覆盖仅0.624909，其固定支持人口目标的参数MSE为0.087418。这里估计的是原始DY及普通夹心诊断区间，保留收缩偏差，未冒充文献的高斯高维修正。误差和覆盖的Monte Carlo精度以500次重复为单位计算，不把11坐标当作5500个独立试验。

## 现在的判断

这些结果削弱了“需要AI辅助才有稳定估计”的解释。成熟惩罚方法可以不使用辅助预测得到有限且有竞争力的结果。论文主张已集中在指定校正目标的存在规律及实际拟合审计，不声称一套更优的新估计器。仍需评估该定理增量对JOCM读者的重要性，并完善实际设计下可执行的权重域审计；当前数学边界不能当作实际DCE样本量建议。

问卷300份真人来源记录与60份AI预测试继续分开，原始问卷未动。这两组新增比较使用已完成公开疫苗支持模拟的原抽样，不把它们写成新增人群或独立验证。近期官方作者指南仍缺可读核验，作者信息与声明事实也未代填。
'''
(r/'最近前驱补查与成熟基线比较_20261110.md').write_text(report)
print(json.dumps({'complete_replayed_draws':1000,'DY_fits':2000,'precision_rows':len(precision),'new_fits_in_report':False}))
