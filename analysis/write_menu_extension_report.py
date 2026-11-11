"""Figures and scoped progress reporting from completed menu calculations."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,row))+'|' for row in rows])


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    phase=pd.read_csv(a.out/'menu_phase_summary.csv');pop=pd.read_csv(a.out/'menu_population_primary_values.csv')
    valid=json.loads((a.out/'menu_independent_validation.json').read_text())
    dce=json.loads((a.out/'balanced_dce_cancellation_checks.json').read_text())
    assert len(phase)==12 and phase.replications.eq(500).all() and phase.unresolved.sum()==0
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(9.4,4.1),sharey=True)
    ordinary=float(pop[(pop.gamma==1.5)&pop.event.eq('ordinary')].iloc[0].critical_ratio)
    for ax,power,color in zip(axes,[1.,.5],['#444444','#0072B2']):
        part=phase[phase.power==power]
        for m,marker,line in [(40,'o','--'),(80,'s','-')]:
            cells=part[part.label_menus==m].sort_values('orthogonal_dimension_ratio')
            ax.errorbar(cells.orthogonal_dimension_ratio,cells.finite_frequency,yerr=1.96*cells.binomial_MCSE,
                        color=color,marker=marker,ls=line,capsize=3,label=f'{m} labelled menus')
        ax.axvline(float(part.iloc[0].critical_ratio),color=color,ls=':',label='Corrected boundary')
        ax.axvline(ordinary,color='#AA3377',ls='-.',label='Ordinary boundary')
        ax.set(title=f'Correction power {power:g}',xlabel='Orthogonal slopes per labelled menu',ylim=(-.02,1.02))
        ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Finite minimizer frequency')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,-.025),frameon=False)
    fig.suptitle('Three alternatives with a utility signal and estimated constants',fontsize=11)
    fig.tight_layout(rect=(0,.075,1,.95));fig.savefig(a.out/'menu_phase.png',dpi=220,bbox_inches='tight');plt.close(fig)
    rows=[]
    for gamma in [0.,1.5]:
      ordinary=float(pop[(pop.gamma==gamma)&pop.event.eq('ordinary')].iloc[0].critical_ratio)
      for tau in [1.,4.]:
        subset=pop[(pop.gamma==gamma)&(pop.tau==tau)]
        fixed=subset[subset.event.eq('fixed_positive')].set_index('power')
        limiting=float(subset[subset.event.eq('positive_limit')].iloc[0].critical_ratio)
        rows.append([f'{gamma:g}',f'{tau:g}',f'{ordinary:.4f}',f'{fixed.loc[1.,"critical_ratio"]:.4f}',
                     f'{fixed.loc[tau/(1+tau),"critical_ratio"]:.4f}',f'{limiting:.4f}'])
    population_table=table(['信号γ','辅助菜单比τ','普通logit','单位功率','既有方差参照功率','正功率趋零'],rows)
    phase_table=table(['标记菜单数','功率','独立噪声属性数','有限解频率','Monte Carlo标准误'],
                      [[int(r.label_menus),f'{r.power:g}',int(r.orthogonal_slopes),f'{r.finite_frequency:.3f}',f'{r.binomial_MCSE:.4f}'] for r in phase.itertuples()])
    report='''# 多项选择推广与真实DCE可用边界

2026年11月11日。本轮把原有校正logit的有限解问题推进到整个多项选择菜单，保留所有选项概率之和为1的约束。核心损失、真人标签目标和两个样本的归属均保持一致。当前结果是同一问题的候选理论推广，尚未改写此前32页主稿，也未据此宣布达到投稿门槛。

## 这轮取得了什么

三选项模型可同时估计两个选项常数、一个共同效用信号斜率以及增长维数的共同属性斜率。推导将二元区间换成菜单概率单纯形，并保留原有补偿目标。候选边界仍是低维的极小极大式；在本轮对称模型中只需优化一个信号乘子和一个尺度。还推导了正功率趋零时的菜单切锥及严格高于普通logit边界的导数条件。完整模型、假设与证明步骤在 `menu_boundary_derivation.md`。

这一步没有引入新的框架名、训练算法或通用线性规划方法。高斯锥运动学、概率支撑与单纯形投影保留已有方法归属。待进一步审查的增量是特定补偿目标在整张菜单上的显式概率边界及其全部正功率结论。问题和损失的一致性可以核对；相对全部文献的原创性仍需继续审查。

## 候选概率边界

下表均为三选项、代理错误率0.25的高斯菜单模型。维数比是共同噪声属性数除以独立标记菜单数；信号斜率与两个选项常数另外估计。参照功率是既有固定维数方差基准τ/(1+τ)。这些比率不能直接转为真人问卷所需人数。

'''+population_table+'''

例如γ=1.5、辅助与标记菜单数量相同，单位功率边界约0.4167，普通logit约0.5590，参照功率约0.7024。单位功率可能缩小可估计区域，较小功率能扩大区域。这里评价的是指定无惩罚目标是否有有限解；误差、区间有效性与训练成本仍有各自的评价要求。

全部72次积分计算保留两档分辨率和两个独立扰动种子，覆盖18个事件条件。最高分辨率的两个种子间最大差小于0.000239；全部分辨率与种子的最大范围小于0.001940。这是数值稳定性诊断，没有建立严格积分误差界。十二个二元退化对照匹配了信号方差约定，最大边界差小于0.000186。低功率0.1在有限积分点上可与极限值数值重合，不能据此断言其在精确总体中相等。

## 有限菜单验证

在上述非零效用信号、辅助比1的条件下，两个预算各生成500次独立菜单样本。两个功率及所有显示维数在每次重复中共享菜单、选择和代理类别，共1000次独立模型样本、6000次有限解判定、12个汇总单元。另有120次独立残差线性规划交叉核对。没有未解决状态，完整频率如下。

'''+phase_table+'''

![三选项有限解频率](menu_phase.png)

图中的误差条为500次重复的近似95%二项Monte Carlo区间，竖线为候选总体边界。边界附近有限频率仍有偏移；两个预算的结果保留原值，没有按理论线调整。不同维数共享重复，表格行并非独立样本。

全部6000条记录及Monte Carlo标准误已独立对账。每次样本中，增加共同噪声属性后的有限解事件满足嵌套关系，共核对2000组。十二个独立完整约束锥二次规划检查了含全部选项常数与信号约束的投影计算，数值误差和优化器状态均在记录中保留。

## 对现有真人问卷的实际核对

两版合计300条真人招募记录，60条AI预测试继续排除于真人标签之外。按版本分别分析，使用AI预测试的题目层面频率构造固定预测。真人按整个人划分为标记与辅助两组，两组都完整重复同一套八道题。此时每道题及其固定预测在两组中的经验频率相同。

对任意参数，预测损失均值相同，校正差严格为零。因此，这种设置下的校正目标等于标记真人的普通菜单logit目标。120个整人划分、四个功率、三个参数点的实际核对中，目标差小于9.2e-15、梯度差小于5.5e-14、Hessian差小于2.0e-13；所有分配的普通目标都有内部可行矩，复用其有限根时梯度残差低于9e-10。

这个抵消条件是可计算的有限设计性质。它没有新增真人，也没有检验跨渠道随机分配、因果版本效应或总体代表性。当前高斯概率边界不适用于这些重复固定菜单。原问卷的真人偏好应用继续保留；本轮识别的是菜单层面预测在此校正目标中的作用边界。

## 保留的问题与修正

初版验证程序错误地把可行边界当成内部可行。原四选项案例的校正类别矩为(7/24,0,17/48,17/48)，正好有一个类别为零；有限softmax常数不能给它零概率。该边界例子、失败版本和错误记录均保留，未改成外部不可行案例。

独立通用二次规划的坐标停止精度也不能代替投影恒等式。其最大坐标差约2.7e-7、目标差约1.1e-13；另一套标量根法与单纯形投影相差小于1.7e-15。两个误差均照实报告。锥验证器最后一次仅修复了NumPy计数的JSON输出类型，统计抽样和已保存的二次规划结果不受影响。

目前仍需复核这个菜单推广的完整证明、近邻理论重合与足够的方法价值，再决定怎样将其融入主稿。二元主稿的训练成本负面结果继续有效。通用LP、平衡菜单抵消恒等式以及这些数值检查均不单独当作原创贡献或投稿门槛证据。
'''
    (a.out/'多项选择推广与真实DCE核验_20261111.md').write_text(report)
    print(json.dumps(dict(report_written=True,figure_written=True,finite_phase_rows=valid['all_phase_rows_reconciled'],
                         independent_QPs=valid['independent_full_constrained_cone_QPs'],human_records_used_for_design_identity=dce['human_records'])),flush=True)


if __name__=='__main__':main()
