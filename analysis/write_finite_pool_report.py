"""Summarize completed finite-pool validation and the finite-ratio candidate."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out
    run=json.loads((out/'finite_pool_numeric_checks.json').read_text());phasecheck=json.loads((out/'finite_ratio_phase_status.json').read_text())
    assert run['completed_replications']==run['requested_replications']==500
    assert phasecheck['completed_cells']==phasecheck['requested_cells']==80
    data=pd.read_csv(out/'finite_pool_trial_diagnostics.csv');summary=pd.read_csv(out/'finite_pool_summary.csv')
    phase=pd.read_csv(out/'finite_ratio_phase_checks.csv');assert len(data)==42000
    assert data.excess_logloss.min()>-1e-12 and int(phase.unresolved.sum())==0
    labels={'human_MLE':'Human MLE','PPI_unit':'PPI, unit power','PPI_power_human':'Power, human start',
        'PPI_power_ridge':'Power, ridge start','PPI_one_step_ridge':'One step, ridge start',
        'PPI_ridge_unit':'Ridge PPI','soft_oracle_pool':'Oracle soft pooling'}
    order=list(labels)
    fig,axes=plt.subplots(1,2,figsize=(11.8,5.1),layout='constrained',sharey=True)
    for ax,m in zip(axes,[20,150]):
        part=summary[(summary.label_rows==m)&(summary.auxiliary_rows==600)&(summary.epsilon==.25)].set_index('method').loc[order]
        y=np.arange(len(order));ax.barh(y-.16,part.conditional_marginal_coverage,height=.28,label='Conditional coverage',color='#267e9c')
        ax.barh(y+.16,part.interval_and_marginal_coverage,height=.28,label='Interval issued and covers',color='#c26939')
        ax.axvline(.95,color='black',ls=':',lw=1);ax.set_xlim(0,1.03);ax.set_title(f'{m} labelled choices; 600 auxiliary')
        ax.set_xlabel('Average marginal coverage over 11 fixed coordinates');ax.spines[['top','right']].set_visible(False)
        ax.set_yticks(y,[labels[k] for k in order]);ax.grid(axis='x',alpha=.15)
    axes[0].invert_yaxis()
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='outside lower center',ncol=2,frameon=False,fontsize=8)
    fig.savefig(out/'finite_pool_coverage.svg');fig.savefig(out/'finite_pool_coverage.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10.4,7),layout='constrained',sharey=True)
    for i,tau in enumerate([1,4]):
        for j,e in enumerate([.1,.25]):
            ax=axes[i,j]
            for power,color in [('unit','#c26939'),('variance_optimal','#267e9c')]:
                for m,marker in [(40,'o'),(80,'s')]:
                    s=phase[(phase.tau==tau)&(phase.epsilon==e)&(phase.power_name==power)&(phase.labels==m)].sort_values('dimension_label_ratio')
                    ax.plot(s.dimension_label_ratio,s.finite_frequency,marker=marker,ms=3,color=color,
                        alpha=.65 if m==40 else 1.,ls='--' if m==40 else '-',
                        label=f'{"Unit power" if power=="unit" else "Optimal power"}; m={m}')
                ax.axvline(s.critical_ratio.iloc[0],color=color,ls=':',lw=1)
            ax.set_title(f'N/m={tau}; proxy error={e:g}');ax.set_ylim(-.03,1.03);ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
            ax.set_xlabel('Nonconstant dimensions / labels (p/m)')
    axes[0,0].legend(frameon=False,fontsize=7);axes[0,0].set_ylabel('Feasible-moment frequency');axes[1,0].set_ylabel('Feasible-moment frequency')
    fig.savefig(out/'finite_ratio_boundary_checks.svg');fig.savefig(out/'finite_ratio_boundary_checks.png',dpi=180);plt.close(fig)
    # Retain tail summaries alongside unstable means in the low-label regime.
    tails=data.groupby(['label_rows','auxiliary_rows','epsilon','method'],as_index=False).agg(
        median_MSE=('MSE','median'),q90_MSE=('MSE',lambda s:s.quantile(.9)),
        median_excess_logloss=('excess_logloss','median'),q90_excess_logloss=('excess_logloss',lambda s:s.quantile(.9)))
    tails.to_csv(out/'finite_pool_tail_summaries.csv',index=False)
    report='''# 有限辅助样本、区间可靠性与理论扩展

2026年11月9日。当前完成了已知真值的有限样本验证和有限辅助比例的理论核对。投稿原创性与完整稿件仍需完成审视。

## 实验回答了什么

实验从公开疫苗数据的1200个属性组合中独立抽取选择行，固定此前收敛的11维参数作为模拟真值，重新生成选择和有信息的代理。它保留真实属性支撑，使用已知的数据生成过程检验方法。它没有增加真人样本，也不是原问卷的新增实证。标签预算20、50、150，辅助池150、600，代理翻转率0.1、0.25，共12个条件，每条件500次。所有方法使用相同随机样本。

七种方法包括真人MLE、固定权重PPI、两种初始化的调权根、一步估计、明确加惩罚的PPI，以及理想软标签合并。后者知道正确的条件均值，是强参照；其表现不能直接推广到未知或有偏的AI预测。

## 有限估计与区间覆盖要分别判断

下表固定辅助池600、代理错误率0.25。覆盖率是11个预先固定坐标的平均；“发出且覆盖”以全部500次重复为分母，缺失区间计为未完成覆盖任务。另一列只在已发出的区间中计算。两个分母同时保留，避免成功子集掩盖失败。

| 真人标签行 | 方法 | 有限估计 | 区间可用 | 条件覆盖率 | 发出且覆盖 | 参数MSE |
|---|---|---:|---:|---:|---:|---:|
'''
    for m in [20,150]:
        part=summary[(summary.label_rows==m)&(summary.auxiliary_rows==600)&(summary.epsilon==.25)].set_index('method')
        for method in order:
            r=part.loc[method];report+=f'| {m} | {labels[method]} | {int(r.estimates)}/500 | {int(r.intervals)}/500 | {r.conditional_marginal_coverage:.3f} | {r.interval_and_marginal_coverage:.3f} | {r.MSE:.4f} |\n'
    report+='''
![有限估计与覆盖任务](finite_pool_coverage.svg)

这里的一步估计在低标签预算下能够给出有限点估计，其名义95%区间覆盖仍需单独检验。加惩罚方法优化的是明确改变后的目标；表中的Wald区间没有校正惩罚偏差。当前真值较接近零，该结果不证明惩罚方法具有普遍覆盖保证。理想软标签的优势依赖正确条件均值这一额外信息。

全部12条件、每个坐标、配对误差和Monte Carlo标准误已保存。低预算MLE和部分调权根的均方误差受极端值影响，另列中位数与90%分位数。条件均值、是否发出区间和数值失败都不作为独立创新。独立的截距闭式方差与一维惩罚根核对已完成；一维检查修复了协方差函数对标量数组的兼容性，11维原实验结果不受影响，原运行与修正版哈希分别保留。

## 更实质的理论候选：有限辅助比例与权重

原推导使用无限辅助样本。新推导同时保留τ=N/m和正的校正权重λ。在平衡零信号、高斯属性、固定零截距和冻结代理条件下，有限根事件可以改写为随机高斯线性映射在一组区间上的可行性。利用既有的随机子空间定理，可将边界化成一个可计算的一维凸最小化问题。

通用圆锥几何与随机子空间定理来自ALMT2014；本文候选增量是这一个校正选择估计器的显式τ、代理误差、权重联合边界。其推导和条件见[有限比例证明](finite_ratio_boundary_candidate.md)。新增80个条件，每条件500次直接生成高斯属性、选择和代理，并进行线性规划。100个独立样本同时用原始矩支撑与区间两种表达核对，43个内部、57个外部情形全部对应。该证据支持推导和实现，不能替代原创性核对。

![有限辅助比例的边界核对](finite_ratio_boundary_checks.svg)

图中的点是有限m的可行频率，虚竖线是候选渐近边界；它们不应被要求在有限样本中呈现完美阶跃。m=40和80分别显示，单位权重与总体方差最优权重分开比较。总体最优权重的公式属于现有PPI理论。当前80个条件没有数值未决情形。

一个可核对的例子：τ=1、代理错误率0.25时，单位权重的候选临界p/m约0.355；现有方差最优权重λ=0.5对应约0.603。无限辅助池下单位权重的值约0.849。辅助样本数量与权重对可计算条件有实质影响，这比指出某次优化失败更接近一个有内容的方法学结论。它仍限定于该高斯基准，不能直接充当实际离散问卷的样本量规则。

## 当前判断与下一步

有限辅助比例的显式边界扩大了理论内容；已知真值实验也补上了一步估计和区间覆盖对照。现有方法的成功与失败均进入结果，原始300份真人招募记录继续保留为应用实证，60份AI预测试单列。

随后完成了非零效用信号和自由截距的条件推导，并完成80个条件、每条件300次的数值实验，见[信号扩展报告](非零效用信号与有限样本边界.md)。该扩展仍需独立证明与优先权审视；非理想预测和实际分析决策的价值还需验证。接下来完善这些科学证据，并将正文、附录和复现入口整理为一份连贯的投稿稿。当前不宣称达到JOCM投稿门槛。
'''
    (out/'有限样本验证与理论扩展.md').write_text(report)
    savechecks={'finite_pool_replications_per_cell':500,'finite_pool_cells':12,'method_condition_outcomes':len(data),
        'finite_ratio_cells':len(phase),'finite_ratio_replications_per_cell':500,'finite_ratio_unresolved':int(phase.unresolved.sum()),
        'intercept_covariance_checks':json.loads((out/'finite_pool_covariance_verification.json').read_text()),
        'independent_novelty_verified':False,'submission_goal_complete':False}
    (out/'finite_pool_report_checks.json').write_text(json.dumps(savechecks,indent=2)+'\n')
    print(json.dumps(savechecks))


if __name__=='__main__':main()
