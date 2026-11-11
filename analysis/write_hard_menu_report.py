"""Scoped report and figure from completed hard-menu calculations."""
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
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    validation=json.loads((a.out/'hard_menu_complete_validation.json').read_text());assert validation['state']=='complete' and validation['unresolved']==0
    population=pd.read_csv(a.out/'hard_menu_primary_values.csv');reference=pd.read_csv(a.out/'hard_menu_variance_reference.csv')
    phase=pd.read_csv(a.out/'hard_menu_matched_phase_summary.csv')
    selected=reference[(reference.alternatives==3)&(reference.gamma==1.5)&(reference.tau==1)&(reference.epsilon==.25)&(reference.seed==2026111171)].iloc[0]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(11,3.9),sharex=True,sharey=True)
    methods=[('ordinary','Ordinary logit','#444444'),('unit','Unit correction','#CC6677'),('oracle_null_slope_variance','Variance reference','#0072B2')]
    for ax,(method,title,color) in zip(axes,methods):
        part=phase[phase.method.eq(method)]
        for m,marker,style in [(40,'o','--'),(80,'s','-')]:
            s=part[part.label_menus==m].sort_values('dimension_ratio')
            n=s.replications.to_numpy();freq=s.finite_frequency.to_numpy();z=1.959963984540054
            center=(freq+z*z/(2*n))/(1+z*z/n)
            radius=z*np.sqrt(freq*(1-freq)/n+z*z/(4*n*n))/(1+z*z/n)
            errors=np.maximum(np.vstack([freq-center+radius,center+radius-freq]),0.)
            ax.errorbar(s.dimension_ratio,s.finite_frequency,yerr=errors,color=color,marker=marker,ls=style,capsize=3,label=f'{m} labelled menus')
        ax.axvline(float(part.iloc[0].critical_ratio),color=color,ls=':',label='Candidate boundary')
        ax.set(title=title,xlabel='Orthogonal slopes per labelled menu',ylim=(-.02,1.02),xlim=(.10,.78));ax.grid(alpha=.15)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Finite minimizer frequency')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.025))
    fig.suptitle('Hard proxy labels: matched menus and dimensions across methods',fontsize=11)
    fig.tight_layout(rect=(0,.08,1,.95));fig.savefig(a.out/'hard_menu_matched_phase.png',dpi=220,bbox_inches='tight');plt.close(fig)
    wrong=[]
    for J in [2,3]:
      for gamma in [0.,1.5]:
       for tau in [1.,4.]:
        f=population[(population.alternatives==J)&(population.gamma==gamma)&(population.tau==tau)&(population.epsilon==1)]
        ordinary=float(f[f.event.eq('ordinary')].iloc[0].critical_ratio);limit=float(f[f.event.eq('positive_limit')].iloc[0].critical_ratio)
        wrong.append([J,f'{gamma:g}',f'{tau:g}',f'{ordinary:.6f}',f'{limit:.6f}',f'{limit-ordinary:.6f}'])
    wrong_table=markdown(['选项数','信号','辅助比','普通边界','正权重极限','差值'],wrong)
    wide=phase.pivot(index=['label_menus','dimension_ratio'],columns='method',values='finite_frequency').reset_index()
    phase_table=markdown(['标记菜单数','噪声属性/菜单','普通logit','单位校正','方差参照'],[[int(r.label_menus),f'{r.dimension_ratio:.2f}',f'{r.ordinary:.3f}',f'{r.unit:.3f}',f'{r.oracle_null_slope_variance:.3f}'] for r in wide.itertuples()])
    report=f'''# 硬类别预测的模型边界与创新复核

2026年11月11日。本轮继续研究同一预测校正目标，把预测向量扩展到单纯形边界，包括直接输出一个类别的预测。当前32页主稿与原模拟保持原样；新的推导和完整记录保存在独立研究补充中。

## 已有研究与当前候选增量

多类别预测校正已经有multiPPI++，共同属性系数的选择logit也已进入已有调查方法。我们补读了Fan等（2024）的Section3.2与AppendixA.2–A.3，以及Ye和Yoganarasimhan（2026）的AppendixJ。模型与校正方程需要明确归于前人。当前候选贡献收敛为有限辅助样本下的存在性边界，以及该边界随菜单结构和校正权重变化的规律。

来源：[multiPPI++原文](https://soc.washington.edu/sites/soc/files/documents/research/ppi_colm.pdf)；[共同系数MNL及校正方程](https://arxiv.org/html/2604.17267v1)。近年JOCM的LLM辅助规格研究已有正文背景引用；它与用预测响应构造估计方程属于不同任务。

## 一个可核对的菜单差别

保持有限信号、高斯剖面、独立样本与固定预测条件。预测可以是任意概率单纯形向量。预测并非完全正确时，标记残差的平均平方范数为正，原边界推导的尺度控制仍成立。完全正确的硬预测则直接化为普通加权合并似然，另行处理。

在正权重趋零的候选边界中，二选项的严格扩张条件是预测给真实类别分配正概率的事件具有正概率。总是选错类别的二元硬预测恰好回到普通logit边界。三选项以上的切锥有至少两个未选类别；高斯投影在不同竞争类别之间变化，使相应导数对任意固定预测均为正。推导保留了完整的条件、投影区域与零误差端点。

下面是“每次预测都选错”的总体积分核对。它展示几何性质，不建议使用已知错误的预测，也不把有限解扩张当作精度改善。

{wrong_table}

全部320次积分覆盖80个事件条件、两档分辨率和两个扰动种子。全部分辨率/种子的最大范围为{validation['maximum_resolution_and_seed_range']:.8f}，最高分辨率两种子的最大差为{validation['maximum_fine_seed_range']:.8f}。二元总错条件的极限差为零；三选项总错的差约0.006–0.023。这一数值幅度仍需放在实际方法价值中判断。积分差异不是严格误差界。

## 按实际预测计算已有方差参照

硬预测的参照权重需要使用它自身的得分协方差。三选项、信号1.5、错误率0.25、辅助比1时，已有控制变量准则对一个正交属性系数给出的总体权重约为{selected.clipped_variance_power:.6f}。该标量准则属于PPI++已有方法，涉及固定维数方差；它没有给出所有系数或高维风险的最优性。

在这个设置下，普通、单位校正、方差参照的候选边界分别约0.5594、0.2164、0.7039。有限验证让所有方法共享同一菜单样本、选择、硬预测和属性维数。两个标记预算各500次独立抽样，共1000次模型样本、9000次支持判定。完整有限解频率如下。

{phase_table}

![相同样本与维数下的有限解频率](hard_menu_matched_phase.png)

误差条为500次重复的95% Wilson二项区间，在频率为零或一时也保留不确定范围。方差参照使用已知模拟总体量，不是本轮新提出的数据调参算法。普通logit的分离对应可行支撑边界；保存的{validation['ordinary_boundary_directions_checked']}个方向已逐一用原始模拟设计验证。没有把这些边界改记为校正矩的外部不可行。边界附近的有限预算偏移保留原值；这些点没有建立收敛速度或精确定位渐近边界。该实验只测可用性，系数误差、区间和训练成本结论继续使用此前各自的比较。完整配对差异与Monte Carlo标准误另存于CSV。

## 数值修正与剩余工作

80个硬预测有限设计提供240个原始矩/残差规划核对，另有240个独立NNLS投影导数核对。三个有限边界来自一次标记样本中预测恰好全对的设计；三个正分离间隔已单独确认。

有限模拟最初在第801个模型样本的一次独立残差规划中遇到HiGHS Unknown。原校正目标已有负衰退代价，独立平移变量后用内点法确认不可行。随后修复了诊断记录字段重名，并从已完成800个样本的检查点续跑。两次中断的源码、7200条检查点和具体模拟设计均保留；最终前7200条与检查点逐条完全一致。最终主分类与独立交叉核对均没有未解决状态。

Google Scholar本轮未能取回结果，Scopus返回403，不能称为已完成这两个库的系统检索。另有一条多项logit相变论文线索尚未取得官方正文。完整近邻重合审查、主稿整合与方法价值判断继续推进。300份真人问卷与60份AI预测试的来源分类保持不变；本轮没有增加真人或LLM答卷。
'''
    (a.out/'硬类别预测与创新复核_20261111.md').write_text(report)
    print(json.dumps(dict(report_written=True,figure_written=True,phase_rows=validation['phase_rows_reconciled'])),flush=True)


if __name__=='__main__':main()
