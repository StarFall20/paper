"""Create aggregate audit tables, a research figure, and the current report."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    out=a.out;mc=pd.read_csv(out/'exact_existence_Monte_Carlo.csv')
    phase=pd.read_csv(out/'candidate_existence_phase.csv')
    valid=mc.Monte_Carlo_SE_at_exact>1e-10
    maxz=float((mc.loc[valid,'absolute_discrepancy']/mc.loc[valid,'Monte_Carlo_SE_at_exact']).max())
    mc['exact_binomial_p']=[binomtest(int(r.finite_count),int(r.replications),float(r.exact_finite_probability)).pvalue for r in mc.itertuples()]
    mc.to_csv(out/'exact_probability_consistency_checks.csv',index=False)
    min_adjusted=min(1.,len(mc)*float(mc.exact_binomial_p.min()))
    checks=json.loads((out/'matched_numeric_checks.json').read_text())
    assert checks['completed_allocations']==checks['requested_allocations']==500
    assert checks['fixed_clipping_comparisons']==2000
    rays=json.loads((out/'nonexistence_ray_checks.json').read_text())
    assert len(rays['cases'])==7 and rays['all_exterior_cases_independently_separated']
    summary=pd.read_csv(out/'point_estimation_summary.csv')
    order=['human_converged','human_Adam200','naive_hard_AI','clipped_AI_self_training','corrected_AI','corrected_X_only']
    table=summary.pivot(index='method',columns='label_rows',values='MAPE_prefix_projection').reindex(order)
    table.to_csv(out/'fixed_target_error_table.csv')
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.3),layout='constrained')
    colors={.5:'#167c80',1.:'#bc5a35'}
    for q in [.5,1.]:
        s=mc[(mc.labels==120)&(mc.q==q)].sort_values('dimension_label_ratio')
        axes[0].plot(s.dimension_label_ratio,s.exact_finite_probability,color=colors[q],label=f'Exact, q={q:g}')
        axes[0].errorbar(s.dimension_label_ratio,s.finite_frequency,
            yerr=1.96*np.sqrt(s.finite_frequency*(1-s.finite_frequency)/2000),
            fmt='o',ms=3,color=colors[q],capsize=2,alpha=.8)
        critical=2/(np.pi*(1+4*(q-.5)**2))
        axes[0].axvline(critical,ls=':',color=colors[q],lw=1)
    axes[0].set_title('(a) Balanced null; m = 120')
    axes[0].set_ylabel('Probability of a finite minimizer')
    axes[0].legend(frameon=False,fontsize=9)
    names={'oracle_probability':'Oracle probability','hard_threshold':'Hard threshold','constant_half':'Constant 0.5'}
    # Predictor codes are fixed in the simulation program.
    available=phase.predictor.unique().tolist()
    if 'oracle_prob' in available:names['oracle_prob']=names.pop('oracle_probability')
    for key,label in names.items():
        s=phase[(phase.labels==120)&(phase.signal==3)&(phase.predictor==key)].sort_values('dimension_label_ratio')
        assert len(s)>0,(key,available)
        line=axes[1].plot(s.dimension_label_ratio,s.finite_frequency,'o-',ms=3,label=label)[0]
        axes[1].axvline(s.critical_ratio_candidate.iloc[0],ls=':',color=line.get_color(),lw=1)
    axes[1].set_title('(b) Nonzero signal; gamma = 3, m = 120')
    axes[1].legend(frameon=False,fontsize=9)
    for ax in axes:
        ax.set_xlabel('Nonconstant dimensions / labelled choices (p/m)')
        ax.set_ylim(-.03,1.03);ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    fig.savefig(out/'finite_solution_benchmark.svg');fig.savefig(out/'finite_solution_benchmark.png',dpi=180);plt.close(fig)
    integration=pd.read_csv(out/'population_moment_numeric_checks.csv')
    audit={'completed_allocations':500,'matched_method_budget_fits':len(pd.read_csv(out/'trial_diagnostics.csv')),
        'clipping_identity_max_error':0.,'exterior_cases_separately_verified':7,
        'minimum_separation_gap':min(c['separation_gap'] for c in rays['cases']),
        'maximum_exact_probability_MC_standardized_difference':maxz,
        'tail_warning':'Normal-standardized discrepancies are unstable near probability0 or1; the largest comes from one rare event',
        'minimum_exact_binomial_p':float(mc.exact_binomial_p.min()),
        'minimum_Bonferroni_binomial_p':min_adjusted,
        'maximum_population_root_gradient':float(integration.moment_gradient_max.max()),
        'maximum_ray_slope_error':float(abs(integration.ray_slope_numeric-integration.ray_slope_theory).max()),
        'independent_novelty_proven':False,'submission_readiness_proven':False}
    (out/'report_checks.json').write_text(json.dumps(audit,indent=2)+'\n')
    report='''# 模型重建进展：从AI预测试来源到有限解问题

2026年11月9日。当前成果包括公开数据重分析和一个待核实原创性的理论候选；投稿目标尚未完成。

## 现有问卷的用途已经确定

用户提供的问卷星原文明确写出由AI样本作答，适用于预研究和预测试。原始360份导出完整保留。主分析使用300份普通招募来源记录，两版各150份，共2400次选择；60份AI预测试另列。300份的来源分类不自动证明代表性或每份答卷的有效性。渠道与版本重合，不能据此估计随机版本效应。

真人模型、个体内条件模型和AI配对审计已经完成。这些工作保留了问卷的应用价值；旧纤维秩理论的独立贡献定位已撤回。

## 公开疫苗选择数据的可核对结果

使用Lu等人公开仓库的固定提交，读取4605行真人二项选择和前1200行对应AI选择。文件不含可认证的受访者ID；五行一组沿用作者脚本的处理惯例。原研究排除始终不选择疫苗的受访者，不能把本文件当作原始全体样本。没有执行下载的作者程序。

发布的参考参数与200步Adam输出的最大差为1.49e-7。在相同4605行数据和相同无惩罚二项损失下，充分收敛可再降低总负对数似然7.013964，最大参数差0.185492。这个比较识别出参考解的未收敛，不证明论文全部改进失效。

代码对校正伪标签逐项裁剪。其代数后果对任意选项数成立：已标注记录退回真人one-hot标签，未标注记录保留预测概率。2000组固定预测、固定分配的比较全部精确满足该恒等式。

## 500次固定目标重分析

每次随机排列前1200行的240个来源定义五行组，在50、100、150、200行标签预算下比较六种估计。所有预算共用同一1200行真人投影目标；这改变了作者随预算改变候选池的规则。它是控制重分析，不是原文表格复现。

下表为100×mean(|估计参数−目标参数|/(1+|目标参数|))。CSV保留原字段前缀MAPE，分母加一的定义在此明确；它不是标准百分比相对误差。数值是500次分配均值，无有限解的方法按有限子集报告，同时完整列出失败次数。单位为标签行数，不是受访者人数。

| 方法 | 50行 | 100行 | 150行 | 200行 |
|---|---:|---:|---:|---:|
'''
    labels={'human_converged':'真人充分收敛','human_Adam200':'真人200步Adam','naive_hard_AI':'直接加入AI硬标签',
            'clipped_AI_self_training':'裁剪后的软标签训练','corrected_AI':'未经裁剪的AI校正','corrected_X_only':'未经裁剪的属性校正'}
    for method in order:
        report+='| '+labels[method]+' | '+' | '.join(f'{table.loc[method,m]:.3f}' for m in [50,100,150,200])+' |\n'
    report+='''
在50行预算下，AI校正有5/500次、属性校正有2/500次目标矩落在可行均值支撑之外，其余预算均找到有限根。7个情形分别用另一线性规划找到了正分离间隙，并沿射线直接核对损失向负无穷下降；最小间隙为'''+f"{audit['minimum_separation_gap']:.9f}"+'''。这些失败来自目标本身，不应通过增加迭代后把最后一步参数当作解。

裁剪方法在当前点误差指标上表现更好，同时不满足未裁剪的校正得分方程。两项结果同时保留。当前结果没有比较PPI++权重调节、嵌入预测或区间覆盖率，不能推出AI信息在一般条件下无价值。

## 当前值得深化的单一问题

研究问题是：AI辅助校正后的选择损失，何时有有限参数解？在高斯属性、独立标签、固定低维预测子空间和无限辅助池的基准下，已推导精确有限样本公式，以及由信号强度和预测Brier误差共同决定的存在边界。有限辅助池的严格两侧极限已补上证明。

直接从X、Y生成的模拟包含160个精确概率条件、480个信号/预测条件，每个2000次；精确概率与模拟频率的最大绝对差为'''+f"{mc.absolute_discrepancy.max():.5f}"+'''。接近概率1的条件出现一次稀有事件，正态标准化偏差达到7.43，因此另以精确二项分布检查；160项Bonferroni校正后的最小p值为'''+f"{min_adjusted:.3f}"+'''。这组模拟没有提供公式错误的统计证据，也不构成公式正确性的证明。9个总体内部根、6个外部射线的独立积分检查已完成。高斯支撑形状、分离LP、普通logit相变和无偏风险问题均有既有文献，不能列作本项目创新。

候选增量集中在精确存在概率与预测误差相关的非零信号边界。它来自一个明确估计问题，未增加术语框架；现有推导也可能被评价为既有几何的短推论。需要继续核对最接近的文献，并证明该边界对选择分析有实用价值。当前不宣称已达到JOCM投稿门槛。

下一项实质工作是比较权重调节与明确的有限解处理策略，补齐有限辅助样本的实用区间和选择任务解释。若候选边界已被发表或不足以改变分析决策，应调整贡献定位。真人问卷应用作为独立、清楚限定的实证材料保留，不把两条线拼成一个理论。

来源、证明、全部计划及重现入口见[README](README.md)和[文献边界](literature_boundary.md)。
'''
    (out/'公开数据重分析与创新进展.md').write_text(report)
    print(json.dumps(audit))


if __name__=='__main__': main()
