"""Write the completed power-tuning and informative-proxy extension."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit
from power_tuned_choice_audit import power_weight


def independent_weight_check():
    rng=np.random.default_rng(137);x=rng.normal(size=(80,4));y=rng.integers(0,2,80)
    g=.1+.8*rng.random(80);m=20;b=np.linspace(-.2,.1,4);mu=expit(x@b)
    _,check=power_weight(x,y,g,m,b);H=x.T@((mu*(1-mu))[:,None]*x)/len(x)
    inverse=np.linalg.inv(H);truth=x[:m]*(mu[:m]-y[:m])[:,None];pred=x*(mu-g)[:,None]
    aa=(truth-truth.mean(0))@inverse;bb=(pred[:m]-pred[:m].mean(0))@inverse
    pp=pred@inverse
    raw=np.mean(np.sum(aa*bb,axis=1))/((1+m/(len(x)-m))*pp.var(0,ddof=1).sum())
    error=abs(raw-check['raw_weight']);assert error<1e-12
    return {'independent_whitened_covariance_weight_difference':error,'check_scope':'deterministic synthetic spot check; no downloaded source executed'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out
    check=json.loads((out/'power_numeric_checks.json').read_text())
    assert check['completed_allocations']==check['requested_allocations']==500
    summary=pd.read_csv(out/'power_point_summary.csv');status=pd.read_csv(out/'power_status_counts.csv')
    paired=pd.read_csv(out/'power_paired_comparisons.csv')
    proxy=pd.read_csv(out/'informative_surrogate_exact_checks.csv')
    signal=pd.read_csv(out/'informative_surrogate_signal_frequency.csv')
    reference=json.loads((out/'power_reference_checks.json').read_text()) if (out/'power_reference_checks.json').exists() else independent_weight_check()
    figure,axes=plt.subplots(1,2,figsize=(10.4,4.1),layout='constrained')
    for epsilon,color in [(.1,'#167c80'),(.25,'#bc5a35'),(.4,'#5163a3')]:
        part=proxy[(proxy.labels==120)&(proxy.proxy_flip_probability==epsilon)].sort_values('dimension_label_ratio')
        axes[0].plot(part.dimension_label_ratio,part.exact_finite_probability,color=color,label=f'Proxy error {epsilon:g}')
        axes[0].plot(part.dimension_label_ratio,part.finite_frequency,'o',ms=3,color=color)
        axes[0].axvline(part.critical_ratio.iloc[0],ls=':',lw=1,color=color)
        part=signal[(signal.labels==120)&(signal.signal==3)&(signal.proxy_flip_probability==epsilon)].sort_values('dimension_label_ratio')
        axes[1].plot(part.dimension_label_ratio,part.finite_frequency,'o-',ms=3,color=color,label=f'Proxy error {epsilon:g}')
        axes[1].axvline(part.critical_ratio.iloc[0],ls=':',lw=1,color=color)
    axes[0].set_title('(a) Balanced null: exact and simulated')
    axes[1].set_title('(b) Utility signal gamma = 3: simulated')
    axes[0].set_ylabel('Probability of a finite corrected-loss minimizer')
    for ax in axes:
        ax.set_xlabel('Nonconstant utility contrasts / labels (p/m)');ax.set_ylim(-.03,1.03)
        ax.set_xlim(0,4.05);ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
        ax.legend(frameon=False,fontsize=8)
    figure.savefig(out/'informative_oracle_existence.svg');figure.savefig(out/'informative_oracle_existence.png',dpi=180);plt.close(figure)
    report='''# 加入权重调节后的结果与创新定位

2026年11月9日。全部500次分配已经完成；本阶段推进了理论和对照证据，独立原创性与投稿完成条件仍待核实。

## 较强对照的实际结果

沿用固定1200行疫苗选择、标签预算、来源定义五行组和预测拟合。新增目标严格采用PPI的辅助集平均损失，加上真人集的校正；它与上一阶段使用全体协变量池的GAI目标有区别。权重按PPI++及固定版本公开实现计算，裁剪到[0,1]。真人根初始化与原实现使用的PPI根初始化分别报告，没有隐式替换失败初始化。

表中为固定1200行真人投影目标的稳定化参数误差：100×mean(|估计−目标|/(1+|目标|))。无有限解的情形保留在失败表，误差均值仅对有限子集计算。预算单位为选择行。

| 预测信息 | 方法 | 50行 | 100行 | 150行 | 200行 |
|---|---|---:|---:|---:|---:|
'''
    labels={'human_converged':'真人充分收敛','ppi_lambda1':'PPI固定权重1','power_human_init':'权重调节，真人根初始化','power_ppi_init':'权重调节，PPI根初始化'}
    for pred in ['AI','X_only']:
        for method in labels:
            values=summary[(summary.prediction==pred)&(summary.method==method)].set_index('label_rows')
            report+='| '+pred+' | '+labels[method]+' | '+' | '.join(f'{values.loc[m,"error"]:.3f}' for m in [50,100,150,200])+' |\n'
    report+='\n完整结果同时给出权重分布与配对Monte Carlo标准误。已计算的权重与公开实现的协方差规范一致，独立白化内积公式核对误差为'+f"{reference['independent_whitened_covariance_weight_difference']:.2e}"+'。未执行下载的作者代码。\n\n'
    report+='| 预测信息 | 方法 | 预算 | 有限根 | 其他状态 |\n|---|---|---:|---:|---|\n'
    for (m,pred,method),part in status.groupby(['label_rows','prediction','method']):
        finite=int(part.loc[part.status.eq('finite_root'),'size'].sum())
        if finite<500:
            others=', '.join(str(r.status)+':'+str(r.size) for r in part.itertuples() if r.status!='finite_root')
            report+=f'| {pred} | {labels[method]} | {m} | {finite}/500 | {others} |\n'
    human=summary[summary.method.eq('power_human_init')]
    report+='\n真人根初始化的权重方法在'+str(int(human.finite_estimates.sum()))+'/'+str(int(human.allocations.sum()))+'个目标中得到有限根。这个已建立方法的表现必须保留。当前结果不支持把有限根问题写成现有方法无法处理的普遍缺陷。新贡献需要落在明确的定量存在条件与实际选择分析决策上。\n\n'
    report+='''## 理论候选已经扩大到有信息的代理

新增基准令代理Z以概率ε独立翻转真实二项选择Y，并使用冻结的条件均值预测g。它确实包含效用属性之外的选择信息。其总体最优PPI权重在辅助样本无限增大时趋于1，适用于固定0<ε<1/2。该结论直接使用现有方差最小化公式。

在高斯属性与平衡零效用信号下，校正损失的有限根边界为κ_c=1/[2π ε(1−ε)]，κ=p/m。精确有限标签概率通过三个二项分布和卡方分布的混合计算。对有信号的效用模型，边界由同一公式的信号项与代理Brier误差给出。它描述指定的无惩罚校正损失，不是所有估计方法的信息极限。

135组精确概率条件每组2000次直接生成X、Y、Z；ε=1/2退化到旧公式的最大误差低于2e-14。另完成162组非零信号条件，各2000次。理想预测与方差最优的极限权重不能自动保证比例增长维度下存在有限根。这是比错误常数预测更有意义的理论候选。

![有信息代理下的有限根概率](informative_oracle_existence.svg)

图中m=120。左图实线为精确有限样本概率，点为模拟频率；右图为非零信号的模拟频率。虚线为各代理误差率对应的比例维度极限。该图使用高斯二项模型和冻结的理想预测，不代表实际问卷的样本量要求。

有限辅助池、Gaussian支撑、冻结代理、独立标签、功效权重的极限均属于结果条件。这里没有证明高维插件权重必然趋于该总体最优值，也没有评估公开疫苗数据的区间覆盖率。真人问卷的300份招募记录继续作为独立应用实证；60份AI预测试不计入真人估计。

## 对投稿贡献的判断

本阶段给出了可复算的存在概率、更强的有信息代理情形，以及现有调权方法的完整对照。通用分离、Gaussian lift zonoid、PPI校正和方差最优调权继续明确归属于既有研究。当前的新候选是这些明确假设下的有限根分布与边界。短推论是否足以支撑JOCM方法论文，仍需要优先权核对与实用价值证据。

下一步应检验有限辅助池和实际DCE属性分布下的可迁移程度，比较现有一步估计与明确惩罚目标，补齐区间覆盖表现。投稿稿应围绕一个估计问题组织，不能把300份问卷、代理理论和代码审计并列当作三项创新。

对应证明见[第8节](existence_candidate_derivation.md)，预先写定的扩展方案见[权重对照计划](power_extension_plan.json)与[信号方案](surrogate_signal_plan.json)。主文献为[PPI++](https://arxiv.org/html/2311.01453v2)及[固定版本实现](https://github.com/aangelopoulos/ppi_py/blob/3d1f0c668444907b39bd045cb9dd38e479ce7dd6/ppi_py/ppi.py)。
'''
    (out/'权重对照与信息代理进展.md').write_text(report)
    print(json.dumps({'completed_allocations':500,'surrogate_exact_cells':len(proxy),'surrogate_signal_cells':len(signal),'goal_complete':False}))


if __name__=='__main__':main()
