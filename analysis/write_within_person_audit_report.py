"""Write the exploratory within-person assumption audit from aggregate outputs."""
from pathlib import Path
import argparse, json
import pandas as pd


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    fits=json.loads((a.out/'conditional_fixed_effect_fits.json').read_text())
    checks=json.loads((a.out/'audit_checks.json').read_text())
    contrast=pd.read_csv(a.out/'conditional_fixed_effect_contrasts.csv').set_index('model')
    cells=contrast.loc['observed_cells'];linear=contrast.loc['linear']
    report=f'''# 个体内选择重建：结果更稳健，独立创新仍待建立

2026年11月9日。主样本保持300条真人来源记录，60条AI预测试另列。本轮完成两种固定效应条件模型和56项参与关联诊断；全部设定见`analysis_plan.json`。这是对已探索数据的追加审计。

## 解决了哪一层不确定性

既有主模型通过总体方案截距及正态随机截距处理稳定倾向。本轮允许每个人分别拥有稳定的A、B方案倾向，并用其8题中的A／B／不购买次数消去这些个体截距。费用和共享规则的共同系数由同一计数下的不同选择顺序识别。300条真人记录均包含至少两种选择，都对条件模型提供信息。

这是[Chamberlain固定效应条件似然的既有方法，参见Pforr（2014）第2节](https://doi.org/10.1177/1536867X1401400409)。它放宽个体截距分布，仍依赖共同属性斜率、给定个体截距后误差独立及固定菜单下的logit结构。时间变化的情境作用、学习、随机斜率及个体误差尺度差异没有因此消失。

## 89元共享规则包对比

| 模型 | G2−G1估计 | 95%记录层面区间 |
| --- | --- | --- |
| 既有普通自由单元模型 | 0.502717 | [0.028858, 0.976576] |
| 既有正态随机截距自由单元模型 | 0.499342 | [0.023815, 0.974870] |
| 新增个体固定效应、自由单元模型 | {cells['estimate']:.6f} | [{cells['ci_low']:.6f}, {cells['ci_high']:.6f}] |
| 新增个体固定效应、线性价格模型 | {linear['estimate']:.6f} | [{linear['ci_low']:.6f}, {linear['ci_high']:.6f}] |

个体截距的这一放宽没有改变核心自由单元对比的方向和量级。线性价格的区间更窄，其中包含额外的价格形状限制；它不能代替自由单元结果。此前条件A/B估计0.787，区间[-0.184,1.759]仍保留。

G2同时包含“决定是否共享”和“决定共享给谁”，G1为去标识后科研共享。问卷没有将两项89元方案直接对置；估计继续依赖跨题效用限制。固定效应不会补上缺少的属性组合，也不会把组合描述分解成纯控制权效应。相近的[数字健康共享DCE，Biasiotto等（2023）](https://www.jmir.org/2023/1/e47066/)已考察知情、同意和治理偏好。当前结果增强了这一应用估计的稳健性，尚未建立新的行为机制或独立方法。

## 参与关联的必要条件

稳定的单维参与倾向若对各题参与概率有同向单调作用，且给定该倾向后各题独立，则逐题参与指标应有非负总体关联。这个性质已有[Mokken相关理论](https://doi.org/10.1007/s11336-007-9034-z)，本案只使用其必要条件作诊断。

每版检验28个任务对，两版共56项。样本协方差有{checks['negative_sample_covariances']}项为负，最小单侧Fisher原始p为{checks['minimum_raw_one_sided_p']:.6f}；Holm校正后的最小p为{checks['minimum_Holm_p']:.6f}，0.05水平拒绝{checks['Holm_rejections_at_005']}项。不能据此确认潜在模型充分，也不能把不拒绝当成有足够功效。本轮没有完成该复合模型的等效性或功效认证。

## 数值与信息边界

枚举全部3⁸=6561条分类序列，按各选项次数分层，33个实际观测计数层进入似然。线性模型和自由单元模型的信息秩分别为3、7。用逐题概率乘积及两组任意个体截距独立核对条件概率，最大差异小于7.2×10⁻¹⁵。梯度和Hessian均经过有限差分校验，三组初值收敛至同一条件对数似然。

60条AI预测试全为固定选项向量，各自在条件模型中属于只含一条序列的计数层，贡献恒为零。将这些向量加入计算对照，两个模型的目标函数和梯度均保持不变，最大目标函数误差2.3×10⁻¹³。这不是通用AI清洗方法：非固定的AI答卷不享有该性质，真人固定答卷也可能存在。

条件分析舍去了各人的选项计数信息。它不恢复个体截距、总体不购买比例或新人的无条件选择概率。条件对数似然不能直接与原主分析的无条件留出分数比较；本轮没有把留出答案的计数用于无条件预测。

## 外部资料与GitHub学习

许可和文件完整性核对见`external_resource_preflight.json`。[RADIANT原始数据，Chen等（2025）](https://doi.org/10.5281/zenodo.15420092)的开发情境含299条记录、1790道选择题：298条各6题，1条2题。任务键为受访者与`choice_card`，`index`是行序号。8条完整记录始终选择同一选项；该事实不构成其无效的依据。外部数据尚未拟合模型，不能称为独立验证。

[Alves（2026）的当前7文件发布包](https://doi.org/10.5281/zenodo.23143985)引用了未随该版提供的模块和逐条预测文件，本轮只审阅代码与汇总结果，没有运行。其预测表现与估计目标之间的区分也应作为近邻研究核查对象，不能把相似的区分重新当作原创。

GitHub检查了statsmodels的`ConditionalMNLogit`条件化实现及其Chamberlain引用，具体提交和阅读范围见`github_reference_inspection.json`。没有安装或运行该库，也未复制其代码。GEMS仓库的相关数据是意见调查和测试类任务，尚不能作为带费用与属性菜单的DCE外部复现。

## 当前决定

保留300条真人主分析，新增个体内稳健性结果及完整诊断附件。固定效应、非负关联和恒定向量的零条件信息均不计为新理论。投稿目标继续保持未完成；下一项贡献主张须先给出最接近工作未回答的实质问题与可核验的识别依据，再进入模型开发。
'''
    (a.out/'个体内模型重建与创新边界.md').write_text(report)
    note=f'''# Within-person sensitivity of the sharing-package association

This exploratory audit conditions each person's eight choices on their counts of A, B and outside selections. Stable person-specific alternative intercepts cancel from the classical conditional multinomial logit likelihood. Shared attribute slopes and conditionally independent logit errors remain assumptions. The source population contains 300 human-recruitment records; all contribute within-count variation.

Using unrestricted observed fee-rule cells, G2 minus G1 at CNY89 is {cells['estimate']:.6f}, with a record-sandwich interval [{cells['ci_low']:.6f}, {cells['ci_high']:.6f}]. This is close to the earlier pooled-cell estimate of 0.502717. Linear fees give {linear['estimate']:.6f} [{linear['ci_low']:.6f}, {linear['ci_high']:.6f}]. Cross-task utility restrictions and the bundled G2 wording continue to limit behavioral interpretation.

The estimator is established; see [Pforr (2014), Section2](https://doi.org/10.1177/1536867X1401400409). We enumerate 6561 sequences, verify the multinomial count-stratum cardinalities, and independently compare normalized task-probability products at two arbitrary intercept vectors. Gradient and information checks are in `conditional_fixed_effect_fits.json`.

The 60 constant AI-pretest vectors contribute zero conditional likelihood, so adding them changes neither the objective nor the score. This narrowly defined invariance follows from singleton count strata. It neither labels constant human answers invalid nor extends to arbitrary synthetic responses. Conditioning also removes the information needed for unconditional participation shares; no unconditional predictive superiority is claimed.

Within version, 56 one-sided exact tests assess negative pairwise participation association, with Holm correction across all pairs. The smallest corrected p is 1.0. Nonrejection does not establish the full monotone latent model. The source theory is established [Mokken nonnegative association](https://doi.org/10.1007/s11336-007-9034-z).

These results strengthen the conditional application estimate. They do not establish an independently new method or the substantive innovation required for the current submission objective.
'''
    (a.out/'within_person_methods_note.md').write_text(note)
    (a.out/'README.md').write_text('''# Exploratory within-person assumption audit — 9 November 2026

Run from the repository root with numpy, pandas, scipy, openpyxl and threadpoolctl.
The frozen DH36 exports remain private. No new responses or AI generations are required.

```sh
python analysis/within_person_choice_audit.py --input-dir "$PRIVATE_DIR" --instrument survey/public_dh36/instrument.json --classification data/dh36_source_classification.json --out reports/within_person_audit_20261109
python analysis/write_within_person_audit_report.py --out reports/within_person_audit_20261109
```

The optional external preflight uses separately downloaded files from the two
CC-BY4.0 deposits listed in `external_resource_preflight.json`, their publisher
metadata and download receipt. It executes no downloaded research code:

```sh
python analysis/fetch_external_choice_resources.py --manifest reports/within_person_audit_20261109/external_resource_preflight.json --out "$EXTERNAL_DIR"
python analysis/external_choice_resource_preflight.py --input-dir "$EXTERNAL_DIR" --out reports/within_person_audit_20261109
```

`analysis_plan.json` fixes the two conditional models and all56 association tests
before these new fits; prior outcomes had already been examined. The method is
classical conditional likelihood, with shared slopes and stable alternative
intercepts. Conditional scores do not evaluate the unconditional population target.
Person-specific error-scale and slope heterogeneity are not eliminated by this
conditioning argument. All56 association tests are retained after Holm correction.
No human/AI origin detector, new theory or submission completion is claimed.
''')
    print('Within-person reports written from aggregate results.')


if __name__=='__main__':
    main()
