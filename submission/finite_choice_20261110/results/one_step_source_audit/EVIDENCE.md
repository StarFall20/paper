This record describes the standalone audit before manuscript integration. The current parent release includes the comparison in Appendix C8.

# 一步法基线与官方来源核查

已读取用户提供的 PDF 和[官方仓库](https://github.com/aangelopoulos/ppi_py)，完成原有全部 24 个模拟条件上的权重范围对照。每条件保留 500 次重复，复用三组实验原先的 1500 组模拟样本。五个比较方法共有 60000 条记录；没有新增参与者、AI 答卷或独立模拟样本。当前已审阅 Word 尚未改动。

## 原文及代码的实际范围

上传的 PDF 共 29 页，首页明确标注 2024 年 3 月 26 日 v2，附录为 A–C。第 8 页 Proposition 1 和第 23 页 A.2 的凸壳证明，支持当前稿件已经计入先行内容的固定维度一致性与有限解推论。第 17–18 页的维度比较展示 d=1–5，关注区间覆盖、宽度和网格终止。此次文件没有提供 2026 正式正文及附录 A–E。[读取范围](source_scope.json)列出实际阅读及图像核对的页面。

第 11 页式 (8) 使用标签信息矩阵，随后的一步更新允许实数权重，并使用该权重下修正损失的 Hessian。原文所需的常规一致起点条件仍需保留。当前实验以相同的 human ridge 起点作操作性比较，没有据此声称比例增长维度下 Corollary 5 仍有效。

官方默认分支固定为 `3d1f0c6`，论文分支固定为 `363265dd`。默认分支的 `ppi_logistic_pointestimate` 没有 `one_step` 参数；论文分支保留该参数。论文分支中，[点估计路径](https://github.com/aangelopoulos/ppi_py/blob/363265dd97919b47e185b9a0b069dcf5463e79d4/ppi_py/ppi.py#L874-L917)在自动调权时仍裁剪权重，其一步更新使用 pooled information。[区间路径](https://github.com/aangelopoulos/ppi_py/blob/363265dd97919b47e185b9a0b069dcf5463e79d4/ppi_py/ppi.py#L1068-L1103)允许不裁剪权重，再调用固定权重的点估计路径；按该文件的分支结构，后者返回优化结果。这里记录的是静态控制流，未运行上游代码，也没有认定这个分支等同于正式 Supplement B。

## 补做的比较

旧基线使用 pooled information 和 `[0,1]` 裁剪权重。补算比较 pooled/labelled information 与裁剪/不裁剪的四个组合，加入相同样本上的直接 human ridge。全部方法使用原来的 ridge 起点；一步更新使用修正损失的实际 Hessian。方案在补算前固定，见[计划](analysis/one_step_power_comparison_plan_20261117.json)。没有调整旧结果、加阻尼、加抖动或去除极端输出。

[独立核验](verification/verification.json)逐项检查全部 60000 条结果、3500 个唯一 ridge 起点，以及不依赖矩阵迹实现的逐行权重公式。梯度与 Hessian 的有限差分检查包括负权重和大于一的权重。所有更新的线性方程检验通过；这说明更新计算得到复核，不意味着它们是原目标的有限全局最小点。

[原基线对照](verification/all_24_legacy_cells.csv)覆盖全部 24 个旧汇总条件。估计数量完全相同，MSE 与预测损失的最大相对差分别约为 1.30e-16 和 7.36e-17。

## 结果及含义

所有 60000 条方法记录均有有限输出。在两个不裁剪版本的 24000 条记录中，996 条权重超出 `[0,1]`，对应 643 个条件与重复组合、340 组原始样本；116 条记录的一步更新矩阵具有负特征值。这些计数包含耦合方法与条件，不能当作独立样本数。

在 500 次重复的观测均值中，pooled 不裁剪版本在 13/24 个条件上的预期新行损失较高，另 11 个条件与裁剪版本相同；labelled 版本为 12 个较高、12 个相同。两个版本均未出现更低的条件均值。这是所检验网格的结果，未证明不裁剪权重普遍有害，也未否定原文的常规渐近结论。

例如原有 20 标签、150 辅助行、代理错误率 0.25 的条件，pooled 裁剪与不裁剪的平均超额 log loss 分别为 0.483805 和 1.454968。配对增加为 0.971163，Monte Carlo 标准误为 0.311390。全部 24 条件的结果和尾部指标均保留，没有用这一条件代替全网格。

直接 ridge 也并非处处最优。pooled 一步版本在 7/24 个条件上有较低的观测平均损失、17 个较高；labelled 一步版本为 5 个较低、19 个较高。应分别报告方法、起点、权重域、误差与适用范围。

当前结论是补齐基线范围的说明与敏感性比较。旧的一步结果需要明确标为裁剪权重版本。论文的有限全局解边界仍针对既定凸权重域；一步计算的可用性不能代替该目标的存在性。该核查不增加新的理论创新点，正式 PPI++ A–E 的结果级对照仍未完成。

## 数据与运行入口

- [全部 60000 条方法记录](results/trial_records.csv.gz)与[3500 个起点](results/pilot_records.csv.gz)。
- [完整方法汇总](summary/method_summary.csv)、[全部 240 个条件与方法对比](summary/all_pairwise.csv)及[48 条裁剪对照](summary/clipping_effects.csv)。
- [所有条件的平均损失表](summary/all_conditions_mean_loss.md)与[计数及范围](summary/overview.json)。

源代码为[补算](analysis/one_step_power_comparison.py)、[独立核验](analysis/verify_one_step_power_comparison.py)及[汇总](analysis/summarize_one_step_power_comparison.py)。输入沿用已固定版本的公开 vaccine 属性文件，生成标签仅重建旧模拟；问卷的 300 条真人答卷及 60 条 AI 预测试没有进入本研究。
