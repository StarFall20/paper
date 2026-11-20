# 固定选择菜单上的有限估计和留出预测

复查最新稿件附录 I 的全软标签对照。本轮使用同一公开数据、500 个已记录的划分、50 和 150 条标注选择、两种预测输入和两种五折方式。将完整校正、固定倍数 0.25 和 0.50、以及保留一半可行距离的校正与合并菜单全软标签、真人加软标签及同标签 ridge 比较。保留每个划分。完整校正不可行时用全软标签产生预测。3,405 条留出选择用于计算损失。 [分析计划](plan.json) · [上游公开选择数据](https://github.com/ChengLu1999/generative-augmented-inference/tree/d18717206867c7a12ff470879a0dd5d488b300c2)

原版五折在行上分组。本次另按上游划分使用的五行块切分，把整块放在训练折或验证折内。块内所有行一起进入验证折，回退概率由训练折计算。公开记录缺少受访者编号。“五行块”是资料中的划分单位，受访者层面的独立性需要相应标识。已核对早期 2,000 组预测配置的逐行分折均没有标注行回退。 [核验明细](results)

| 50 条标注选择，分块交叉拟合 | AI 预测 | 仅属性预测 |
| --- | ---: | ---: |
| 完整校正，含全部回退 | 0.88863 | 0.89362 |
| 保留一半可行距离 | 0.76318 | 0.76683 |
| 固定校正倍数 0.50 | 0.71130 | 0.71332 |
| 固定校正倍数 0.25 | 0.69020 | 0.69257 |
| 真人加软标签 | 0.68644 | 0.68792 |
| 合并菜单全软标签 | 0.68694 | 0.68812 |
| 同标签 ridge | 0.76235 | 0.76235 |

指标是同一批真实留出选择上的平均对数损失。按行分折的完整校正有 7 个无界配置，按五行块分折有 6 个。两个固定小倍数和半距离规则在两种分折下均有有限解。半距离规则降低了完整校正的平均损失，在表中仍高于全软标签和真人加软标签。150 条标注选择的四组比较中，完整校正和全软标签都全部有限，完整校正平均损失仍更高。重复使用同一个留出集和重叠的标注样本，因此这些是描述性比较。 [按行分折汇总](results/row/summary.csv) · [按块分折汇总](results/block/summary.csv)

计算了 4,000 个残差上限，并用可行概率与支持方向逐一互证。双侧计算最大误差低于 1.7×10⁻¹⁴。24,000 个方法配置保留完整状态，23,987 个有限拟合直接核验概率得分，960 个用另一种求根法复核系数和留出损失，另核对 6,000 个已归档配置。全部 48 组汇总由独立代码从逐划分记录重算。 [运行代码](evaluate_paths.py) · [汇总校验](verify_report.py)

## 与最接近的既有比较

Wang、Zhang 和 Zhang 在 §5.3 已将 AAE、PPI 和 PPI++ 用于新冠疫苗 conjoint。§6.2 又在医疗保险数据及模拟中对比这些估计量，并报告系数误差。本文 §4.4 的增量是固定二阶段的菜单权重与预测，隔离线性残差项如何把目标推出可行支持，并对同一批真实留出选择计算全部方案的预测损失。两种方法的应用对照已有先例。 [论文 §5.3](https://arxiv.org/html/2412.19363v3#S5.SS3) · [论文 §6.2](https://arxiv.org/html/2412.19363v3#S6.SS2)

Li 和 Ignatiadis 用多个任务的预测来收缩均值估计。半距离规则只约束一次选择拟合中的残差矩，作用是保持目标在可行集合内部。在总体标签和预测的特征矩有偏差时，长期固定的小倍数会改变估计目标。这里将有限估计与预测误差作为两个独立的实证指标。 [Prediction-Powered Adaptive Shrinkage Estimation](https://proceedings.mlr.press/v267/li25ak.html) · [投稿附录源文](supplement_note.md)

## 复算

在仓库根目录，沿用[固定依赖和公开数据下载步骤](../aae_comparator_audit_20261120/README.md)。以下命令使用已经校验的公开输入目录。

```sh
python reports/public_residual_margin_audit_20261120/evaluate_paths.py --data local_inputs --scheme row --out regenerated/margin/row
python reports/public_residual_margin_audit_20261120/evaluate_paths.py --data local_inputs --scheme block --out regenerated/margin/block
python reports/public_residual_margin_audit_20261120/verify_report.py regenerated/margin
```
