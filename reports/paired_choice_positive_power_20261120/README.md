# 两道重复选择题在非单位校正权重下的有限解

此前稿件在校正权重为 1 时证明，同一受访者题目的属性相关性会移动校正 Logit 的有限解临界值。本次把判定推广到固定的 $0<\lambda<1$，此时真人菜单和辅助菜单都进入对数配分函数。仍可让单题分布、普通 Logit 的精确存在概率以及常用的固定维度聚类协方差在两种设计中完全相同。 [证明](proof.md) · [上一版单位权重结果](../paired_choice_boundary_20261120/README.md)

在每人两题、两组人数相等、校正权重为 0.90 的模型里，每位标注受访者的临界斜率数从题目独立时的 **0.75355** 降至题目相关系数 0.99 时的 **0.57194**。不用二维多边形积分也能确认差异。独立设计的解析下界高于 0.750，相关设计的解析上界低于 0.635。预先指定的设计规则在二者之间选取了每位受访者约 0.663 个斜率。 [计算方案](plan.json) · [15 个临界值](boundaries.json) · [解析界的数值验算](strict_bounds.json)

| 两组各有受访者 | 斜率数 | 题目独立时有限 | 相关系数 0.99 时有限 | 普通 Logit 两种设计合计 |
| --- | --- | --- | --- | --- |
| 96 | 64 | 78/100 | 8/100 | 200/200 |
| 192 | 127 | 96/100 | 9/100 | 200/200 |

两种设计使用相同的人类标签和基础高斯随机量。192 人时有 87 对从有限解变为目标无下界，反向转换为零。保留了 400 条配置的原始状态和原始对偶校验。32 条预选配置改用原概率坐标验算，半径差小于 $1.5\times10^{-13}$。15 个临界值还用另一种二维投影计算在所报尺度上复核。 [成对状态](results/cohort_results.csv) · [汇总](results/summary.json) · [投影复核](projection_verification.json)

两题模型保持标签独立，变化的是同一人两道题的属性相关性。理论采用固定权重和校准常数预测。这保证方程与对照的可解释性。真实公开选择数据的预测和有限解结果已在稿件第 4 节另行报告。本结果为主张提供第二个严格的权重条件。聚类稳健 PPI 的既有结论关注推断和权重选择，本证明关注给定权重时原目标能否取得最小值。 [Broska 和 Howes](https://arxiv.org/abs/2610.09601v1)

从仓库根目录，在现有投稿复现包的固定依赖环境中运行下列脚本。模拟的预设是 100 对独立随机重复，分别在 96 人和 192 人两个样本量执行。计算用约三分钟。

```sh
d=reports/paired_choice_positive_power_20261120
python "$d/boundary.py" --out "$d/boundaries.json"
python "$d/verify_boundaries.py" --out "$d/strict_bounds.json"
python "$d/verify_projection.py" --boundaries "$d/boundaries.json" --out "$d/projection_verification.json"
python "$d/simulate_positive.py" --out local_positive_power_results
```
