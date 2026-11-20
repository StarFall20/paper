# 同一受访者多道选择题怎样改变有限估计

在一个每人回答两道二元题的模型中，单题的属性分布、真人选择和冻结预测都相同。每题的预测固定为正确校准的 0.5，校正权重为 1。提高同一人两题之间的属性相关性，会缩小校正 Logit 得到有限解的参数范围。普通 Logit 的精确有限解概率，以及真值处的总体 Hessian 和按人计算的低维渐近协方差，在这两个设计中相同。[定理与推导](proof.md)

在辅助人数与真人标签人数相等时，任务独立的理论临界值为每名有标签受访者 0.6748 个斜率。同一人的两题相关性为 0.99 时，临界值为 0.4945。独立写法用矩形外包给出高相关一侧的上界 0.5508，用一维公式给出独立一侧的下界 0.6704。两侧的严格差距无需依赖二维积分精度。四档辅助人数比、八档相关性的 32 个数值点均用独立的高斯投影积分核验，最大差异约 0.0000057。[独立积分核验](boundary_verification.json)

## 模拟结果

以下是名义斜率人数比 0.585 的主比较。每行 200 个独立模拟，各档相关性用同一批真人选择和独立高斯驱动构造。所有比例按有标签受访者计，每人保留两道题。

| 有标签受访者 | 斜率数 | 两题独立 | 相关性 0.80 | 相关性 0.99 |
| --- | ---: | ---: | ---: | ---: |
| 48 | 28 | 160/200 | 92/200 | 34/200 |
| 96 | 56 | 170/200 | 102/200 | 26/200 |
| 192 | 112 | 189/200 | 85/200 | 7/200 |

在 192 人的配对中，182 个案例从有限解翻到无下界，反向翻转为零。主比较的 1,800 个普通 Logit 对照都有有限解。两档额外斜率人数比保留在原表。全部 2,280 个校正状态用原始概率线性规划独立复核，最大概率余量差为 $3.45\times10^{-11}$。每个外部目标另有主程序核对的负衰退值。[计划](plan.json) · [逐案结果](cohort_results.csv) · [独立核验](cohort_verification.json)

这个结果给主文增加了一个直接的选择实验问题。即使通常报告的局部信息一致，将题目当成独立观测仍可能给出错误的存在性判断。推导限定于同规模高斯题组、独立公平真人选择、恒定预测和单位校正权重。它刻意隔离属性相关性的作用，实际面板中的偏好异质性及其他依赖结构仍需具体检验。正文实证继续以实际菜单上的有限设计判别为依据。

两题模型来自本项目较早的三题依赖探索。本轮用更简单的条件高斯化推导，增加普通 Logit 在有限样本中的精确概率对照，并把校正临界值差的保守下界提高到 0.119。两项外部先例分别提供聚类稳健 PPI 推断与普通相关分类容量。 [11 月 18 日的三题结果](../cluster_boundary_certificate_20261118/README.md) · [Broska 和 Howes](https://arxiv.org/abs/2610.09601) · [López 等](https://arxiv.org/abs/cond-mat/9505093)

## 复现

使用 Python 3.12、NumPy 2.5.3 和 SciPy 1.18.1。生成完全合成，四条命令均可在本目录运行。

~~~sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python compute_boundary.py --out regenerated/boundaries.json
python verify_boundary.py --boundaries regenerated/boundaries.json --out regenerated/boundary_verification.json
python simulate.py --out regenerated
python verify_cases.py --records regenerated/cohort_results.csv --out regenerated/cohort_verification.json
~~~
