# 固定真人标签时替换辅助选择题

本次检验用两套公开选择实验，固定每组真人选择、冻结预测器、模型变量、权重和辅助样本量，反复替换辅助属性设计。RADIANT 每次有 64 名辅助受访者、384 道题。Swissmetro 每次有 64 名辅助受访者、576 道题。训练及留出人群沿用原划分。辅助样本的真人选择隐藏于校正过程。

| 数据 | 真人标签人数 | 固定标签组数 | 单位权重下存在性翻转 | 半权重下存在性翻转 |
|---|---:|---:|---:|---:|
| RADIANT | 5 | 24 | 4 | 1 |
| RADIANT | 10 | 24 | 2 | 0 |
| Swissmetro | 5 | 24 | 2 | 1 |
| Swissmetro | 10 | 24 | 4 | 0 |
| 合计 | | 96 | 12 | 2 |

翻转指同一组标签的 8 套辅助设计里，至少一套有唯一有限解，至少一套目标无下界。单位权重的 12 组里有 10 组普通 Logit 有有限解。半权重的 2 组也都有有限的普通 Logit。两种权重使用相同的辅助划分。1,536 次校正计算采用两种独立线性约束形式复核，对 129 次外部目标逐一用原始可选菜单核实负衰退方向。

该结果描述固定公开样本中的辅助设计敏感性。不同划分可复用同一批受访者。翻转比例取决于每个标签组的抽取次数和辅助预算。它没有检验真实数据的局部导数是否完全相等，也没有新增预测模型或问卷。本轮稿件中的高阶匹配是精确数学构造，真实数据单独回答仅替换辅助设计是否也会改变有限解。

[固定方案](plan.json) · [结果](results/summary.json) · [逐划分记录](results/allocations.csv)

## 复现

压缩包的 upstream 目录保留原公开分析读取程序、冻结系数及两个设计方案。原公开数据单独获得。环境为 NumPy 2.5.3、SciPy 1.18.1、pandas 2.2.3、openpyxl 3.1.5 和 threadpoolctl 3.7.0。[RADIANT 下载页](https://zenodo.org/records/15420092) [Swissmetro 下载页](https://github.com/michelbierlaire/biogeme/blob/master/src/biogeme/data/data/swissmetro.dat)

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run.py \
      --food-data /path/to/COSCBB.xlsx --food-archive upstream/food \
      --swiss-data /path/to/swissmetro.dat --swiss-archive upstream/swissmetro --out regenerated
