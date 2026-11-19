# 安全校正权重的留出预测检验

本次分析回到两套公开选择实验的 96 个真人标签组。每组有 8 个辅助设计，共 768 个配对。训练、校正、留出按受访者分开，沿用既有预测器及固定留出样本。权重和回退规则在检查留出结果前固定。预测损失按受访者平均。 [方案](plan.json)

## 结果

| 对照方法 | 对照范围 | 半安全上限损失更低 | 更高 | 相同 | 平均配对对数损失差 |
|---|---|---:|---:|---:|---:|
| 同标签普通 MNL | 两者均有原生有限解的 656 个配对 | 514 | 142 | 0 | −0.0529 |
| 同标签岭 MNL | 768 个配对，含事先设定的岭回退 | 223 | 433 | 112 | +0.0942 |
| 固定半权重校正 | 768 个配对，两边均按相同规则回退 | 318 | 418 | 32 | −0.0273 |
| 固定单位权重校正 | 768 个配对，两边均按相同规则回退 | 343 | 382 | 43 | −0.1141 |
| 合并训练及校正真人选择的普通 MNL | 相同总真人标签成本，768 个配对 | 0 | 768 | 0 | +0.2734 |

差值为半安全上限校正减对照。RADIANT 和 Swissmetro 的同标签普通 MNL 比较分别在 241/296 和 273/360 个配对中改善。令权重等于安全上限时，两套数据共 656 个有正上限的配对均得到原生有限解，配对平均差为 −0.0715。半上限及全上限是事先列明的比较，留出结果只用于评价。

有 52 个配对的单位权重目标无下界，改用半安全上限后可拟合。其留出损失在全部 52 个配对中高于同成本普通 MNL，平均高 0.4794。相较同标签岭回归，52 个中有 8 个改善、44 个变差，平均差为 +0.2628。固定大权重比安全权重的平均损失大，且安全权重只在部分配对中改善。固定权重是否回退依据存档的支撑判定。新增原生拟合没有数值失败。

从普通 MNL 起步，以已设留出损失计算的零权重导数在 533/656 个配对为负，和 1% 安全上限所给有限差商的符号全部一致。这个符号和半安全上限的实际损失改变量在 637/656 个配对一致。该计算用于解释固定留出样本上的误差走向。拟合样本和预定规则决定权重，评价样本只用于测量损失。 [导数推导](slope.md)

## 可核验范围

训练费计入合并真人 MNL 的 100 或 600 位预测器训练受访者。预算分别加上 5 或 10 位新揭示选择的受访者。已有预测器不计入增量真人标签时，比较使用校正标签的普通 MNL 与岭 MNL。每个估计目标和全部指标以原始菜单复算。3,552 条记录包括重复使用的对照、冻结预测和回退记录。独立复核的原始得分最大绝对分量小于 \(10^{-9}\)，预测指标最大差异小于 \(9\times10^{-16}\)。同一数据、分割和留出人群内的配对重复使用受访者，结果是固定样本分析。[完整结果](results/summary.json) · [最近邻文献](literature.md)

## 复现

压缩包的 upstream、reference 和 certificate 目录保留依赖的冻结来源与已生成的可行权重。另行取得原公开数据文件。依赖版本为 NumPy 2.5.3、SciPy 1.18.1、pandas 2.2.3、openpyxl 3.1.5、threadpoolctl 3.7.0。[RADIANT](https://zenodo.org/records/15420092) · [Swissmetro](https://github.com/michelbierlaire/biogeme/blob/master/src/biogeme/data/data/swissmetro.dat)

    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

    python fit.py --food-data /path/to/COSCBB.xlsx --food-archive upstream/food \
      --swiss-data /path/to/swissmetro.dat --swiss-archive upstream/swissmetro \
      --reference reference --certificate certificate --out regenerated

    python verify.py --food-data /path/to/COSCBB.xlsx --food-archive upstream/food \
      --swiss-data /path/to/swissmetro.dat --swiss-archive upstream/swissmetro \
      --reference reference --certificate certificate --results regenerated
