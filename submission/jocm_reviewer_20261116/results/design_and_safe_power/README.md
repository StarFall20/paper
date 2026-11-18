# 作答前的设计风险与作答后的安全权重

固定辅助题目和冻结预测器后，可以把指定方向上的衰退率拆成两项。第一项来自真人实际选择与该方向最优选择的差距。第二项是预测中心化后的辅助与真人设计之差，并与校正权重成正比。只要第二项为负，在每道题选择该方向下最优的备选项，就会让同一套作答在所有正权重下使目标无下界。取两个方向的正负号还能用属性宽度检验是否可能对所有作答都保证有限解。[推导](proof.md)

在 RADIANT 和 Swissmetro 的 768 个既有配对中，备选项常数方向均找到这种作答组合。对已经观察到的选择，常数方向在单位权重的 95 个外部目标中识别了 0 个。再加入单个属性方向，识别了 12 个。半权重下的 34 个外部目标均需完整支撑检验。这些结果说明作答前的存在性保证和实际作答后的失败诊断有不同用途。

真人选择收齐后，可从其题目、选择和冻结预测计算一个安全权重上限。任何具有全对比秩、且可用选项预测严格为正的辅助设计，都能在这个正上限内给出唯一有限解。上限仅用真人样本计算。对允许对比宽度任意缩小的辅助设计，界限是紧的。固定可行菜单可能允许更大的权重。

| 数据 | 标签组数 | 有正安全上限 | 安全上限至少为 0.5 | 正安全上限的中位数 | 用半上限验证的辅助配对 |
|---|---:|---:|---:|---:|---:|
| RADIANT | 48 | 37 | 6 | 0.2918 | 296 |
| Swissmetro | 48 | 45 | 13 | 0.3511 | 360 |
| 合计 | 96 | 82 | 19 | 0.3141 | 656 |

这里的 656 次均有原菜单概率支持证据，其中 52 次在单位权重、2 次在半权重时原目标无下界。两种线性程序独立核对同一安全上限。其余 14 个标签组的计算上限为零。这个界限衡量有限解，新的实验没有评价预测损失、区间覆盖或总体翻转率。样本与结果早已公开，分析没有新增问卷和预测训练。

[作答前检验方案](plan.json) · [结果](results/summary.json) · [安全权重方案](safe_power_plan.json) · [结果](safe_results/summary.json)

## 复现

压缩包的 upstream 和 reference 目录包含所需的归档源代码及派生文件。公开原始数据单独取得。依赖同此前固定标签分析，分别是 NumPy 2.5.3、SciPy 1.18.1、pandas 2.2.3、openpyxl 3.1.5 和 threadpoolctl 3.7.0。[RADIANT 数据](https://zenodo.org/records/15420092) [Swissmetro 数据](https://github.com/michelbierlaire/biogeme/blob/master/src/biogeme/data/data/swissmetro.dat)

    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

    python audit.py --food-data /path/to/COSCBB.xlsx --food-archive upstream/food \
      --swiss-data /path/to/swissmetro.dat --swiss-archive upstream/swissmetro \
      --reference reference --out regenerated/design

    python safe_power.py --food-data /path/to/COSCBB.xlsx --food-archive upstream/food \
      --swiss-data /path/to/swissmetro.dat --swiss-archive upstream/swissmetro \
      --reference reference --out regenerated/safe
