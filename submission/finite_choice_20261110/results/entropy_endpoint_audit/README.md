# 正权重上限附近，拟合目标与留出损失朝相反方向变化

对 107 个经过原始菜单核验的正权重上限，从同一组已验证系数重算 856 个点的目标值和路径导数。权重处于各自上限的 **90%** 时，107 个划分的补偿拟合目标均随权重下降，留出对数损失均随权重上升。半权重时有 88 个划分同时满足这两个方向。两项导数描述相同划分在给定权重处的局部变化。

| 权重占上限比例 | 拟合目标局部下降 | 留出损失局部上升 | 两者同时出现 |
| ---: | ---: | ---: | ---: |
| 50% | 98/107 | 88/107 | 88/107 |
| 90% | 107/107 | 107/107 | 107/107 |
| 99.9% | 107/107 | 107/107 | 107/107 |
| 99.99999% | 107/107 | 107/107 | 107/107 |

[推导](proof.md)给出拟合最优值的端点展开及一、二阶导数。最大熵表述来自标准凸共轭。传统熵惩罚研究沿固定优化问题改变惩罚强度。这里的补偿权重移动矩目标和备选项质量约束。系数发散速度是这一条件下的专门推论。稿件应将理论增量集中在比例维度下的存在阈值和相同局部信息对应不同存在极限的构造。固定菜单部分用导数与留出结果解释选择建模中的后果。 [Wainwright 与 Jordan](https://www.cs.columbia.edu/~blei/fogm/2025F/readings/WainwrightJordan2008.pdf) · [Cominetti 与 San Martín](https://www.ceremade.dauphine.fr/~carlier/Cominetti-SanMartin.pdf) · [Weed](https://proceedings.mlr.press/v75/weed18a.html)

856 个点的拟合目标与最大熵值最大绝对差为 \(2.77\times10^{-12}\)。最接近上限时，缩放留出损失导数与理论极限的绝对误差中位数为 0.00026，第 95 百分位为 0.5373。尾部差异说明单一最小效用差近似在部分划分中收敛很慢。分析使用同一食品 DCE 和固定留出组，结果的分母是分配次数。这一局部结论依赖规范化的拟合目标与留出样本，单凭目标下降无法判定调权提高了预测性能。

## 复现

在复现包根目录使用与 [端点审计](../dce_frontier_economic_path/README.md) 相同的依赖和公开工作簿。通过下载入口校验工作簿文件指纹。

    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python results/power_frontier/analysis/fetch_source.py --out local_inputs/COSCBB.xlsx
    python results/entropy_endpoint_audit/audit.py --workbook local_inputs/COSCBB.xlsx --frontier reports/power_frontier_20261118 --paths reports/dce_frontier_economic_path_20261118 --out regenerated/entropy_endpoint

[逐点记录](results/records.csv)和[汇总](results/summary.json)包含所有 856 个核对点。[重跑回执](results/replay.json)使用审稿复现包中的独立上游副本，两个数值文件逐字节一致。原始公开工作簿保留在数据发布平台。
