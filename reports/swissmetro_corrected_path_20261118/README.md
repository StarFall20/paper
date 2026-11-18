# Swissmetro 独立重复检验

补偿 Logit 的系数仍然有限时，提高补偿权重也可能使留出预测恶化。Swissmetro 的 93 个严格介于零和一之间的权重上限全部通过边界面核验。权重从各自上限的 50% 移到 90%，其中 **90/93** 个划分的留出对数损失上升。有 59 个正上限划分的普通真人 Logit 已有有限估计，说明补偿过程本身也会带来权重上限。

| 模型 | 真人标签预算 | 正上限数 | 50% 与 90% 留出损失中位数 | 配对损失上升数 | 单位权重仍可估计数 | 对照的配对损失差中位数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 标准四参数 | 2 | 25 | 1.967、3.002 | 24/25 | 50 | −0.016 |
| 标准四参数 | 5 | 13 | 1.657、3.315 | 13/13 | 82 | 0.002 |
| 标准四参数 | 10 | 3 | 0.994、2.439 | 3/3 | 93 | 0.002 |
| 八参数 | 2 | 20 | 2.160、3.704 | 18/20 | 23 | −0.082 |
| 八参数 | 5 | 24 | 1.970、4.635 | 24/24 | 64 | 0.016 |
| 八参数 | 10 | 8 | 1.389、2.922 | 8/8 | 85 | 0.010 |

留出损失越低越好。每个模型和预算组合含 96 个划分，所有 576 条划分记录均留存。其中 86 个上限为零或接近零，397 个在单位权重仍有有限估计，93 个上限为正且低于一。表格的比较限定在各自的划分组。八参数模型包含分交通方式的时间系数和两种轨道班次间隔，标准模型用通用时间和成本系数。两者使用相同的人与题目。

## 数据与同标签的基准

按 Biogeme 的备选项可用性、票价和缺失选择编码读取 1,191 位受访者的 10,719 道题。600 人用于训练固定预测器，351 人进入标签分配和代理样本池，240 人的 2,160 道题只用于评价。预测器是标准四参数 MNL。固定的预分析计划和受访者划分适用于两个模型。 [Swissmetro 原始研究](https://transp-or.epfl.ch/documents/proceedings/BierAxhaAbay01.pdf) · [Biogeme 标准模型](https://biogeme.epfl.ch/sphinx/auto_examples/assisted/plot_b00logit.html) · [公开数据文件](https://github.com/michelbierlaire/biogeme/blob/master/src/biogeme/data/data/swissmetro.dat)

同标签比较把预测器训练用的 600 人和每个划分揭示的 2、5 或 10 人全部交给直接真人 MNL。两个计划权重点共有 1,152 次配对比较，其中 980 次使用通过核验的补偿拟合，172 次依照计划使用同划分的标签组 ridge。直接拟合在全部比较中留出损失更低。这个比较适用于预测器训练标签可直接用于目标拟合的情形。真正外部预训练模型的标签成本及可迁移性需要另行判断。 [成本比较记录](results/label_cost/summary.json)

## 数值核对与复现

主程序在 490 个有内部路径的划分上计算了 1,960 个拟合点。独立程序从原始题目重构可用备选项，核对全部 1,960 次原三项得分、576 次标签组 ridge、全部 93 个正上限的支持方向与边界面概率，并用不同变量布局重做 36 个上限规划。最大原得分差低于 \(3.0\times10^{-13}\)。数值实现使用同一组备选项和同一个线性规划，首轮试算比较了两种求解方式。正式运行采用较快的内点方法结合单纯形回退，独立校验使用另一种矩阵布局。划分、模型、预算和评价固定。 [分析计划](plan.json) · [独立核对](results/verification.json)

在仓库根目录运行以下命令。数据来源由计划中的 SHA-256 校验，原始数据从上游取得。软件版本沿用 [端点审计依赖](../dce_frontier_economic_path_20261118/requirements.txt)。

    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    mkdir -p local_inputs regenerated/swissmetro
    curl --fail --location https://raw.githubusercontent.com/michelbierlaire/biogeme/master/src/biogeme/data/data/swissmetro.dat --output local_inputs/swissmetro.dat
    python reports/swissmetro_corrected_path_20261118/audit.py --input local_inputs/swissmetro.dat --numerics reports/dce_frontier_economic_path_20261118/analysis/audit_path.py --out regenerated/swissmetro/paths
    python reports/swissmetro_corrected_path_20261118/verify.py --input local_inputs/swissmetro.dat --results regenerated/swissmetro/paths --out regenerated/swissmetro/verification.json
    python reports/swissmetro_corrected_path_20261118/compare_label_cost.py --input local_inputs/swissmetro.dat --numerics reports/dce_frontier_economic_path_20261118/analysis/audit_path.py --results regenerated/swissmetro/paths --out regenerated/swissmetro/label_cost

统计量描述这两种指定模型、同一个公开交通数据集和确定的标签分配。标准模型的五人、十人预算分别有 13 个和 3 个正上限，读者可结合分母判断稳定性。
