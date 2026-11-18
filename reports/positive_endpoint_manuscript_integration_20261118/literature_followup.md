# PPI++、有限估计和正权重上限的核对

2026 年 11 月 18 日，Chrome 列出了 Scopus 首页。打开该标签及备用窗口连接均超时，完整数据库权限与实际检索结果未获核实。公开入口也未返回本轮 Google Scholar 查询的结果。Web of Science 保留先前的查询记录，本轮的比对来自公开原文、作者代码和用户提供的 PDF。下列核对说明已核实的重合范围。

| 来源及阅读范围 | 已有结果 | 当前稿件的归属和增量 |
| --- | --- | --- |
| PPI++ v2 的正文、附录 A.1–A.2，作者 Python 实现的二元 Logit 目标及权重更新 | 预测校正目标、在固定维度正则条件下的一致性与推断、方差调权、Logit 和多类 Logit 例子。加权目标经数值优化求解。 | 第 3.2 节承认固定维度凸性和一致性证明的有限估计含义。候选贡献是标签预算与维度同步增长时的特定存在阈值，以及局部得分与常规协方差匹配时，存在极限仍可不同。 [PPI++ v2](https://arxiv.org/html/2311.01453v2) · [正式论文](https://doi.org/10.1214/26-AOAS2215) · [作者代码](https://github.com/aangelopoulos/ppi_py) |
| Correia、Guimarães 与 Zylkin，表 1、命题 1 和附录 A.3 | 从单观测似然和结局支持推导 GLM 的分离判据，并处理边界与上界条件。其研究已经建立可计算的存在性诊断。 | 有限可估性和可计算判据已有先例。此处比较的是预测校正移动充分统计量时产生的正权重上限，以及比例维度下的存在阈值。列为待纳入正式参考文献的邻近研究。 [作者全文](https://arxiv.org/html/1903.01633v7) |
| Geyer 与 Rinaldo、Fienberg、Zhou 的指数族边界理论 | 方向衰退、边界面及条件极限模型均有理论基础。 | 附录 C13 标明归属。当前稿计算固定菜单和冻结预测下，从内部沿权重路径走向单一边界面时的系数速度、留出损失斜率和价格比率，再用原题逐项核验。 [Geyer](https://www.stat.umn.edu/geyer/gdor/) · [Rinaldo 等](https://arxiv.org/abs/0901.0026) |
| Cominetti 与 San Martín 的指数惩罚轨迹，以及 Weed 的显式界 | 固定线性规划的熵惩罚轨迹可指数收敛，且已有显式常数界。 | 这是最小效用差控制指数衰减的邻近技术。当前附录中变化的是权重诱导的矩约束。仍需检验完整速率能否由一个等价惩罚或熵投影问题直接推出，再决定速率在投稿信中的权重。 [Cominetti 与 San Martín](https://www.ceremade.dauphine.fr/~carlier/Cominetti-SanMartin.pdf) · [Weed](https://proceedings.mlr.press/v75/weed18a.html) |
| JCM 近期价格设定论文的出版社页面 | 以留出表现、需求和支付意愿比较不同价格设定，指出预测与定价反事实可以表现不同。 | 同刊读者会关心“仍可拟合”时概率预测恶化，以及价格指标是否也不稳定。当前 DCE 结果支持前一项。价格方向稳定，大多数划分的指定溢价选择判断也稳定。 [同刊近作](https://www.sciencedirect.com/science/article/pii/S1755534526000412) |

用户提供的 PPI++ PDF 共 29 页，含正文和附录。SHA-256 为 5c9dc26aa1a537785d34b1d690a7387cf21e0493f0a3f945905dc556d30e70bd。包内保留的作者代码核对与此前真实 DCE 工作流比较可分别在 [代码审计](../../submission/finite_choice_20261110/results/official_ppi_runtime/README.md) 和 [独立工作流证据](../../submission/finite_choice_20261110/results/independent_released_workflow/README.md) 查看。

本轮材料支持相关方法的明确归属与可复现的具体增量。通过已读论文对照仍无法判断全球优先性，也未进行逐字相似度检测。JCM 的范围与主题匹配。投稿判断仍取决于比例维度下的理论增量能否经受邻近理论审查，以及固定留出样本中的机制在独立任务上是否可复现。 [期刊范围](https://shop.elsevier.com/journals/journal-of-choice-modelling/1755-5345)

补充复审核对了 Ye 与 Yoganarasimhan 的第 5.3 节。该文把 PPI++ 的方差分配扩展到联合分析的多项 Logit 效用，正文已引用这项用途。当前候选贡献聚焦补偿估计在比例维度下的有限解。标准最大熵共轭与单一边界面计算的归属详见新增的 [权重路径复审](../entropy_endpoint_audit_20261118/proof.md)。 [Ye 与 Yoganarasimhan](https://arxiv.org/html/2604.17267v1)
