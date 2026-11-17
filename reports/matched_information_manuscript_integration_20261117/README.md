# 匹配参考信息构造已整合

当前正文第 3.5 节与附录 B5 已加入一个带估计截距的确定性设计。参考点处的逐行校正得分、完整 Hessian、经验 sandwich 矩阵和 Newton 更新完全一致，目标的全局存在性仍随固定权重改变：1/2 ≤ λ < 11/20 时有唯一有限解，边界处下确界有限却不取到，更大权重下原始损失无下界。参考点并非拟合根；这不是实际抽样方差相同或方差最优选权重的结论。

[精确分数计算](../../submission/finite_choice_20261110/results/matched_local_information/results/exact_information.json)和[独立 90 位原始损失核对](../../submission/finite_choice_20261110/results/matched_local_information/verification/verification.json)通过。[推导](../../submission/finite_choice_20261110/results/matched_local_information/derivation.md)说明对连续权重区间的判断；有限网格验证不替代该证明。宽泛的局部与全局区别及凸支撑几何属于已有知识。正文引用 [Li 与 Cai（2026）](https://arxiv.org/abs/2611.08959v1)的原始误差 Gram 与学生风险区别，避免将宽泛观点认作新贡献。具体读取范围仍以[清单](../../submission/finite_choice_20261110/results/matched_local_information/source_scope.json)为准。

[完整 Word](../../submission/finite_choice_20261110/manuscript/paper.docx) 69 页，[主文阅读版](../../submission/finite_choice_20261110/manuscript/main_text.docx) 29 页。摘要 174 词，主文 8737 词。原有 18 张表、141 个显示公式主体、83 个根分析源文件及 776 个既有科学文件全部保持哈希；新增 4 个显示公式。当前 Word 分别有 1319 与 236 个原生公式对象。全部最终页面有直接原分辨率检查或精确图像继承记录。

[版本指针](../../submission/finite_choice_20261110/reports/current_manuscript.json)、[排版记录](../../submission/finite_choice_20261110/reports/matched_information_visual_qa_20261117.json)及[验证范围](../../submission/finite_choice_20261110/reports/matched_information_validation_20261117.json)绑定交付文件。此前 60000 条单步方法记录与独立完整重跑证据保留，本次不重跑无变化的模拟。旧独立补充包及其交付记录为整合前快照，不作为当前 Word 指针。

这是同一研究主线内更严格的受控解释，没有新增框架、独立模拟队列或真人／AI 答卷。正式 PPI++ 正文及 A–E 附录尚未读完，具体结果优先性及足够的选择分析价值仍待确认。本轮不能据此认定达到投稿门槛，研究目标保持进行中。
