# EPI 最近邻原文与既有设计核查

此前仅保留摘要的 [Wang、Wen 与 Zou（2025）](https://arxiv.org/html/2512.16363v1)已补读其目标、有限辅助样本及交叉拟合约束、常规理论和计算单元。[读取范围](source_scope.json)逐项记录实际单元；未声称逐行审读全文。其经验似然方法属于已有预测辅助推断，需在稿件的方法边界中明确对照。

将其监督得分约束专门化为普通 Logit 后，人工完全或准完全分离会排除正经验似然的可行点。[推导](derivation.md)给出对所有有限系数及正权重成立的符号证明。这是经典正权重分离的推论，未作为新理论贡献。

沿用当前附录 B5 的同一标量设计，[分数运算](results/exact_existing_design.json)确认补偿目标在原权重下有有限解，单位权重路径终点无界；EPI 人工得分在两个情形都保持不可行。[独立核对](verification/verification.json)采用 80 位精度，原始损失导数残差小于 1e-65。另保留 175 个权重与系数组合的数值检查，普遍结论由符号证明承担。未运行 EPI 作者代码，未新增独立模拟、真人或 AI 答卷。

复核命令从本目录执行：

```sh
python check_existing_design.py --out regenerated/exact
python verify_existing_design.py --certificate regenerated/exact/exact_existing_design.json --out regenerated/verification
```

这项对照缩小了一项最近邻不确定性，并修正先行研究范围。它没有建立新的预测优势，未证明全球优先权，也不能单独确认投稿门槛。正式 PPI++ 正文与 A–E 附录的公开入口再次访问仍未提供可读内容；作者主页及 arXiv 仍指向已读 v2。新检索是七条公开网页查询，不计作 Scopus、Google Scholar 或 WOS 数据库检索。
