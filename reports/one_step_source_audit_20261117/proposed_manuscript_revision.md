# 待并入正文的基线范围说明

当前 Word 尚未改动。以下内容用于第 5.1 节方法说明及相应附录，完整 24 条件的 CSV 应随当前补充复现材料提供。

The one-step comparator uses the same ridge start and power clipped to [0,1]. Section 6.2 of PPI++ also permits unrestricted power in its one-step formula. We repeat all 24 original conditions with both power domains and with pooled or labelled information for tuning. Every method retains all 500 repetitions. The original clipped comparator is reproduced, and the unrestricted versions have equal or higher observed mean human probability loss within each information convention. These comparisons concern a finite computed update. They provide no finite-global-minimizer or high-dimensional interval guarantee.

正文及附录应同时说明：共同起点为原来的 human ridge；标签信息版本对应预印本式 (8)，pooled 信息版本延续既有基线的约定；使用修正损失实际 Hessian 作一步更新。本轮没有执行上游 API，不能将重新实现公式称为逐行重现软件调用。

新结论应保留观测均值和 Monte Carlo 误差的限定。该比较不证明不裁剪权重普遍有害，不改变论文的凸域全局存在性问题，也不应计作新的理论创新点。正式发表版及 A–E 的对照仍待完成。
