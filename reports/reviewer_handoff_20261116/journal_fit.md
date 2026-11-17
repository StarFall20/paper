# 当前稿件与 Journal of Choice Modelling 的关系

判断截至 2026 年 11 月 16 日。主题与读者有明确联系；创新价值仍需落在补偿目标的具体存在性边界及其后果。整理文件、增加模拟次数和增加篇幅都不提高理论优先权。

## 对照实际发表的工作

| 论文及读取范围 | 对当前稿件的直接要求 |
|---|---|
| [van Cranenburgh 等 2022](https://doi.org/10.1016/j.jocm.2021.100340)：前期读取及本次出版社摘要复核 | 机器学习与选择建模的结合已经是该刊议题。当前稿件须明确预测如何进入既有 logit 目标、影响哪项建模决策；“引入 AI”本身没有贡献资格。 |
| [Sfeir 等 2026](https://doi.org/10.1016/j.jocm.2026.100623)：前期阅读、已留存机构索引摘要 | 其主题是语言模型辅助模型设定和估计。当前结果涉及预测响应修正的有限解；应直接交代这个对象，避免把所有机器辅助建模混为同一方法。 |
| [Hess 等 2026 正式版](https://eprints.whiterose.ac.uk/id/eprint/244754/1/1-s2.0-S1755534526000175-main.pdf)：本次定位并读取第 2.3 节及第 6 节相关段落，PDF 共 19 页 | 该文区分不确定性计算、统计结论及行为意义。当前稿件分别报告有限解、条件误差、区间发放及覆盖；这些维度不能互相代替。存在性改善需要有具体选择分析后果。 |
| [Campbell 等 2026 正式版](https://eprints.whiterose.ac.uk/id/eprint/245939/1/1-s2.0-S1755534526000308-main.pdf)：本次读取引言、第 2.2 节、结论及附录 A 接口说明，PDF 共 14 页 | 该文把一个明确的下游推断问题连接到既有方法与可执行代码。当前稿件也应给读者清楚的适用对象、决策含义和可运行入口。该文没有被用来增添当前模型的理论组件。 |
| [Fok–Paap 2025](https://doi.org/10.1016/j.jocm.2024.100531)：本次出版社摘要；正式 PDF 访问返回 403，未声称读全文 | 该文比较 MNL 误设检验。当前论文研究一个指定修正目标的解是否有限。曾经的纤维充分性检验方向已撤回，不能再作为当前创新贡献累加。 |
| [Kessels–Jones–Goos 2019](https://doi.org/10.1016/j.jocm.2018.12.002)：前期出版社摘要及成熟基线核查 | Firth 解决选择数据估计困难已有该刊先例。当前稿件保留 Firth、ridge 和其他成熟比较；独立贡献须由边界计算和受控反例承担。 |

以上是主题与呈现方式的有限对照。两篇新取得的正式版只读取表中指定单元；没有声称整篇逐行审读，也没有把相似写作结构当作创新证据。新检索是公开网页检索，未新增 Scopus、WOS 或 Google Scholar 数据库检索次数。

## 当前贡献判断

研究集中在同一个未惩罚的预测修正 logit 目标。可辩护的贡献候选有三个相连部分：有限辅助样本下增长维度的显式边界；匹配得分或常规协方差仍出现相反存在性结果的具体构造；权重、辅助预算和预测损失之间的后果。独立对称离散噪声扩展只改变原模型的一项分布条件，服从该主线。

支撑几何、线性规划、统计维度、一般普适性、方差权重和秩论证都有既有来源。原稿不应把这些工具列为独立原创方法。现有误差比较也限制推广：更多有限解没有转化成一般预测优势。其定位是存在性与方法诊断论文。

潜在审稿风险仍集中在实际重要性：该边界的参数区间是否足以影响常见选择分析；已发表的惩罚估计方法是否已满足读者的实际需求；独立噪声扩展与实际相关属性设计之间还有距离。当前稿件有明确范围和负面比较，无法据此消除这些价值判断。

## 尚未完成的最近邻核对

PPI++ 的正式发表信息已确认。[作者主页](https://stanford.edu/~jduchi/publications.html)列出 Annals of Applied Statistics 20(3), 2235–2252；其 PDF 链接仍指向 arXiv。[正式全文入口](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-20/issue-3/PPI-Efficient-prediction-powered-inference/10.1214/26-AOAS2215.full)的后续访问已由用户完成验证。正文仍受全文权限限制，附录 A 下载被明确拒绝；学校认证入口返回应用未注册。正式附录简介列出 A–E，并提到维度增长实验。读取旧代码分支没有解决正式附录的结果级核对。既有预印本 v2 的固定维度一致性证明已计入先行内容。

后续补读的 [Lee–Kim（2026）](https://arxiv.org/abs/2606.08730v1)也需正面对照。其全体预测加标签残差公式对应本稿权重 N/(m+N) 的目标；抽样记号不同不能承担创新区分。当前第 3.2 节及参考文献已补入该项先行研究，详见[修订记录](../regular_prior_revision_20261116/README.md)。

在正式版及附录核对完成前，不应作出“已排除最近邻重合”或“已经达到投稿创新门槛”的结论。当前状态是具备具体贡献候选与完整内部数值证据，尚有一项关键全文来源缺口。最终作者审阅与期刊门户声明另列于投稿包。


## 17 November 2026 EPI source-scope follow-up

[Wang, Wen and Zou (2025), v1](https://arxiv.org/html/2512.16363v1) has now been read at its objective/constraints, regular assumptions and results, and specified computational and proof units. The [reading inventory](../epi_scope_20261117/source_scope.json) records that scope. Its retained supervised-score equality has the classical positive-weight logistic-separation obstruction. The same existing scalar example distinguishes the constraints from the compensated-loss target; this supplies method-scope attribution, not another original component or a performance ranking. Section 3.2 and Appendix B6 now include the comparison. Formal PPI++ A–E reconciliation and the choice-analysis value judgment remain open.


## 17 November 2026 observed-curvature strengthening

The [new checked finite design](../matched_local_information_20261117/README.md) holds the complete reference Hessian and empirical sandwich fixed, in addition to every corrected score row. Its exact 11/20 transition answers a narrower curvature objection to the existing score path. The reference point is not a fitted root, the compared powers are not claimed variance-optimal, and the example gives no practical prevalence or prediction-advantage result. Recent weak-supervision risk theory and earlier non-squared-loss results are explicitly scoped in the companion assessment. This strengthens explanation within the same compensated-choice objective; it does not resolve the formal PPI++ gap or sufficient journal value. The current reviewed Word is unchanged pending integration.


## 17 November 2026 matched-reference-information integration

The construction described in the preceding standalone progress record is now integrated in Section 3.5 and Appendix B5. It is one controlled finite-design illustration within the existing compensated objective. All score rows, observed Hessian, empirical sandwich and Newton direction at a nonstationary reference agree while attainment differs. Li and Cai (2026) are cited for the broader raw-Gram-versus-student-risk distinction. This does not establish a new framework, variance-optimal tuning counterexample, actual sampling-variance equality or sufficient JCM contribution. Formal PPI++ reconciliation and the choice-analysis value judgment remain open.
