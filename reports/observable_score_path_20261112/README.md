# 可观察得分、方差参考与有限解：公开选择数据复核

2026年11月12日。本轮沿用当前论文的同一个补偿Logit目标，检验理论中的局部等价路径能否连接到实际计算。当前42页正文和Word保持原样；本补充材料供下一次实质整合使用。

## 这次补充解决什么

理论中的真实属性概率可换成一个已知参考参数产生的预测。固定该参考后，变换预测并补偿权重，能让每条人工与辅助得分、参考点处的总梯度及整个经验得分协方差中间矩阵完全一致。对严格处于0和1之间的初始权重，重新计算原有方差权重规则会返回路径上的对应权重。目标函数的曲率和有限解状态仍可变化。

[推导与精确例子](observable_anchor_derivation.md)明确区分可观察参考与未知真实参数。这是一项有限设计比较，没有把拟合参考当成真实概率，也没有声称拟合预测的数据复用自动满足原来的独立性条件。实际Hessian构成的完整夹心协方差可以变化；相同的是得分中间矩阵及使用共同信息矩阵的局部参考。校准、收缩与通用凸支撑继续归属已有方法。

精确标量例子只用两条人工记录、两条辅助属性和已知参考0。原有方差规则给出权重15/88。有限解区间为[15/88,9/44)，在9/44处无有限驻点，超过9/44后目标无下界。整个路径的得分行和经验得分中间矩阵相同，所有预测严格处于0和1之间。[分数运算记录](results/high/exact_example/observable_rational_example.json)不使用优化器或数值积分。该例子用于解释机制，未作为脱离主定理的独立方法贡献。

## 公开数据的两个预算

两组均重放原公开疫苗数据的同一500个区组排列，固定1200条前缀和原11个属性对比。五行区组沿用原研究代码约定，不能当作已经核实的受访者标识。没有增加真人、AI或模拟选择答卷。

- 50条人工选择：使用原有人工Logit参考和既有方差权重，先固定[方案](plan.json)，再计算全部500次分配、两类预测和五个路径点。所有5000个目标都有有限解，未观察到存在性差异。
- 20条人工选择：在前一组首260次分配未显示差异后，单独固定[低样本方案](low_label_plan.json)，用同一20条人工选择拟合既有尺度的岭回归参考，再完成全部500次分配。它是后续压力检验，不能写成事先与第一组同时设定。原预测程序的先验回退全部保留。

下表只列初始方差权重严格处于0和1之间的分配；这些比较中的重新计算权重均有定义。每行的分配数是条件分母，不能替换为500。

| 人工选择条数 | 预测信息 | 分配数 | 原目标有有限解 | 终点有有限解 | 无解转有限 | 有限转无解 |
|---|---|---:|---:|---:|---:|---:|
| 50 | AI辅助 | 201 | 201 | 201 | 0 | 0 |
| 50 | 仅属性 | 176 | 176 | 176 | 0 | 0 |
| 20 | AI辅助 | 80 | 76 | 79 | 4 | 1 |
| 20 | 仅属性 | 65 | 62 | 64 | 3 | 1 |

20条选择时的4个与3个原目标均有严格负衰退方向，路径终点由独立可行性计算确认处于内部。两组还各有1个反向案例：原目标有有限解，终点位于支撑外。共9个“预测信息×分配”比较的存在性结论改变，涉及6个区组排列；不能把它们当成9个独立参与者。这些案例说明：在真实的公开属性和选择记录上，相同的可观察局部得分参考仍需配合对拟合目标的全局检查。它们没有确立一个新的最佳权重或预测处理策略。

零初始权重的机制单列。20条选择的AI辅助组有420个零权重分配，原目标在85个分配中有有限解、335个发生普通Logit边界分离；仅属性组对应435个分配中的88个与347个。路径终点分别有356个与371个有限解，各有64个支撑外目标。正权重下预测完全等于参考概率，预测得分方差为零，重新计算的方差比例可能无定义，不能把这些终点称为原规则选择的等价最优权重。

20条选择的所有500次分配中，终点目标均为435个有限解；参考岭回归本身在全部500次分配中有有限解。普通人工Logit仅109个有有限解、391个有已核实的弱分离方向。这里没有评价预测误差或区间覆盖，也不据此推荐路径终点优于岭回归。

## 独立核对与复现

共保留10000条路径目标记录，另核对20条选择下的500个人工目标。来源、方案、原预测程序和当前分析文件的哈希均已保留。完整记录没有未决状态。

[独立核对](results/high/verification/observable_path_complete_verification.json)重新计算870个支撑状态，并用另一种残差内部点线性规划逐一比对；验证全部490条支撑外方向及391条普通Logit弱分离方向。逐行得分最大误差2.23e-16，中间协方差最大误差2.17e-18，重新计算权重最大误差5.17e-13。分母和零权重分层见[完整分层配对表](results/high/verification/power_stratified_pairs.csv)。这次核对没有重新执行全部10000个原始支撑程序。

初次核对在标量例子的协方差接口处停止：旧的11变量权重辅助函数对单列协方差返回值没有作矩阵包装。修正仅位于新核对器的标量公式，原始11变量分析依赖和两组结果不变。[失败记录与原核对器](diagnostics/initial_verifier_failure.json)保留，修正后的完整核对已重新运行。

从仓库根目录运行，依赖沿用当前投稿复现包。公开数值输入按[现有下载说明](https://github.com/StarFall20/paper/blob/35b98d47d24b204324e3a7ffac21f75cc8002e06/submission/finite_choice_20261110/README.md)准备为local_inputs，也可使用下列下载命令和[固定输入清单](public_source_manifest.json)。[1000行原权重参考](reference_power_weights.csv)仅保留50条选择、人工初始化方法的原始权重和状态，可由原有公开数据权重对照程序重新生成。

```sh
python analysis/fetch_public_conjoint_audit.py --manifest reports/observable_score_path_20261112/public_source_manifest.json --out local_inputs
python analysis/observable_score_path_audit.py --plan reports/observable_score_path_20261112/plan.json --input-dir local_inputs --reference reports/observable_score_path_20261112/reference_power_weights.csv --out regenerated/observable_high
python analysis/observable_low_label_path.py --plan reports/observable_score_path_20261112/low_label_plan.json --input-dir local_inputs --out regenerated/observable_low
python analysis/certify_observable_score_path.py --out regenerated/observable_high/exact_example
python analysis/verify_observable_score_path.py --input-dir local_inputs --high regenerated/observable_high --low regenerated/observable_low --out regenerated/observable_high/verification
```

也可把本目录results/high、results/low的保留结果复制到新目录后运行核对器，无须重跑所有分配。当前依赖闭包见[最终来源记录](final_source_hashes.json)。原始问卷、下载的数值输入和外部研究代码均不随本补充发布。

## 对创新主张的影响

[最接近文献阅读记录](closest_prior_scope.json)补充了CC-PPI的固定维度存在性与条件风险结果、预测校准文献和2026年带正性约束的多项选择存在性论文。广义的“一阶相同、有限样本不同”、校准预测、收缩和诊断有限解都已有明确基础。

本论文仍应以指定补偿损失的明确比例维度存在边界及其同一模型内的后果为核心。可观察参考路径加强了机制解释和公开数据联系。精确标量例子、代数恒等式和这9个案例不能单独证明足够原创或达到投稿门槛。

带正性约束论文的完整定理尚未取得。OpenReview上一条多项Logit相变标题线索的官方端点返回验证要求，未读取其正文。15条[网络检索记录](search_log.json)已保存；Scopus与Google Scholar系统检索仍未完成。搜索未检出同结论不能当作原创性证据。当前42页论文及其既有复现包仍保留，研究目标继续进行。
