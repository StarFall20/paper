# 官方 PPI 程序的实际运行核对

本次首次实际调用固定版本的官方 `ppi_logistic_pointestimate` 接口。此前的官方代码记录是静态读取，既有单步比较是明确标注的公式实现。本次没有改写官方源码，以正常方式导入整个包，使用独立依赖环境；仅通过两个原样转交参数和结果的包装函数记录优化器状态与选出的权重。代码来源为 [aangelopoulos/ppi_py 固定提交](https://github.com/aangelopoulos/ppi_py/tree/3d1f0c668444907b39bd045cb9dd38e479ce7dd6)，每个源文件的 Git blob 和 SHA-256 在运行前后均核对。

## 已完成的证据

[方案](plan.json)在第一次官方拟合前固定。真实数据部分复用既有公开疫苗选择分析中，50 标签预算下全部 8 个单位权重外部目标案例，并为两种预测各取最早的 4 个内部目标案例作为正常对照。每个案例调用零、半、单位权重和官方自动选权重，共 64 次。另用已有匹配参考信息构造中的 4 个解析设计核对有限、边界和外部状态，总计 68 次调用。没有增加参与者、独立模型队列或 AI 答卷。该疫苗源缺少受访者编号，不能作为独立受访者研究解释。

真实数据结果如下。每行 8 个选定案例，不能将此表当作失效率估计。

| 已知案例组 | 请求权重 | 返回数值有限的数组 | 底层最终优化器成功 | 接口异常 |
|---|---:|---:|---:|---:|
| 先前外部目标 | 0 | 8 | 8 | 0 |
| 先前外部目标 | 0.5 | 8 | 8 | 0 |
| 先前外部目标 | 1 | 8 | 0 | 0 |
| 先前外部目标 | 自动 | 0 | 0 | 8 |
| 正常内部对照 | 0 | 8 | 8 | 0 |
| 正常内部对照 | 0.5 | 8 | 8 | 0 |
| 正常内部对照 | 1 | 8 | 8 | 0 |
| 正常内部对照 | 自动 | 8 | 8 | 0 |

所有 8 个单位权重外部目标的原始损失都有独立核对的负衰退方向。官方接口返回的系数范数约为 1.14×10³⁵ 至 4.87×10⁵⁴，原始得分最大分量在 0.00417 至 0.02071 之间。底层优化器有 7 个异常线搜索结束、1 个达到本次设定的迭代上限；接口直接返回其系数数组，没有返回该优化状态。本次统一设置为 `ftol=1e-15`、`gtol=1e-8`、最多 1000 次迭代和 20000 次函数计算。这些是记录过的运行设置，不能称为所有环境下的默认表现。

这 8 个案例的自动选权重调用均在单位权重起点后报 Hessian 奇异，尚未产生最终权重。另写的原始行计算独立验证了这 8 个初始矩阵的秩不足。零权重和半权重在这些选定案例中均具有内部目标；8 个正常对照的全部 32 次调用也具有内部目标。这些控制保留了官方程序正常工作的情形，没有把所有返回值归为错误。

解析设计进一步区分了优化状态与全局存在性。既有精确推导给出临界权重 11/20。在该有理数设计的浮点输入上，官方程序返回系数范数约 32.10，得分最大分量约 5.11×10⁻¹⁵，底层优化器报告成功。精确设计的下确界有限却不取到；浮点 LP 仍标注为边界或近边界。梯度容差和优化器成功标志均不能单独证明存在有限最优解。半权重则返回约 3.67677 的有限斜率；0.6 和 1 的原始目标无界。

[另写的核对程序](analysis/verify_original_objectives.py)不导入主核对器或官方包，重验全部 68 条接口记录、60 个返回系数、76 次原始残差盒 LP、8 个自动选权重结果和 8 个初始奇异矩阵。原始得分差异不超过 3.1×10⁻¹⁶，权重差异不超过 1.9×10⁻¹⁵。[完整核对记录](verification/verification.json)、[原始调用记录](results/api_records.json)及[汇总](results/overview.json)保留全部结果。这里的“独立”指另写的计算实现，不表示外部同行评审。

## 对论文的作用

本次运行补上了软件层面的实际证据：本稿的支撑判别可以在接受拟合结果之前识别原目标无有限解的情况。无界案例由原始目标证明，不能通过增加迭代次数改变；正常对照也必须保留。选定案例不支持普遍失效率或不同方法预测优劣的结论。官方 PPI++ 的固定维度定理不因此受到反驳。

这不是新增估计器或独立理论框架。原始损失的凸支撑、分离和衰退判断归属既有理论，现有论文的候选增量仍是特定补偿选择目标的比例维度边界及其后果。当前已审阅 Word 和原有数值结果保持不变；[拟加入稿件的文字](proposed_manuscript_text.md)单独列出，尚未自动写入正文。正式版 PPI++ 正文及 A–E 附录仍未取得，作者主页目前也链接到 29 页的 2024 年 arXiv v2。[本次检索与访问记录](access_and_reading_scope.json)不将其记为正式版读取。研究目标尚未完成。

## 复现

原始公共数据和官方源码均单独下载，不包含在本包中。`reference/analysis` 里的四份既有分析文件用于恢复相同的公共输入和预测；它们保持原哈希。准备环境与官方运行环境分开，以保持既有冻结预测不受软件版本切换影响。两个环境各自的依赖版本分别见 `requirements_prepare.txt` 和 `requirements_official.txt`。

公共数值文件来自 [Lu、Wang、Zhang 与 Zhang 的生成增强推断研究仓库](https://github.com/ChengLu1999/generative-augmented-inference/tree/d18717206867c7a12ff470879a0dd5d488b300c2)。其固定 README 将代码标为 MIT、数据标为 CC-BY-4.0；下载清单绑定原文件大小与 Git blob。这里重用已发布选择，不声称复现其论文表格或使用已知真实效用系数。官方 PPI 源码的 MIT 许可见 `UPSTREAM_LICENSE.txt`。

在准备环境运行：

```sh
python reference/analysis/fetch_public_conjoint_audit.py --manifest reference/source_manifest.json --out local_inputs
python analysis/prepare_inputs.py --data-dir local_inputs --out local_frozen
python analysis/fetch_upstream.py --out local_upstream
```

切换到官方运行环境，固定单线程运行：

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python analysis/run_official_api.py --input-dir local_frozen --upstream local_upstream --out regenerated/results
```

主脚本的 `--out` 可用于保存新的调用记录。独立核对器默认检查包内 `results` 的已完成记录；若要核对重跑结果，在另一个完整包副本中将 `--out` 指向该副本的 `results`，再运行：

```sh
python analysis/verify_original_objectives.py --input-dir local_frozen --out regenerated/verification
python analysis/summarize_audit.py
```

先运行 `python verify_package.py` 核对文件清单。下载器只读取固定公开地址并检查内容哈希，不执行下载的源文件；官方主脚本明确导入并执行已核对的官方软件。原始公共响应行、用户问卷、原始论文 PDF、身份和会话信息均不进入本包或 GitHub 备份。
