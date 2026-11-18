# 固定权重和预测器的有限估计反例

三套模型共用真人标签、样本量、预测器及校正权重。只改变辅助题目的属性组合。校正得分逐条相同，Hessian、经验协方差和预先指定阶数内的局部导数均相同。三套校正目标分别有唯一有限解、有限下确界但无法达到、无下界。真人普通 Logit 有唯一有限解。

32 条标签和 32 条辅助记录即可实现一个三阶完全相同的例子。固定单位权重和同一个校准预测器，正斜率方向的衰退率依次是 \(+1/32,\ 0,\ -1/32\)。有限情形的斜率是 \(2\log 7\)，边界情形的下确界是 \((11/16)\log2\)。三套数据都可以在同一个正确设定的总体中以正概率出现。频率与外部数据价值由稿件内的真实选择数据分析另外衡量。[通用证明与小样例](proof.md)

复用此前的 15 组有限矩匹配设计，在 \(1/2\) 和 1 两种固定权重下分别构造，共计 30 组，每组包含三套辅助设计。最高验证到 33 阶。验证器从分组计数重算原三个损失、每组协方差、导数和衰退率，并以 75 位十进制精度核对有限斜率和边界损失。[构造方案](plan.json) · [核验结果](verification.json)

可用 Python 标准库复现。以此前复现包中的 local_derivative_attainment/results/designs.json 为输入。脚本校验其指纹。

    python construct.py --source /path/to/local_derivative_attainment/results/designs.json --out regenerated-designs.json
    python verify.py --designs regenerated-designs.json --out regenerated-verification.json

对最接近工作的具体差异见[文献核对](literature.md)。
