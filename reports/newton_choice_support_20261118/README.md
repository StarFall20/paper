# 校正多项 Logit 的有限解诊断

在两套公开离散选择数据的 2,576 个既有固定划分中，344 个单位权重校正目标在原支撑检验中判定为无下界。每个目标在带标签岭回归的评估点上都有正定 Hessian 和有限 Newton 步。优化器能够计算局部步长，仍需单独检查有限解是否存在。2,576 个记录来自重复划分。同一公开数据内的划分共享受访者。

利用同一个步长可构造逐菜单概率。如果概率严格为正且精确满足校正矩，目标有唯一有限解。另一个可直接检查的条件是沿该方向的衰退值严格为负，此时目标无下界。两个条件均不触发时，交给原来的支撑程序。

| 数据 | 原判定有唯一有限解 | 原判定无下界 | 岭回归评估点确认有解 | 岭回归评估点确认无下界 |
|---|---:|---:|---:|---:|
| RADIANT 食品 | 1,835 | 165 | 365 | 34 |
| Swissmetro 交通 | 397 | 179 | 23 | 82 |
| 合计 | 2,232 | 344 | 388 | 116 |

预先记录的主检验直接判定 504/2,576 宗，均与存档的独立支撑判定一致。只用最大范数的充分条件确认了 2 宗。查看主结果后进行的敏感性检验改用冻结辅助预测的菜单概率。两种评估点合计直接判定 606 宗，包括 455 宗有解与 151 宗无下界，剩余 1,970 宗进入支撑程序。此处报告的是既有固定划分的覆盖率，计算时间和样本外覆盖率均未估计。

复核使用另一种协方差计算和属性单位，将系数列乘以 0.25 到 8 的不同固定倍数，逐个对照两种评估点的符号与概率约束。[独立复核结果](results/verification.json)

数学构造及与经典卡方校准的关系见 [推导](proof.md)。实证增量是量化局部数值诊断在该类校正选择模型里的盲点，以及可跳过支撑程序的固定样本覆盖率。现行 Word 稿件此前已整合有限阶局部匹配定理，本报告供独立复核。

## 复现

软件版本见 `requirements.txt`。压缩包中的 `upstream` 保存必要的派生结果与预先冻结的配置。两个公开数据文件需单独下载。脚本检验输入文件的 SHA-256，并按参与者恢复既有划分。[RADIANT 数据](https://zenodo.org/records/15420092) [Swissmetro 数据](https://github.com/michelbierlaire/biogeme/blob/master/src/biogeme/data/data/swissmetro.dat)

```sh
FOOD_DATA="/path/to/COSCBB.xlsx"
SWISS_DATA="/path/to/swissmetro.dat"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python audit.py --food-data "$FOOD_DATA" --food-archive upstream/food \
  --swiss-data "$SWISS_DATA" --swiss-archive upstream/swissmetro --out regenerated/primary

python frozen_proxy.py --food-data "$FOOD_DATA" --food-archive upstream/food \
  --swiss-data "$SWISS_DATA" --swiss-archive upstream/swissmetro \
  --primary regenerated/primary --out regenerated/primary/frozen_proxy

python verify.py --food-data "$FOOD_DATA" --food-archive upstream/food \
  --swiss-data "$SWISS_DATA" --swiss-archive upstream/swissmetro \
  --results regenerated/primary --sensitivity regenerated/primary/frozen_proxy \
  --out regenerated/primary/verification.json
```
