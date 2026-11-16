"""Reconcile all new population,rotation and paired phase outputs."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from public_conjoint_optimizer_audit import save


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();out=a.results
    population=json.loads((out/'aligned_prediction_numeric_checks.json').read_text())
    calibrated=json.loads((out/'calibrated_aligned_numeric_checks.json').read_text())
    phase=json.loads((out/'aligned_phase_status.json').read_text())
    rotation=json.loads((out/'trained_prediction_rotation_checks.json').read_text())
    assert population['population_conditions']==18 and calibrated['population_conditions']==12
    assert population['original_oriented_LP_matches']==234 and len(population['separately_verified_rank_deficient_cases'])==6
    assert phase['completed_label_blocks']==2 and phase['paired_draws']==1000 and phase['feasibility_calls']==8000
    assert phase['completed_cells']==16 and phase['unknowns']==0 and rotation['corrected_objective_support_rotation_matches']==180
    source=Path(__file__).parent
    for name,record,key in [('attribute_aligned_prediction.py',population,'source_sha256'),('calibrated_aligned_prediction.py',calibrated,'source_sha256'),('aligned_prediction_phase.py',phase,'source_sha256'),('verify_trained_prediction_rotation.py',rotation,'source_sha256')]:
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==record[key]
    assert calibrated['population_helper_sha256']==population['source_sha256']
    frame=pd.read_csv(out/'aligned_prediction_phase.csv');raw=pd.read_csv(out/'aligned_phase_trial_diagnostics.csv')
    assert len(frame)==16 and frame.replications.eq(500).all() and len(raw)==8000
    assert raw.groupby(['labels','total_slopes','model']).size().eq(500).all()
    assert raw.status.isin(['interior','outside_mean_support']).all()
    pairs=[]
    for (m,d),part in raw.groupby(['labels','total_slopes']):
        wide=part.pivot(index='replication',columns='model',values='status');assert len(wide)==500
        diff=wide.aligned.eq('interior').astype(float)-wide.external.eq('interior').astype(float)
        displayed=frame[(frame.labels==m)&(frame.total_slopes==d)]
        for model in ['external','aligned']:
            r=displayed[displayed.model==model].iloc[0]
            assert r.interior==int(wide[model].eq('interior').sum()) and r.outside_mean_support==500-r.interior
            assert abs(r.paired_finite_difference_aligned_minus_external-diff.mean())<1e-12
            assert abs(r.paired_difference_MCSE-diff.std(ddof=1)/np.sqrt(500))<1e-12
        pairs.append({'labels':m,'total_slopes':d,'external_frequency':wide.external.eq('interior').mean(),
                      'aligned_frequency':wide.aligned.eq('interior').mean(),'paired_difference':diff.mean(),
                      'paired_MCSE':diff.std(ddof=1)/np.sqrt(500)})
    pair=pd.DataFrame(pairs);pair.to_csv(out/'aligned_phase_paired_summary.csv',index=False)
    calibrated_grid=pd.read_csv(out/'calibrated_aligned_boundaries.csv')
    assert (calibrated_grid.boundary_gap>=0).all()
    example=calibrated_grid[(calibrated_grid.epsilon==.1)&(calibrated_grid.tau==1)&(calibrated_grid.power==1)].iloc[0]
    report={'new_population_conditions':30,'paired_Gaussian_draws':1000,'complete_phase_cells':16,'phase_support_checks':8000,
            'phase_unknowns':0,'all_8000_local_raw_statuses_reconciled':True,'same_response_prediction_law_by_construction':True,
            'calibrated_example_Brier':.09,'external_unit_boundary':example.external_critical_ratio,'aligned_unit_boundary':example.aligned_critical_ratio,
            'unit_boundary_gap':example.boundary_gap,'finite_reverse_order_rows':int((pair.paired_difference>0).sum()),
            'original_box_checks':234,'rank_cases_separately_retained':6,'independent_cone_projection_cases':16,
            'independent_trained_rotation_fits':60,'trained_support_identity_checks':180,
            'independent_training_parameters_must_converge':True,'arbitrary_cross_fitting_proven':False,
            'new_behavioral_framework_or_training_method':False,'priority_or_submission_readiness_proven':False}
    save(out/'aligned_report_checks.json',report)
    text=f'''# 预测器与入模属性的关系及模型深化

2026年11月10日。原稿要求剩余高斯属性与预测器使用的信息独立；这不能直接覆盖用这些属性训练的预测器。新增命题4把固定数量的预测索引保留为已拟合属性，沿用同一个校正logit目标和几何证明。附录E给出完整论证。没有再引入框架名称，也没有增加行为机制。

## 现在成立的结果

在相同响应、相同预测值、相同校准程度和相同边际高斯属性分布下，改变预测索引是否属于入模属性，会改变有限解边界。这不是用不同误差水平比较两个预测器。校准预测的例子保持Brier风险0.09不变，单位权重、等量辅助样本时，边界分别为{example.external_critical_ratio:.6f}与{example.aligned_critical_ratio:.6f}，差{example.boundary_gap:.6f}。严格位于两者之间的渐近维数比有相反的存在性结论。

差异由外部模型最优点处严格为负的索引方向导数证明。模型不靠更多重复次数或更小p值定义创新。最初18个无预测能力的条件中，最大差距不足0.006；后续12个校准条件中最大约0.011。完整网格均保留，不把这种幅度写成普遍的大效应。

若预测器是独立训练的线性logistic分数，旋转高斯属性后，它可由效用方向、一个正交预测方向及截距表示。新增推论2要求这三个训练量收敛到有限确定极限，并据此给出固定正权重与趋零正权重的存在性边界。证明使用统一定位及条件几何。它没有证明任意学习算法的训练量收敛，也不覆盖任意交叉拟合、黑箱预测或一般离散DCE属性。

## 完整计算

新增30个总体条件、120组有限设计的234次原矩／盒约束对照、16个独立锥投影及60次独立训练后的180次旋转支撑对照。另有6个单位权重案例的辅助矩阵秩不足；它们按不可识别单列，没有混入预测器效应。最初将秩状态与可行性状态直接比较的断言失败及原始源码快照保留，修正未改变原120个诊断设计。

配对模拟选择校准网格中差距最大的条件作说明，选择发生在总体计算之后，正文已标明。80与160个标注行各500组共享模拟，总计8000次支撑计算；16个单元全部保留500次，没有未决状态。逐次原始状态与所有汇总已核对。每行比较拟合相同数量的系数，有限样本中的高斯余项列数相差一列，已明确记录。

| 标注行 | 斜率数 | 外部索引有限解频率 | 入模索引有限解频率 | 配对差 | 蒙特卡洛标准误 |
| --- | --- | --- | --- | --- | --- |
'''
    for r in pairs:text+=f"| {r['labels']} | {r['total_slopes']} | {r['external_frequency']:.3f} | {r['aligned_frequency']:.3f} | {r['paired_difference']:.3f} | {r['paired_MCSE']:.4f} |\n"
    text+='''
配对差为入模减外部。80行、38个斜率处出现反向有限样本排序，完整保留；渐近边界顺序不要求每个有限样本逐点嵌套。各维数共用模拟，不能把表中各行当作独立重复实验。这组结果没有证明估计误差、区间覆盖或最佳权重的优势。

## 与已有工作的界限

相关教师／拟合特征空间的研究早已考虑几何关系；独立训练与交叉拟合本身也已有成熟方法。原文和源码读取范围已单独记录。新增内容是同一补偿型目标下的索引约束、明确的同分布校准反例和有条件的训练预测器边界。它补上实际适用范围的一处缺口；是否足以构成JOCM要求的独立重要增量，仍须结合最近前驱与选择分析用途判断。

旧的DY成熟基线、七种原方法网格、全部三个主表以及上一轮1000个公开数据案例均保留。新版正文增加3.7节、命题4、推论2、附录E及一张附录表；没有把私人300份真人来源问卷或60份AI预测试带入公开方法稿。原始问卷文件不变，没有新增真人招募或AI作答。
'''
    (out/'预测索引模型扩展与当前进展_20261110.md').write_text(text)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
