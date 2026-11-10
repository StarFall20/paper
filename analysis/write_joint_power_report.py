"""Summarize the completed joint-power replay and exact-zero control."""
from pathlib import Path
import argparse,hashlib,json
import pandas as pd
from public_conjoint_optimizer_audit import save


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();r=a.results
    state=json.loads((r/'joint_power_status.json').read_text())
    assert state['completed_DGP_blocks']==12 and state['original_draws_replayed']==6000
    assert state['new_positive_power_fits']==12000 and state['original_limit_aggregate_matches']==12
    f=pd.read_csv(r/'joint_power_phase_verified.csv');raw=pd.read_csv(r/'joint_power_phase_raw.csv')
    assert len(f)==len(raw)==36 and f.unresolved.sum()==raw.unresolved.sum()==0
    assert ((f.interior+f.outside+f.boundary_or_near_boundary+f.unresolved)==500).all()
    old=pd.read_csv(r/'vanishing_power_phase_verified.csv')
    assert state['reference_sha256']==hashlib.sha256((r/'vanishing_power_phase_verified.csv').read_bytes()).hexdigest()
    original_zero=old[old.power.eq('0.0')].set_index(['labels','epsilon','dimensions'])
    positive=f[f.power_name.eq('inverse_m')].set_index(['labels','epsilon','dimensions'])
    exact_zero=[]
    for key,row in positive.iterrows():
        zero=original_zero.loc[key]
        exact_zero.append({'labels':key[0],'epsilon':key[1],'dimensions':key[2],
            'ordinary_zero_frequency':zero.finite_frequency,'strictly_positive_inverse_m_frequency':row.finite_frequency,
            'independent_half_zero_selector_expected_frequency':.5*(zero.finite_frequency+row.finite_frequency),
            'type':'analytic mixture of completed empirical frequencies;not a new random experiment'})
    pd.DataFrame(exact_zero).to_csv(r/'joint_power_exact_zero_control.csv',index=False)
    geometry=json.loads((r/'joint_power_geometry_checks.json').read_text())
    text='''# 同时选择权重的存在上界与联合极限

本轮完成了同一修正logit模型的量词补强。平衡、独立Gaussian斜率、固定零截距条件下，正权重右极限也是全部权重的同时存在上界。超过该维度/标签比例时，0至1范围内没有任何权重能恢复有限根。严格为正且依概率趋零的权重序列达到这个存在区域，权重可依赖当前数据，无需额外速率条件。

证明先把全部正权重锥放进一个闭多面锥，再计算其统计维度。闭包必须保留零尺度下非负标签坐标、零辅助坐标的非零方向。48个独立投影比较同时检验标量计算、全变量约束二次规划、显式极锥不等式和Moreau正交关系。最大坐标差7.25e-7，极锥与正交误差分别低于5.8e-15和3.5e-14。这些是几何核查，概率边界依靠正文证明及经典随机子空间定理。

原6000个Gaussian样本已按原随机种子和抽样顺序重放，在每个样本上计算1/sqrt(m)与1/m两个正权重，共12000次新权重可行性计算。12个原极限聚合结果全部一致；36个报告单元各保留500次，原始与复核结果均无未知状态。该复算没有新增独立响应样本，不能把12000次计算说成12000位参与者。

标签数80、代理错误率0.25时，两组正权重在维度36、48、56的有限比例均为0.990、0.798、0.340，与原极限可行性频率一致。对应的渐近上界为0.664488。错误率0.5时该上界为0.608737。这些有限样本频率验证与边界相容，不把0.7维度比例处仍有有限解误写成对渐近定理的反例或证明。

严格正权重条件有实质含义。若在1/2<维度/标签比例<正权重上界的模型中，独立地以一半概率取精确零、另一半取1/m，有限根概率趋于一半。附带控制表给出已有经验频率的解析混合，没有另外生成随机响应。该控制在主模拟完成后补充，用于解释条件，不冒充预设验证。

一般信号/估计截距定理仍限定固定正权重。新的联合结论没有推广到拟合预测器、重复选择相关性、任意离散属性分布，也没有给出误差分布或区间保证。固定样本的系数发散说明了更大存在区域可能付出的估计代价；两个不同极限不互相替代。

本轮也核对了2026年JOCM关于参数不确定性、偏好异质性和重复选择的论文，并阅读了最新PPI实践综述的相关章节。一般诊断流程、推断效率与合成数据正则化均已有前驱。当前候选增量是指定补偿目标的显式边界、全部权重上界及系数代价。尚未用独立增量核查、数学检查或文献未命中替代真实的投稿判断。
'''
    (r/'联合权重上界与论文深化_20261110.md').write_text(text)
    save(r/'joint_power_report_checks.json',{'completed_original_draw_replay':6000,'new_positive_power_feasibility_checks':12000,
        'original_limit_aggregate_matches':12,'raw_unresolved':0,'remaining_unresolved':0,
        'all36cells_retain500repetitions':True,'geometry_checks':geometry,
        'exact_zero_control_is_postrun_analytic_mixture':True,
        'full_signal_estimated_intercept_joint_theorem_claimed':False,
        'error_distribution_or_coverage_guarantee_claimed':False,'novelty_or_submission_readiness_proven':False})
    print('Completed joint-power report:6000 original draws,12000 new power checks,36 cells.')


if __name__=='__main__':main()
