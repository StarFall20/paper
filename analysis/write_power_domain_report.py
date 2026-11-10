"""Report the completed finite-design audit without changing the paper's claims."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from public_conjoint_optimizer_audit import save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    out = args.results
    check = json.loads((out/'power_domain_replay_checks.json').read_text())
    certificates = json.loads((out/'power_domain_exterior_certificates.json').read_text())
    summary = pd.read_csv(out/'power_domain_summary.csv')
    assert check['completed_allocations'] == 500 and check['completed_cases'] == 1000
    assert check['issues'] == 0 and check['original_unit_status_mismatches'] == 0
    assert check['interior_witnesses_independently_checked'] == check['existing_selected_power_support_interior'] == 1000
    assert check['interior_cutoff_certificates'] == len(certificates) == 8
    assert summary.allocations.eq(500).all() and summary.positive_intervals.eq(500).all()
    assert summary.endpoint_below_one.sum() == summary.unit_exterior.sum() == 8
    rows = []
    for case in certificates:
        h, a, cut = case['human_recession'], case['unit_recession'], case['closed_endpoint']
        assert h > 0 and a < 0 and 0 < cut < 1
        assert abs(h/(h-a)-cut) < 1e-7 and case['maximum_independent_identity_error'] < 1e-7
        assert case['above_endpoint_direct_recession'] < -1e-8
        assert 0 <= case['existing_selected_power'] < cut
        bracket = case['fixed_power_brackets']
        assert bracket['half_endpoint']['status'] == bracket['below_endpoint']['status'] == 'interior'
        assert bracket['at_endpoint']['status'] == 'boundary_or_near_boundary'
        assert bracket['above_endpoint']['status'] == 'outside_mean_support'
        rows.append({'prediction': case['prediction'], 'allocation': case['trial'],
                     'closed_endpoint': cut, 'existing_selected_power': case['existing_selected_power'],
                     'human_recession': h, 'unit_recession': a,
                     'above_endpoint_recession': case['above_endpoint_direct_recession'],
                     'maximum_certificate_identity_error': case['maximum_independent_identity_error']})
    cases = pd.DataFrame(rows).sort_values(['prediction', 'allocation'])
    cases.to_csv(out/'power_domain_exterior_table.csv', index=False)
    zero = check['existing_selected_zero_powers']
    lo, hi = cases.closed_endpoint.min(), cases.closed_endpoint.max()
    text = f'''# 已有选择数据的完整权重区间审计

2026年11月10日。审计沿用同一个校正logit目标、公开疫苗选择数据、原500次分配和两种原预测方式。50条标注选择下共1000个案例全部完成。它没有增加参与者、问卷或AI作答，也没有重新选择原有估计权重。

## 审计得到什么

全部1000个案例都存在正权重的严格内部见证。单位权重在992个案例中位于内部，另外8个案例位于矩支撑之外，与原结果完全一致。这8个案例的闭可行上端点介于{lo:.6f}和{hi:.6f}之间；有限解区间为严格开区间(0,L)。端点处的原始矩约束检查均为边界／数值邻近状态，上端点以上均有直接从原损失计算的负衰退方向。

| 预测方式 | 分配编号（从0开始） | 闭可行上端点L | 原已选权重 |
| --- | --- | --- | --- |
'''
    for row in cases.to_dict('records'):
        text += f"| {row['prediction']} | {row['allocation']} | {row['closed_endpoint']:.6f} | {row['existing_selected_power']:.6f} |\n"
    text += f'''
这里的分配编号是公开数据重分配的编号，不是受访者编号。五行一组沿用原资源的处理约定，不能据此认定组内属于同一个人。结果描述固定资源上的这组计算，不估计总体发生率。

原有真人初始化的权重选择在1000个案例中均位于可估计位置，包括上述8个单位权重失败案例；{zero}次选择恰好为零，按普通logit端点单独核验。新计算没有发现原选择规则遗漏的失败，也不支持用最大可行端点代替既有权重选择。最大权重差为{check['maximum_selected_power_replay_error']:.3g}。

## 完整区间如何确定

设标注样本为m，辅助样本为N，tau=N/m。保持预测g不变，写m b(lambda)=b0+lambda b1，其中b0=X_L^T Y，b1=−X_L^T g_L+(m/N)X_U^T g_U。变量u=m w p满足

X^T u=b0+lambda b1；0<=u_L<=1−lambda；0<=u_U<=lambda/tau；0<=lambda<=1。

联合最大化lambda得到闭可行端点L。另一个线性规划最大化所有盒约束的共同严格松弛t。满列秩下，t>0给出内部权重见证；有限解的内部权重区间恰为(0,L)。L=1时另验辅助矩阵的秩和单位权重矩的内部性。t=0时，闭可行性不能证明有限解：所有正权重都处在边界的例子也可得到L=1。

证明使用已有的权重嵌套关系、可行点的凸组合及衰退成本连续性。若0<L<1，端点支持方向d满足h=H^infty(d)>0、a=Q_1^infty(d)<h，且L=h/(h−a)。盒约束与端点线性规划对偶提供计算途径；原损失直接计算h和a提供独立核验。这些都是经典凸支撑与参数线性规划的应用，不作为新的通用定理或最优权重方法。

## 核验与论文中的位置

120个独立构造的数学诊断设计完成720次原矩／重缩放固定权重比较；59个有正权重内部区间，61个具有独立非零弱分离阻碍。30个端点比值另有衰退成本核验。五个标量例子覆盖可恢复、全边界、全外部、L=0.625及L=1端点排除的情况，另检查秩不足。

公开数据1000个案例分别核验严格内部见证和原选权重，所有原单位状态一致。最大原始等式／盒约束误差为{check['maximum_primal_residual_or_violation']:.3g}，最大端点对偶恒等式误差为{check['maximum_endpoint_dual_identity_error']:.3g}。所有数值状态均保留，没有把未知状态改写为成功。

这项审计可作为正文有限样本支撑检查的计算补充。它把原来逐个权重检查扩展为完整区间，结果也表明既有选择规则已经处理这组失败案例。它不能单独补足论文的原创性与投稿贡献，故不新增框架名称，不把它扩张成正文的独立核心贡献。主稿、已有成熟DY比较和三个主表保持原样。原稿的高斯边界结论仍受原假设约束，不能据这项离散数据计算宣称获得一般DCE概率边界。

输入、原分配、预测例程和原结果的哈希在冻结记录中。完整1000行逐分配诊断仅保留本地；公开包提供汇总、8个分配层面的方向证书及重算入口，不含私人问卷或响应向量。样本来源说明与该公开数据审计分开保存。
'''
    (out/'完整权重区间审计_20261110.md').write_text(text)
    save(out/'power_domain_report_checks.json', {
        'completed_cases': 1000, 'original_exterior_cases_retained': 8,
        'every_exterior_cutoff_and_original_selected_power_checked': True,
        'minimum_closed_endpoint_among_exterior': float(lo), 'maximum_closed_endpoint_among_exterior': float(hi),
        'all_original_selected_powers_remain_admissible': True,
        'positive_power_domain_does_not_include_zero_endpoint': True,
        'new_estimator_or_optimal_power_claim': False,
        'classical_parametric_LP_claimed_independently_novel': False,
        'manuscript_scientific_claims_or_main_tables_changed': False,
        'submission_readiness_proven': False,
    })
    print(cases[['prediction','allocation','closed_endpoint','existing_selected_power']].to_string(index=False))
    print(json.dumps({'cases': 1000, 'certificate_cases': 8, 'cutoff_min': float(lo), 'cutoff_max': float(hi), 'classical_audit_only': True}))


if __name__ == '__main__': main()
