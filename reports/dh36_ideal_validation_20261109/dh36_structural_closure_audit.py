"""Closure audit: multinomial information, candidate-map scope and version contrasts.

This post-outcome explanatory audit uses the same full 360-record population.
It does not replace the unrestricted sufficiency test or verify recruitment.
"""
from __future__ import annotations
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import norm
from threadpoolctl import threadpool_limits
import dh36_observational_analysis as old
from dh36_assumption_conditioned_validation import js
from dh36_panel_model import fit_fixed_price


def categorical_information(p,unequal=False,null_direction=False):
    J=3; T=8; total=2*T; H=np.vstack([np.eye(2),-np.ones((1,2))])
    fractions=np.linspace(.2,.8,T) if unequal else np.full(T,.5)
    h=np.column_stack([-.6-np.arange(T)/10,.9+np.arange(T)/12])
    if null_direction: h[:]=np.linspace(-.5,.7,T)[:,None]
    a=np.column_stack([.02+.001*np.arange(T),-.012-.0004*np.arange(T),-.008-.0006*np.arange(T)])
    assert np.max(abs(a.sum(1)))<1e-14
    joint=np.zeros((17,17)); residual_formula=0.; cases=[]
    for t in range(T):
        q=np.array([1-fractions[t],fractions[t]])
        vcov=np.diag(p[t,:2])-np.outer(p[t,:2],p[t,:2])
        kappa=float(a[t,:2]@np.linalg.solve(vcov,a[t,:2]))
        kappa_full=float(np.sum(a[t]**2/p[t])); assert abs(kappa-kappa_full)<1e-13
        mu=float(q@h[t]); var=float(q@((h[t]-mu)**2)); residual_formula+=kappa*var/T
        for v in range(2):
            D=np.zeros((J,17)); D[:,2*t:2*t+2]=H; D[:,-1]=a[t]*h[t,v]
            weight=q[v]/T; joint+=weight*(D.T/p[t])@D
            cases.append((t,v,weight))
    schur=float(joint[-1,-1]-joint[-1,:-1]@np.linalg.solve(joint[:-1,:-1],joint[:-1,-1]))
    klrows=[]
    for eta in [.1,.05,.01,.005,.001]:
        kl=0.
        for t,v,weight in cases:
            px=p[t]+eta*a[t]*h[t,v]
            q=np.array([1-fractions[t],fractions[t]]); pooled=p[t]+eta*a[t]*float(q@h[t])
            assert min(px.min(),pooled.min())>0
            kl+=weight*float(np.sum(xlogy(px,px/pooled)))
        approx=.5*eta*eta*residual_formula
        klrows.append({'eta':eta,'profiled_KL':kl,'half_eta2_information':approx,'ratio':kl/approx if approx>1e-15 else None})
    assert abs(schur-residual_formula)<1e-12
    if not null_direction: assert abs(klrows[-1]['ratio']-1)<1e-3
    else: assert abs(schur)<1e-12 and max(abs(r['profiled_KL']) for r in klrows)<1e-14
    return {'categories':J,'fibres':T,'support_points':total,'conditional_fibre_weights':'0.2 to 0.8' if unequal else '1/2 each','direction':'function of phi only' if null_direction else 'varies inside fibres','nuisance_probability_dimensions':16,'joint_Fisher_dimensions':17,'Schur_information':schur,'E_kappa_conditional_variance':residual_formula,'absolute_difference':abs(schur-residual_formula),'profiled_KL_checks':klrows,'scope':'local categorical iid observation information; not a repeated-record NCP and not an originality certificate'}


def raw_menu_geometry(spec):
    steps=[]; design=[]; menus=[]
    for t,task in enumerate(spec['tasks']):
        for v in range(2):
            raw=[]
            for j,alt in enumerate(['A','B']):
                vals=np.array([int(s[1:]) for s in spec['profiles'][f'V{v+1}'][alt]])
                raw.append(vals)
                design.append([j==0,j==1,*vals,(task['prices'][j]-89)/10,task['sharing'][j]=='G1',task['sharing'][j]=='G2'])
            menus.append((t,v,raw,task))
        for alt in ['A','B']:
            a=np.array([int(s[1:]) for s in spec['profiles']['V1'][alt]])
            b=np.array([int(s[1:]) for s in spec['profiles']['V2'][alt]])
            steps.append(b-a)
    steps=np.array(steps); design=np.array(design,float)
    assert np.all(steps==np.array([1,-1,1,-1]))
    rows=[]
    for wD,wI in itertools.product([.8,1.,1.2],repeat=2):
        keys=[]
        for t,v,raw,task in menus:
            vec=[]
            for j in range(2):
                D,S,I,E=raw[j]; vec.extend([round(wD*D+S,12),round(wI*I+E,12),task['prices'][j],task['sharing'][j]])
            vec.extend(['outside']); keys.append(tuple(vec))
        unique=len(set(keys)); rows.append({'D_weight_S_weight_1':wD,'I_weight_E_weight_1':wI,'raw_menus':16,'distinct_candidate_menus':unique,'finite_menu_contrast_dimension':16-unique,'changes_the_target_null':not(wD==1 and wI==1)})
    return {'raw_step_all_16_alternatives':steps[0].tolist(),'step_span_rank':int(np.linalg.matrix_rank(steps)),'raw_linear_design_columns':design.shape[1],'raw_linear_design_rank':int(np.linalg.matrix_rank(design)),'identified_additive_raw_attribute_combination':'beta_D - beta_S + beta_I - beta_E','additive_model_version_prediction':'the same shift in A and B utility relative to outside; no version shift in conditional A/B odds','candidate_weight_preservation_conditions':['w_D = w_S','w_I = w_E'],'interpretation':'Weights define different candidate maps and different nulls. Loss of fibres after changing the map is a scope fact, not a failed test of the original map.'},pd.DataFrame(rows)


def model_contrasts(name,beta,cov,version_indices):
    ia,ib=version_indices; rows=[]
    for nm,weights in [('common_service_shift',[.5,.5]),('relative_A_vs_B_log_odds',[1.,-1.])]:
        c=np.zeros(len(beta)); c[[ia,ib]]=weights; effect=float(c@beta); se=float(np.sqrt(c@cov@c)); z=effect/se
        rows.append({'model':name,'contrast':nm,'estimate':effect,'cluster_se':se,'ci_low':effect-1.95996398454*se,'ci_high':effect+1.95996398454*se,'p_model_conditional_exploratory':float(2*norm.sf(abs(z))),'odds_ratio_if_relative_contrast':float(np.exp(effect)) if nm.startswith('relative') else None})
    return rows


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',type=Path,required=True); ap.add_argument('--instrument',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args(); out=args.out
    spec=json.loads(args.instrument.read_text()); d,*_=old.read_exports(args.input_dir,spec)
    d=d[['participant_id','version','version2']+[f'q{i}' for i in range(1,9)]].copy(); assert len(d)==360
    base=old.menu_arrays(spec); fits={}; rows=[]; covariances={}
    common=old.fit_mnl(d,base,'mnl_common',validation=True)
    p=common['probabilities'][0]
    cases=[categorical_information(p,unequal=u,null_direction=n) for u in [False,True] for n in [False,True]]
    js(out/'multinomial_information_verification.json',{'cases':cases,'all_algebra_and_KL_checks_passed':True,'theory_status':'standard nuisance-projection identity; no claim of a new general theorem'})
    geometry,weights=raw_menu_geometry(spec); js(out/'additive_profile_geometry.json',geometry); weights.to_csv(out/'candidate_weight_scope_audit.csv',index=False)
    fit=old.fit_mnl(d,base,'mnl_cohort_intercepts',validation=True)
    rows.extend(model_contrasts('MNL_version_ASC',fit['beta'],fit['cov'],[5,6]))
    covariances['MNL_version_ASC']={'beta':fit['beta'],'covariance':fit['cov'],'score_unit':'entire record of 8 choices'}
    yy=d[[f'q{i}' for i in range(1,9)]].to_numpy(int)-1
    fit=fit_fixed_price(yy,d.version2.to_numpy(int),base,nodes=161,interaction=True)
    rows.extend(model_contrasts('random_intercept_version_ASC',fit['beta'],fit['cov'],[6,7]))
    covariances['random_intercept_version_ASC']={'beta':fit['beta'],'covariance':fit['cov'],'score_unit':'integrated entire record likelihood'}
    pd.DataFrame(rows).to_csv(out/'exploratory_version_contrast_decomposition.csv',index=False); js(out/'aggregate_parameter_covariances.json',covariances)
    # Cross-check displayed version estimates against the already released fits.
    prior=pd.read_csv(out/'random_intercept_model_coefficients.csv').query("model=='random_intercept_version_ASC'")
    assert np.max(abs(prior.estimate.to_numpy()-fit['beta']))<1e-5
    tech=json.loads((out/'technical_verification.json').read_text())
    assert all(hashlib.sha256((args.input_dir/f'{v}_raw.xlsx').read_bytes()).hexdigest()==tech['raw_sha256'][v] for v in ['V1','V2'])
    m=rows[0:2]; r=rows[2:4]
    text=f'''# 三分类理论与属性结构收尾核查

本次仍合并使用360条记录，不按来源分组，不加入渠道变量，沿用理想分配情景。以下是看到结果后的解释性核查，原主检验、校正和留出结果均保持原样。

## 三分类信息公式

对8个三分类纤维，直接建立16个任意候选响应面参数和1个偏离参数的17×17 Fisher信息矩阵。对候选响应面作Schur补后，与 E[κ(Z)Var(h|Z)]逐项一致。分别检查1∶1及0.2至0.8不等的纤维内权重。h只依赖φ时，剩余信息为0。另在5个偏离幅度下直接计算最优候选响应面对应的KL距离，验证二阶项为η²I/2。所有四种情景均通过代数与数值核对。

这一核查覆盖本问卷的三分类形式，补充先前的二分类数值例子。它没有给连续支撑定理增加原创性，也不能将单次选择信息直接乘以2880作为重复记录实验的非中心参数。

## 实际只改变了一个原始属性方向

两版所有A、B服务的原始等级变化都为 Δ(D,S,I,E)=(1,−1,1,−1)。这些变化只张成一个方向。包含两个服务常数、四个数值属性、价格和两个共享项的9列设计矩阵秩为{geometry['raw_linear_design_rank']}。

在同质加性原始属性效用下，唯一可识别的版本组合是 τ=β_D−β_S+β_I−β_E。A、B都增加同一个τ，A对B的条件效用差保持不变。四个原始系数不能单独解释。分类或非线性属性模型可以给出不同A/B变化，但也无法仅凭本设计分离这些机制。

## 版本差异的解释性分解

将已经估计的两个版本服务常数τ_A、τ_B重新表达为平均服务偏移(τ_A+τ_B)/2，以及A对B相对对数赔率变化τ_A−τ_B；这只是现有模型参数的线性对比。

| 模型 | 方向 | 估计 | 聚类标准误 | 探索性模型条件p |
| --- | --- | ---: | ---: | ---: |
| MNL | 平均服务偏移 | {m[0]['estimate']:.5f} | {m[0]['cluster_se']:.5f} | {m[0]['p_model_conditional_exploratory']:.5f} |
| MNL | A对B相对变化 | {m[1]['estimate']:.5f} | {m[1]['cluster_se']:.5f} | {m[1]['p_model_conditional_exploratory']:.5f} |
| 随机服务截距 | 平均服务偏移 | {r[0]['estimate']:.5f} | {r[0]['cluster_se']:.5f} | {r[0]['p_model_conditional_exploratory']:.5f} |
| 随机服务截距 | A对B相对变化 | {r[1]['estimate']:.5f} | {r[1]['cluster_se']:.5f} | {r[1]['p_model_conditional_exploratory']:.5f} |

参数模型中的信号主要位于A对B的相对变化。该分解不能证明非线性原始属性效应，更不能把多个竞争解释变成已识别机制。这里的较小p值针对受限参数模型，是事后解释；不能替代整体充分性检验的p≈0.0487、精确边际多重检验的p≈0.1056或留出检验的p≈0.2022。

## 候选摘要的适用边界

对加权候选(w_D D+w_S S, w_I I+w_E E)，本次版本配对保持条件为 w_D=w_S 且 w_I=w_E。在w_S=w_E=1、另外两权重取0.8、1、1.2的9格审计中，只有等权情形保留8个二点纤维。其他权重对应16个单点纤维，对比维数为0。

更换权重就是更换候选摘要和原假设；这个结果没有推翻原等权摘要的证书。它说明证书严格依赖所声明的表示，不能拿本例证明任意合理效用压缩都有同样盲区。等权跨属性加和需要独立于本次结果的实质理由。

## 论点判断

技术层面已经覆盖全量记录、三分类理论、属性设计秩、完整菜单保持、主检验、精确边际校准、参数与面板模型、留出验证及情景功效。现有数据没有稳定确认核心边际充分性论点，也没有分离四属性作用或说明一个新检验优于经典对比。理论信息恒等式的正确性与方法的独立创新性需要分别判断。

模型可以在假设下处理全部记录。真人来源和实际分配不能从参数分解、积分精度或显著性推出；原始记录保持原样。
'''
    (out/'三分类理论与结构收尾核查.md').write_text(text)
    js(out/'closure_audit_verification.json',{'records_used':360,'source_subgroups_or_channel_covariates':False,'multinomial_scenarios_checked':4,'profiled_KL_amplitudes_per_scenario':5,'weighted_candidate_maps_checked':9,'original_menu_shift_rank':geometry['step_span_rank'],'additive_raw_design_rank':geometry['raw_linear_design_rank'],'decomposition_parameter_covariances_published':True,'raw_exports_unchanged':True,'primary_test_results_changed':False,'supports_confirmatory_core_claim':False})
    print('STRUCTURAL_CLOSURE_AUDIT_COMPLETE',json.dumps(rows),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1): main()
