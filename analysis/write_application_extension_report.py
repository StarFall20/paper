"""Report the fixed background extension and evidence against inflated claims."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out
    df=pd.read_csv(out/'background_contrasts.csv');money=pd.read_csv(out/'background_linear_fee_ratios.csv');fits=json.loads((out/'background_model_fits.json').read_text());external=json.loads((out/'public_healthcare_design_audit.json').read_text())
    assert len(df)==10 and len(money)==4
    # Validate reported contrasts and uncertainty against executed fits.
    maxerr=0.;maxse=0.;validation={}
    for r in df.itertuples():
        key='_'.join([r.population,r.modifier,r.form,r.kind]);f=fits[key];b=np.array(f['beta']);cov=np.array(f['covariance']);c=np.array(f['contrast_vector'])
        maxerr=max(maxerr,abs(c@b-r.difference_in_G2_G1_utility));maxse=max(maxse,abs(np.sqrt(c@cov@c)-r.cluster_se));assert f['checks']['gradient_max']<1e-3
        assert len(c)==len(b) and len(b)==len(cov)
        validation[key]={'parameters':len(b),'utility_design_rank':f['group_counts']['rank'],'gradient_max':f['checks']['gradient_max'],'finite_covariance':bool(np.isfinite(cov).all())}
    assert maxerr<1e-10 and maxse<1e-10
    validation['table_max_contrast_error']=maxerr;validation['table_max_cluster_se_error']=maxse;validation['private_record_data_exported']=False
    (out/'extension_numerical_checks.json').write_text(json.dumps(validation,indent=2)+'\n')
    selected=df[(df.modifier=='ever_paid')&(df.kind=='MNL')].copy()
    fig,ax=plt.subplots(figsize=(9,4));labs=[]
    for i,r in enumerate(selected.itertuples()):
        ax.errorbar(r.difference_in_G2_G1_utility,i,xerr=[[r.difference_in_G2_G1_utility-r.ci_low],[r.ci_high-r.difference_in_G2_G1_utility]],fmt='o',color='#2166ac' if r.form=='cells' else '#b2182b',capsize=4)
        labs.append(('All 360' if r.population=='all_360' else 'Other-source 300')+' / '+('Flexible cells' if r.form=='cells' else 'Linear fee'))
    ax.axvline(0,color='#777777',lw=1);ax.set_yticks(range(len(labs)),labs);ax.invert_yaxis();ax.set_xlabel('Ever-paid minus never-paid difference in G2-G1 utility');ax.set_title('Exploratory modifier contrast and record-clustered 95% intervals')
    fig.text(.02,.015,'Utility scales and functional forms are model dependent. Full 360-record analysis retained; 300-record source sensitivity.',fontsize=8);fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(out/'payment_modifier_sensitivity.svg');fig.savefig(out/'payment_modifier_sensitivity.png',dpi=170);plt.close(fig)
    rows=['| 样本与模型 | 付费经历的规则差异 | 95%区间 |','| --- | --- | --- |']
    for r in df[df.modifier=='ever_paid'].itertuples():rows.append(f'| {r.population} / {r.form} / {r.kind} | {r.difference_in_G2_G1_utility:.3f} | [{r.ci_low:.3f}, {r.ci_high:.3f}] |')
    report='''# 背景差异、独立文献与投稿条件复核

2026年11月9日。本轮在保留360条主分析的基础上，完成10个背景交互模型、整条答卷五折验证和一份独立公开DCE的输入预检。原始问卷、答卷和上一轮结果全部保留。当前目标仍是形成具备独立贡献的投稿稿；这轮证据没有证明已达到该条件。

## 背景题给出了什么

既有付费经历分为曾付费90条、未付费250条、不确定／不愿回答20条。每周使用分为121条、非每周225条、不愿回答14条。不确定回答保留并设置单独方案截距，不将其强制归入“没有经历”。全部360条模型调整原始来源组和版本／方案截距；其余300条作为来源敏感性样本。

预先固定本轮比较后，拟合线性费用与自由价格—规则单元模型，并为主要付费经历对比加入随机服务截距。核心对比是“有无付费经历，是否改变89元下G2相对G1的模型效用差”，不将经历解释为随机处理，也不把效用尺度差异当作纯偏好机制。

'''+ '\n'.join(rows)+'''

线性费用模型里，全部360条的差异为−0.395，探索性p=0.00444，在本轮10项对比内作Holm校正为0.0444。这个校正不覆盖整个项目之前的探索，也不提供确认性结论。放开费用函数后，差异为−0.197，区间[−1.099, 0.704]；加入随机服务截距后为−0.122，区间[−1.073, 0.829]。其余300条的自由单元估计接近零且区间宽。周使用频率同样没有给出稳定的规则差异。

线性模型的组间金额比值差异区间也都包含零；例如全360条曾付费与未付费的金额权衡差为−15.32元，区间[−32.40, 1.76]。这些是线性费用及其尺度假设下的比值，不能与自由费用模型下的效用差异相互替代。

所有比较沿用同一组按答卷划分的五折。其余300条中，加入付费／使用经历均没有改善联合预测分数；全360条的个别预测改善不足以归因于规则偏好差异，模型同时加入了购买倾向与不确定回答截距。结果完整保留在CSV，不挑选显著模型重写贡献。

## 更接近的文献

[Nittas等（2020）](https://www.jmir.org/2020/11/e18889/)的健康应用DCE已同时纳入费用、隐私控制、共享对象和提醒设置；其结果包含重复同意和共享给自选医疗提供者的偏好。已查阅摘要、属性表和结果。这比笼统的“隐私影响健康应用选择”更加接近当前问卷，必须纳入相关研究。

[Xie与Or（2023）](https://mhealth.jmir.org/2023/1/e25908/)还比较过既有使用及购买经历的分组偏好。因此，增加“曾付费人群”的分组模型本身不提供独立创新。当前贡献需要更明确、可识别、稳健的实质结论。这里是定向核查，未宣称完成穷尽性的Scopus或Google Scholar查重。

## 外部DCE预检结果

独立读取[BRAC大学公开的CHORUS Bangladesh DCE](https://zenodo.org/records/19478777)，原始CSV的MD5与发布页一致。目标是检验在另一份医疗DCE中将线性诊疗费放开为分类水平后，效用比较是否仍可识别；没有读取或拟合其选择结果。

当前文件中，5,184个服务方案行里有162行的六个方案属性全部缺失；36个声明的“区组×题号”位置各存在两个观察到的菜单版本。其原因和原始计划设计尚未由资料解释。这可能涉及整理或设计版本信息，不能据此判断原研究的结论。已停止设计秩和结果模型估计，没有用众数填补、删除缺失记录或拼出一个看似完整的设计。药品费用的“免费／半价／全价”也是类别，不能与诊疗费的货币金额相加。

这份数据目前不能承担独立验证，也没有被计作支持本文创新的实证案例。公开档案的可下载性不等于实验结构和编码已核验。

## 问卷星来源核对

你转述的平台真人招募说明已登记，主分析继续保留360条。读取问卷星时，平台当前明确提示“您还未登录，请先登录”，Chrome已停在登录页。此前缓存的答卷页不被当作本次已认证的服务说明。来源字段与具体服务的对应关系待重新登录后核对；不需要参与者身份信息。

## 投稿条件逐项判断

| 条件 | 当前证据 | 判断与下一步 |
| --- | --- | --- |
| 独立的新方法贡献 | 固定支撑的信息和秩已与经典lack-of-fit构造核对一致 | 原方法创新主张不成立，保留经典设计审计资源 |
| 有实质增量的应用问题 | 健康应用费用／控制与经历分组均有近邻研究 | 不能用这些标签或新增交互项代替贡献 |
| 原问卷能识别目标机制 | G2同时改变是否共享及共享对象；G0没有共同价格；任意题目效应会吸收部分目标 | 可估规则包的模型条件关联，不能隔离纯控制机制 |
| 结论对合理设定稳健 | 共享规则、付费经历结论都依赖价格／参与设定 | 尚无足以承担主贡献的稳定结论 |
| 数值与复现一致 | 10项对比回查系数和协方差、梯度、五折分组与公开输入MD5 | 本轮技术检查已完成；它们不等于创新或外部有效性 |
| 来源与研究陈述可核对 | 真人招募为研究者转述；平台重新认证尚未完成；实际适用的研究说明仍待作者整理 | 保留原字段，待登录后核对服务说明，按真实材料填写 |
| 独立验证 | 新公开DCE未通过输入完整性预检，旧公开SP资源已完成经典对照 | 不把问题输入当作新增验证成功 |

下一步继续核对来源，并评估能否围绕“价格支撑是否足以支持规则比较”找到资料完整的独立设计。只有取得区别于已有研究的实质证据，才重构正式投稿稿。现有问卷继续作为已完成的探索性实证材料；不追加相同题目来替代识别与贡献问题。
'''
    (out/'背景扩展与投稿条件复核.md').write_text(report)
    (out/'application_background_appendix.md').write_text('''# Appendix: Background-modifier sensitivity

This appendix reports an exploratory extension fixed after earlier outcomes and descriptive counts had been inspected. It is not preregistered confirmation. Ever-paid status and weekly application use are self-reported modifiers. Unknown responses retain separate alternative intercepts. Main models retain all 360 records and adjust original source-group and version/package intercepts; the other 300 records provide a source sensitivity.

The pooled linear-fee model estimates an ever-paid modifier of −0.395 for the G2–G1 utility contrast, 95% interval −0.668 to −0.123. Its exploratory p-value is 0.00444 and Holm adjustment within the ten extension tests is 0.0444. Flexible fee-rule cells give −0.197, interval −1.099 to 0.704; a random service intercept gives −0.122, interval −1.073 to 0.829. In the other-source 300 records, the corresponding flexible estimates are 0.005 and 0.008, with intervals spanning approximately −1.07 to 1.08. Utility scales and ratio assumptions differ across models. The linear-fee money-ratio difference also has an interval spanning zero.

Modifier models provide no predictive improvement in the 300-record sensitivity population on the same whole-record five folds. Some pooled improvements can involve general participation or unknown-response intercepts; they do not identify the contribution of a sharing-rule preference interaction. These results are retained as sensitivity evidence and are not used to name a causal mechanism or an independent innovation.

The related-work comparison includes [Nittas et al. (2020)](https://www.jmir.org/2020/11/e18889/), which studies fee, privacy control, sharing and reminders, and [Xie and Or (2023)](https://mhealth.jmir.org/2023/1/e25908/), which examines prior purchase/use experience. Neither a control-related attribute nor an experience subgroup establishes the contribution by itself.
''')
    (out/'README.md').write_text('''# Application extension and readiness reassessment — 2026-11-09

The fixed exploratory plan, all ten modifier tests, grouped validation diagnostics, linear-fee ratios, aggregate fitted parameters/covariances and a failed external-data preflight are retained. No independent innovation or submission readiness is claimed. Main analyses retain all 360 records; original source labels and the researcher's reported human-recruitment statement remain unchanged.

From the repository root, with numpy, pandas, scipy, matplotlib, openpyxl and threadpoolctl:

```sh
python analysis/application_background_extensions.py --input-dir "$PRIVATE_DIR" --instrument survey/public_dh36/instrument.json --folds "$PRIVATE_DIR/ideal_cv_folds_private.csv" --out reports/application_extensions_20261109
python analysis/public_healthcare_price_support_audit.py --input "$CHORUS_CSV" --out reports/application_extensions_20261109
python analysis/write_application_extension_report.py --out reports/application_extensions_20261109
```

`PRIVATE_DIR` contains the original two exports and private fold file. Its historical “ideal” filename does not establish random assignment. `CHORUS_CSV` is the unmodified CSV released at https://zenodo.org/records/19478777, MD5 0804176ec8f3114fd220da90966d8f26. The public audit records a preflight requiring clarification and deliberately fits no outcome model. The source file is not redistributed; the license and intended design must be clarified before any further release or inference. No emails or requests to the authors have been sent.

Whole-record folds are reused internally and do not constitute independent external validation. Holm adjustment is limited to the ten stated extension contrasts, not the entire exploratory project. Numerical summaries are checked against saved coefficient and covariance arrays. Raw answers, personal metadata, private record identifiers, individual predictions and external respondent rows are excluded from the public backup.
''')
    print('Extension report, appendix, figure and numerical checks written.')

if __name__=='__main__':main()
