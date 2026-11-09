"""Current reports after the task-specific AI-pretest service clarification."""
from pathlib import Path
import argparse,json,re
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from application_support_models import save


def table(d):
    rows=['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']
    for row in d.itertuples(index=False,name=None):rows.append('| '+' | '.join(f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(rows)


def prose(text):
    text=re.sub(r'\b([A-Za-z]{3,}|to|of|at|in|by|or|AI)(?=\d)',r'\1 ',text).replace("Cerina's2026", "Cerina's 2026")
    return re.sub(r',(?=\S)',', ',text)


def main():
    p=argparse.ArgumentParser();p.add_argument('--primary-out',type=Path,required=True);p.add_argument('--pairing-out',type=Path,required=True);p.add_argument('--classification',type=Path,required=True);a=p.parse_args()
    role=json.loads(a.classification.read_text());primary=a.primary_out;pair=a.pairing_out
    models=pd.read_csv(primary/'primary_model_comparison.csv');contrasts=pd.read_csv(primary/'same_price_rule_contrasts.csv');scores=pd.read_csv(pair/'human_target_prediction.csv');params=pd.read_csv(pair/'fold_model_parameters.csv')
    checks=json.loads((pair/'numerical_checks.json').read_text());assert checks['all_50_training_marginal_controls_passed'] and role['primary_records']['total']==300
    within_path=primary.parent/'within_person_audit_20261109/conditional_fixed_effect_contrasts.csv'
    within_cells=None
    if within_path.exists():
        within_cells=pd.read_csv(within_path).set_index('model').loc['observed_cells']
    summary=scores.groupby(['condition','model'],as_index=False).agg(mean_human_joint_nll=('human_joint_nll_per_choice','mean'),minimum=('human_joint_nll_per_choice','min'),maximum=('human_joint_nll_per_choice','max'))
    summary.to_csv(pair/'condition_summary.csv',index=False)
    pretty=summary.replace({'condition':{'human_only':'Human training only','mixed_original_AI':'Original AI pretests added','mixed_repaired_AI':'AI answers re-paired'},'model':{'MNL_linear':'MNL','RI_linear':'Random service intercept','spike_linear':'Outside component'}}).rename(columns={'condition':'Training condition','model':'Model','mean_human_joint_nll':'Mean human joint NLL','minimum':'Minimum','maximum':'Maximum'})
    parameter_summary=params.groupby(['condition','model'],as_index=False)[['service_intercept_sd','outside_component_weight','training_AI_constant_vectors']].mean()
    parameter_summary.to_csv(pair/'parameter_summary.csv',index=False)
    # Source-provided service wording is decisive; answer patterns are not a detector.
    (primary/'service_definition_user_provided.txt').write_text(role['service_definition_original']+'\n')
    report=f'''# 样本来源澄清后的真人主分析

2026年11月9日。用户提供了对应服务的明确说明：平台让AI样本作答问卷，用途是预研究、预测试。结合此前已核实的两项各30份任务，原“AI样本”60条确认为AI预测试作答。此前转述的“真人招募”说明不再适用于这60条。

## 当前样本口径

| 数据 | 人数／记录数 | 选择次数 | 当前用途 |
| --- | --- | --- | --- |
| 普通提交来源、按招募记录列为真人的两版问卷 | 各150，共300 | 2,400 | 真人偏好主分析 |
| 平台AI预测试作答 | 各30，共60 | 480 | 独立的AI预测试与混合数据诊断 |
| 原始合并导出 | 360条 | 2,880 | 保存原始记录；不能写成360名真人 |

300条的划分依据是来源说明和提交元数据，不是选择结果或显著性。它不自动证明人口代表性、成年资格或每条记录均有效。两版由不同招募渠道分别发放，未保留分配日志，不能识别版本因果效应。原始360条、来源字段及文件哈希均未改动。

## 已重跑的主模型

{table(models[['model','human_records','choice_rows','joint_nll_per_choice','marginal_nll_per_choice']])}

沿用整条答卷五折，全部测试对象均来自这300条真人来源记录。普通线性价格模型的联合负对数分数为1.031745；随机服务截距模型为1.031899。增加随机截距或自由价格单元未显示留出预测改善。这些折曾用于探索，不能称为新的独立验证。

## 同价共享规则包的比较

{table(contrasts[['population','model','utility_difference','ci_low','ci_high']])}

自由价格单元普通模型估计89元时G2−G1为0.503，95%记录聚类区间[0.029,0.977]；随机截距模型为0.499，[0.024,0.975]。条件A/B模型为0.787，[-0.184,1.759]。这些是模型条件下的探索性区间，不构成选择后或项目范围的多重检验保证。

G2包含“由用户决定是否共享、共享给谁”的组合描述，G1为去标识后向科研机构共享。没有一道题直接把两项89元方案同时对置；对比依赖跨题效用及固定方案截距假设。不能拆成纯粹的控制权机制，也不能由此宣布确定的金额溢价。

此前标记的一条较早提交记录仍保留在300条主分析中。另报移去该条后的299条敏感性，自由单元对比为0.508，[0.030,0.987]；这不是对其无效性的认定。普通模型结果的方向稳定，条件A/B的较宽不确定性仍需同时保留。

## 稿件如何使用这些结果

真人应用稿采用300条。历史360条估计改列为混合来源诊断，不能继续用其类别比例描述消费者类型。AI60条可用于已明确标注的预测试／计算审计，不计入真人样本量、偏好系数或真人显著性证据。

现有问卷内容仍可利用。原创性不能由样本分类纠正或某个p值代替：健康应用的费用、共享和隐私属性已有接近文献，新的应用主张仍需实质增量。本轮进一步完成的跨题配对对照见[AI预测试计算审计](../synthetic_pairing_audit_20261109/AI预测试配对审计.md)。
'''
    if within_cells is not None:
        extra=f'''## 新增个体内稳健性审计

允许每个人有不同的稳定A／B方案截距，用各自的选项次数消去这些截距。自由费用—共享规则单元下，89元时G2−G1为{within_cells['estimate']:.6f}，95%记录层面区间[{within_cells['ci_low']:.6f}, {within_cells['ci_high']:.6f}]。共同属性斜率和条件logit假设仍然保留。56项参与关联检验经Holm校正后没有拒绝，不能据此宣称模型充分。详见[个体内重建审计](../within_person_audit_20261109/个体内模型重建与创新边界.md)。

'''
        report=report.replace('## 稿件如何使用这些结果',extra+'## 稿件如何使用这些结果')
    (primary/'真人主分析与来源纠正.md').write_text(report)
    human_note='''# Data-sharing packages in a paid health-service questionnaire

*Exploratory application revision, 9 November 2026. Human primary population corrected after the supplied AI-pretest service definition.*

## Abstract

This analysis examines sharing-rule packages in 300 human-recruitment questionnaire records, each containing eight health-service choices. Sixty additional platform AI-pretest records are excluded from human estimation and retained for separate diagnostics. The observed fee support connects a user-decision sharing package and deidentified research sharing at CNY89. A flexible-cell MNL estimates their utility difference at 0.503 (record-clustered 95% interval 0.029–0.977); a random service-intercept model gives 0.499 (0.024–0.975). Conditional A/B estimation gives 0.787 (−0.184–1.759). A linear-fee ordinary model has the best internal joint prediction among the four fixed primary specifications. The package association depends on cross-task utility restrictions and does not isolate a pure control mechanism or a monetary premium.

## Questionnaire and records

Each version contributes 150 human-recruitment records. Monthly fees range from CNY59 to CNY119. Sharing descriptions cover no external sharing (G0), deidentified research sharing (G1), and user decisions about whether and with whom to share (G2). Service-card attributes fixed within version and alternative are represented by version-specific package intercepts. Their separate effects cannot be recovered.

Different recruitment channels administered one version each, without a retained allocation log. Version adjustments describe channel/version associations. Ordinary submission metadata and the researcher recruitment report define the 300-record analytic set; demographic representativeness and eligibility were not independently measured. The supplied platform wording explicitly says that AI samples answer the questionnaire for preliminary research/pretesting. The 60 task-linked records are synthetic pretests, not additional human respondents. All 360 original records remain archived.

## Estimation

The four fixed specifications use linear fees or unrestricted observed fee-rule cells, with or without a normal random service intercept. Each person's entire eight-answer vector enters the panel likelihood and stays in one of the existing five folds. Predictions integrate the population distribution without held-out class posteriors. These reused folds provide exploratory internal validation.

G0 occurs only at fees79 and109, G1 at69,89,99, and G2 at59,89,119. G1 and G2 share CNY89; G0 shares no fee with either. The cell representation has rank11. The CNY89 contrast relies on cross-task utility restrictions; removing task2 or5 removes its identification. It is not a direct same-fee randomized comparison. Conditional A/B estimates and the stated early-record sensitivity accompany the primary models.

## Results and interpretation

Ordinary linear-fee logit has a held-out joint negative log score per choice of1.031745. Ordinary observed cells, random-intercept linear fees and random-intercept cells give1.032050,1.031899 and1.032229. More flexible specifications show no predictive improvement for the current human target.

The positive same-fee package association has intervals above zero in ordinary and random-intercept cells, while conditional A/B uncertainty spans zero. Excluding the flagged early record gives0.508 (0.030–0.987) in ordinary cells. That record remains in the primary analysis; no invalidity is inferred from its timing alone. These model-conditional exploratory results do not establish an isolated decision-right mechanism or population WTP.

App fees, privacy, sharing and usage-history heterogeneity have close prior applications, including [Nittas et al. (2020)](https://www.jmir.org/2020/11/e18889/) and [Xie and Or (2023)](https://mhealth.jmir.org/2023/1/e25908/). The current positive package estimate is retained as an application result. It does not by itself establish an independently innovative JOCM submission. Applicable recruitment, consent and research statements must follow the actual materials, with no invented approval.
'''
    if within_cells is not None:
        human_note=human_note.replace('Conditional A/B estimation gives 0.787 (−0.184–1.759).',f"Conditional A/B estimation gives 0.787 (−0.184–1.759). Removing stable person-specific alternative intercepts through classical count conditioning gives {within_cells['estimate']:.3f} ({within_cells['ci_low']:.3f}–{within_cells['ci_high']:.3f}) under the same free cells.")
        methods='''A further sensitivity uses the established fixed-effect conditional multinomial logit. Conditioning on each person's A/B/outside counts removes stable alternative intercepts, while retaining common attribute slopes and independent conditional logit errors. See [Pforr (2014), Section2](https://doi.org/10.1177/1536867X1401400409). Exact enumeration covers all6561 sequences. Conditional likelihood does not evaluate unconditional participation or population prediction.

'''
        human_note=human_note.replace('## Results and interpretation',methods+'## Results and interpretation')
        results=f"The fixed-effect free-cell sensitivity gives {within_cells['estimate']:.3f} [{within_cells['ci_low']:.3f}, {within_cells['ci_high']:.3f}] for the CNY89 package contrast. It is close to the pooled and normal-intercept results. This robustness concerns stable intercept heterogeneity; dynamic task effects, person-specific error scales and random attribute slopes remain outside this sensitivity. The count-conditioned likelihood is a different target from the earlier unconditional held-out scores.\n\n"
        human_note=human_note.replace('App fees, privacy, sharing and usage-history heterogeneity',results+'App fees, privacy, sharing and usage-history heterogeneity')
        human_note=human_note.replace('The current positive package estimate is retained as an application result.', '[Biasiotto et al. (2023)](https://www.jmir.org/2023/1/e47066/) also studies health-data consent and governance preferences. The current positive package estimate is retained as an application result.')
    (primary/'human_application_draft.md').write_text(prose(human_note))
    negative_path=pair/'human_negative_control_validated_summary.csv'
    negative_section=''
    negative_abstract=''
    originality_status='Independent originality and submission readiness remain under assessment.'
    if negative_path.exists():
        negative=pd.read_csv(negative_path)
        high=json.loads((pair/'human_rank_alignment_quadrature_audit.json').read_text())
        assert high['all_five_boundary_stress_fits_validated']
        negative_pretty=negative.groupby(['condition','model'],as_index=False).agg(mean=('joint_nll_per_choice','mean'),min=('joint_nll_per_choice','min'),max=('joint_nll_per_choice','max'))
        negative_pretty=negative_pretty.rename(columns={'condition':'Human training control','model':'Model','mean':'Mean human joint NLL','min':'Minimum','max':'Maximum'}).replace({'Human training control':{'human_original':'Original human vectors','human_random_repaired':'Randomly re-paired human answers','human_rank_aligned':'Rank-aligned human answers'},'Model':{'RI':'Random service intercept','spike':'Outside component'}})
        negative_abstract=' Human-only fixed-margin controls reproduce the general sensitivity: deliberately rank-aligning human training answers worsens the same two panel scores to 1.220683 and 1.253002. The rank-aligned random-intercept fit reaches its prespecified upper bound. '
        originality_status='Human negative controls and the closest literature do not support an independently new method. This case remains a research note; the JOCM submission objective is unachieved.'
        negative_section=f'''## 5. Human negative controls

We repeat the operation using only the 300 human-recruitment records. Original human test vectors remain fixed. Within each version and training fold, we either independently permute each task column using the same ten fixed seeds, or sort each task's alternatives into a common row order. Rank alignment is a deliberately extreme dependence stress control. All 55 additional fixed-margin controls preserve MNL's objective and gradient, within numerical precision.

{table(negative_pretty)}

Random human re-pairings give random-intercept scores from 1.030981 to 1.031772; the outside-component estimate remains at its near-zero lower bound. Rank alignment yields an outside-component mean weight of 0.236533 and human score 1.253002. Every rank-aligned random-intercept fit reaches the fixed upper log-standard-deviation bound of 2 (standard deviation 7.389056); its score is 1.220683. The boundary limits the interpretation of its magnitude. These constructed controls demonstrate generic dependence sensitivity and do not identify an AI-specific mechanism. Different source margins prevent the smaller random-human effect from establishing AI specificity.

The initial 61-node quadrature was inadequate for the extreme rank-aligned random-intercept fits. Re-estimation at 1281 stable Gaussian nodes, checked against 2561, limits the maximum training log-likelihood discrepancy to 7.3e-7 and held-out discrepancy to 4.6e-7. Original 61-node stress results remain an audit trail; the validated table uses the higher-order fits. The human primary fits and original AI-pairing results are unchanged.

'''
    case_note=f'''# AI pretest answer pairing and inferred participation heterogeneity: a health-service DCE case audit

*Exploratory case record, 9 November 2026. {originality_status}*

## Abstract

Synthetic pretest responses may enter a panel choice dataset with dependence patterns that differ from human submissions. We audit300 human-recruitment records and60 platform AI-pretest records, each containing eight health-service choices. All AI vectors select one option throughout; none of the human vectors do. Adding the original AI batch worsens human-target joint negative log scores from1.031899 to1.064631 for a random service-intercept model and from1.031745 to1.049832 for an outside-component model. A computational control re-pairs only training AI answers within version and task, preserving every task's choice counts. Ordinary MNL estimates and human predictions remain invariant. Across ten fixed re-pairings, human-target scores average1.044161 and1.042139 for the two panel models; the fitted outside weight falls from11.66% to1.14%.{negative_abstract}The empirical result concerns sensitivity in one pretest deployment.

## 1. Question

Panel choice models use the joint sequence of an individual's answers. AI pretests provide complete answer vectors whose joint structure can differ from human submissions. The question here is what changes in panel estimation when synthetic answer pairing changes and every observed task margin remains fixed.

Recent DCE work already compares synthetic and human choices, including utility-scale differences and pretest design uses. [Califano and Caracciolo (2026)](https://doi.org/10.1016/j.foodqual.2026.105930) supplies a close comparison across three food DCEs. [Barrie and Cerina's2026 working paper](https://csmapnyu.org/research/academic-research) also documents dependence differences in synthetic survey belief systems. Non-trading rules are established in [Kazagli and de Lapparent (2023)](https://doi.org/10.1016/j.jocm.2023.100413).

[Doudkin (2026), Appendix A.8](https://arxiv.org/html/2609.07305v1), already shuffles synthetic probability columns across personas before realizing multiselect responses. Our control fixes realized DCE counts exactly and measures downstream panel estimation. This is a case-specific extension of a close control idea. The dependence itself follows the sequence likelihood in [Train (2009), Section 6.7](https://eml.berkeley.edu/books/choice2nd/Ch06_p134-150.pdf).

## 2. Data and source definition

Two versions contribute150 human-recruitment submissions each. The platform additionally produced30 AI-pretest records per version. The researcher supplied the service wording explicitly describing AI samples answering questionnaires for preliminary research/pretesting. Prior authenticated inspection verified the two task counts, original source labels and AI-point records. The wording itself is researcher-provided. The AI model, prompting, persona-selection and decoding configuration are unavailable; no particular LLM is assumed.

The publication archive contains rendered service cards and records eight uploaded cards per version. The generator's representation of these cards is unavailable. Input access, parsing, profile construction and output handling remain possible explanations for the constant vectors; this audit does not attribute them to an AI preference mechanism.

The AI batch contains15 all-A, three all-B and42 all-outside vectors. All300 ordinary-submission vectors change option at least once. The source definition identifies the AI records; constant choices are not used as an origin detector. Recruitment channels differ by version, so source contrasts do not identify population-level causal differences between matched human and AI respondents.

The platform describes a pretest use. The mixed-data conditions below deliberately examine the computational consequence of pooling those records for estimation; they are not a claim that the service recommends doing so. Human preference estimation uses only the300 human-recruitment records.

## 3. Fixed computational comparison

We compare ordinary MNL, a normal random service intercept and a classical outside-component mixture, using the existing version-adjusted linear-fee utility specification. In each of the five existing record folds, menus, human training vectors and human held-out vectors remain fixed. Conditions use human training records alone, the original training AI batch, or re-paired training AI answers. The human target contains300 original held-out records across folds.

Re-pairing independently permutes each AI task column within version and training fold. It preserves all task-by-version choice counts, sample size and source composition. Ten seeds are fixed in the exploratory plan. These controls are generated arrangements, not newly collected responses or corrected human data.

For the specified MNL, AI feature vectors are identical within version and task. Its log likelihood is a sum over those task/alternative choice counts. Preserving those cells preserves its objective and gradient, coefficient point estimates and human predictions. Record-clustered uncertainty is not claimed to remain invariant. This established invariance is an experimental control. Panel likelihoods depend on the pairing of answers and need not remain invariant. All50 training controls pass, with maximum objective difference{checks['maximum_MNL_objective_difference']:.2g} and gradient difference{checks['maximum_MNL_gradient_difference']:.2g}.

## 4. Results

{table(pretty)}

With the original mixed batch, random-intercept and outside-component models predict human vectors less accurately than the same models trained on human records alone. The mixed validation target gives a different ranking: its joint scores are1.040306 for MNL,0.981693 for the random intercept and0.956676 for the outside component. Evaluating only original human held-out records gives1.042676,1.064631 and1.049832. A score averaged over human and AI records answers a different question from human prediction.

Re-pairing changes none of MNL's task counts or human predictions. It reduces the random-intercept human-target loss to a mean1.044161 (range1.043981–1.044377), and the outside-component loss to1.042139 (1.041655–1.042524). Mean fitted outside weight across training folds falls from0.116622 with original AI pairing to0.011382 with re-pairing; mean random-intercept standard deviation falls from1.355882 to0.722955. Human-only fits have means near zero for the outside weight and0.221594 for the random-intercept deviation. These are fitted summaries, not changes in real consumers' tastes.

The remaining human loss after re-pairing shows that changing dependence does not make the AI margins a suitable human data source. The computational comparison isolates one sensitivity of panel estimates; it does not authorize pooling repaired AI records into human inference.

{negative_section}## 6. Interpretation and scope

This case distinguishes task margins, answer pairing and the prediction population. The fitted heterogeneity from mixed records partly represents the construction of the synthetic pretest batch. A corresponding improvement on a mixed validation target does not establish improved human preference recovery.

The case quantifies a downstream consequence for this questionnaire. Known panel-likelihood dependence, the close synthetic shuffle precedent and the human negative controls remove the basis for presenting the operation as an independent methodological innovation. Its present role is an exploratory case record alongside the corrected human application.

Ten re-pairings measure procedural sensitivity. The questionnaire, service and two batches are fixed. Unreported generator settings, convenience recruitment and reused folds limit transfer. The computational controls support neither a permutation p-value nor an inference across ten independent AI deployments. A submission contribution still needs a substantive choice-research finding beyond these generic sensitivities.

## Reproducibility

The package contains source classification, the fixed plan, aggregate fold scores and parameter summaries, and exact marginal-control checks. Original answer vectors, identifiers, source-detail names, timestamps, IPs and derived individual controls remain private. Re-estimation requires authorized access to the two frozen exports and existing record folds. All300 human submissions remain in the primary sample; an independently flagged early-record sensitivity is reported separately.
'''
    (pair/'pairing_case_manuscript.md').write_text(prose(case_note))
    chinese=f'''# AI预测试答卷的跨题配对审计

2026年11月9日。来源说明已确认60条为AI预测试作答。当前真人主分析为300条，原始360条保留。这里的混合估计是明确标注的计算审计，不计为360名真人。

## 当前可核验的发现

在相同真人训练集、真人留出集、菜单和AI数量下，只改变AI答案在不同题目之间的配对，面板模型的估计与预测就会变化。每题、每版、每个训练折的选项人数完全不变，普通MNL的目标函数与梯度也不变；50组检查全部通过。

| 模型 | 只用真人训练 | 加入原AI预测试 | 加入重新配对AI，10次均值 |
| --- | --- | --- | --- |
| 普通MNL | 1.031745 | 1.042676 | 1.042676 |
| 随机服务截距 | 1.031899 | 1.064631 | 1.044161 |
| 全程不购买成分 | 1.031745 | 1.049832 | 1.042139 |

表中均为原始真人留出答卷的联合负对数分数，越小越好。重新配对只发生在训练AI答卷中；没有修改真人记录，也没有用真人测试答案拟合模型。

原AI配对下，不购买成分的训练折平均权重为11.66%；重新配对后约1.14%。随机截距标准差从1.356降为0.723。真人行为没有发生变化；这些变化来自向模型输入的数据结构。不能把原360条中的类别权重写成消费者类型比例。

## 这个对照排除了什么解释

配对前后样本量、每题选项人数、版本组成和费用—共享规则菜单均相同。MNL的人类留出预测精确不变。这给“跨题组合影响面板模型”的解释提供了直接计算证据。重新配对并未让AI数据变成真人数据；其余误差仍在，修复AI答卷后用于真人估计不是本研究的建议。

## 创新性复审前的候选定位

目前更具体的研究候选是：**AI预测试的答案配对结构，能够在边际选择计数不变时改变面板异质性与模型排名，混合来源验证会掩盖对真人预测的损失。**问题和对照来自同一份数据中的原生矛盾，未将旧充分性框架重新拼装为理论。

这是一项有明确对照的案例发现。普通MNL不变性、非交易成分和随机截距均为已有工具。2026年已有GPT与真人DCE比较、合成问卷关联结构研究，故不能仅凭“AI+选择模型”宣布创新；当前候选的具体增量是保持每题计数不变的配对干预及对真人预测目标的影响。独立原创性和JOCM投稿门槛仍在审查。

10次固定配对是计算敏感性分析，不是10批独立AI样本、功效分析或置换p值。AI模型、提示词和画像选取不可得；结论限定于这一服务部署与问卷。平台明确定位预测试，本文不声称其承诺这些答卷可用于真人总体推断。

问卷发布档案还包含逐题渲染的服务卡，记录每版上传8张卡片。AI服务实际获得了哪些卡片内容、以什么形式获得，尚无记录。固定回答也可能涉及内容读取或解析等流程问题，不能直接归因为AI的偏好机制；这不影响已经标明的配对计算对照，但限制了它的技术解释。

原始文件未变，个人信息和派生逐条答卷不进入公开包。英文稿见`pairing_case_manuscript.md`；真人应用稿与样本口径见[真人主分析](../human_primary_20261109/真人主分析与来源纠正.md)。
'''
    if negative_path.exists():
        prefix=chinese.split('## 创新性复审前的候选定位')[0]
        chinese=prefix+'''## 真人负对照与创新性复审

仅用300条真人来源记录，保持每题、每版、每折的选项人数不变，再做10次随机重配及一次按选项排序的极端对齐。真人留出答卷始终保持原状。新增55组边际不变检查全部通过。

| 真人训练条件 | 普通MNL | 随机服务截距 | 全程不购买成分 |
| --- | --- | --- | --- |
| 原始真人答卷 | 1.031745 | 1.031899 | 1.031745 |
| 随机重配，10次均值 | 1.031745 | 1.031470 | 1.031745 |
| 按选项极端对齐 | 1.031745 | 1.220683 | 1.253002 |

极端对齐是人为构造的压力对照。它使不购买成分的平均权重升至23.65%。随机截距的5折估计全部触及预定上界，标准差7.389；这不是无约束总体异质性估计。其初始61节点积分存在误差，已用1281节点重估并以2561节点核验，训练总对数似然最大差异小于7.3×10⁻⁷。表中采用核验后的分数，原始低精度结果另存为排查记录。

真人数据也能出现这种现象，说明面板模型对跨题配对的敏感性具有一般性。真人随机重配的影响较小，也不能证明AI专属性，因为两类来源的边际和构造不同。

最近文献进一步收紧了增量：[Doudkin（2026）附录A.8](https://arxiv.org/html/2609.07305v1)已有保持概率边际的个体重排；本案固定的是已实现的DCE选择计数，考察的是后续面板估计。两者保留差别，但重排思路本身已有接近先例。[Train第6.7节](https://eml.berkeley.edu/books/choice2nd/Ch06_p134-150.pdf)给出了面板序列似然的经典来源。

**当前决定：保留该案例审计，撤回把配对操作作为独立新方法的定位；尚未达到用户要求的独立创新与投稿完成条件。**不再通过给已知敏感性命名、增加模型或挑选结果来包装贡献。详细比对见[创新性复审](originality_review_20261109.md)。

问卷发布档案记录每版上传8张服务卡。AI服务获得这些卡片的具体方式仍无记录，固定回答的流程原因尚未识别。平台明确将AI样本定位为预测试，本案审计也限于这一部署。10次配对均为计算敏感性安排。

原始360条记录未变。真人主分析继续使用300条；60条AI预测试独立保存。英文案例记录见`pairing_case_manuscript.md`；真人主结果见[来源纠正与真人主分析](../human_primary_20261109/真人主分析与来源纠正.md)。
'''
    (pair/'AI预测试配对审计.md').write_text(chinese)
    for folder,kind in [(primary,'human primary estimation'),(pair,'fixed synthetic-pretest computational control')]:
        (folder/'README.md').write_text(f'''# Current {kind} — 9 November 2026

Source roles follow `data/dh36_source_classification.json`:300 human-recruitment
records for human preference estimation;60 AI-pretest records for separate
diagnostics. The360-row export is never described as360 human respondents.

Run from repository root with numpy, pandas, scipy, openpyxl, matplotlib and
threadpoolctl. Raw exports and existing folds remain private.

```sh
python analysis/dh36_human_and_pairing_audit.py --input-dir "$PRIVATE_DIR" --instrument survey/public_dh36/instrument.json --folds "$PRIVATE_DIR/ideal_cv_folds_private.csv" --classification data/dh36_source_classification.json --primary-out reports/human_primary_20261109 --pairing-out reports/synthetic_pairing_audit_20261109
python analysis/pairing_negative_controls.py --input-dir "$PRIVATE_DIR" --instrument survey/public_dh36/instrument.json --folds "$PRIVATE_DIR/ideal_cv_folds_private.csv" --out reports/synthetic_pairing_audit_20261109
python analysis/pairing_quadrature_audit.py --input-dir "$PRIVATE_DIR" --instrument survey/public_dh36/instrument.json --folds "$PRIVATE_DIR/ideal_cv_folds_private.csv" --out reports/synthetic_pairing_audit_20261109
python analysis/write_human_ai_revision_report.py --primary-out reports/human_primary_20261109 --pairing-out reports/synthetic_pairing_audit_20261109 --classification data/dh36_source_classification.json
```

The historical fold filename does not establish random assignment. Validation
uses reused internal whole-record folds. Re-pairings are labelled controls,
not observations or a proposal to replace humans. Their ranges are procedural
sensitivity, not confidence intervals or formal permutation tests. No new
AI-generation task, sample purchase or human fielding was performed.

Human negative controls use the independently fixed plan in
`reports/synthetic_pairing_audit_20261109/negative_control_plan.json`.
All negative-control output filenames below refer to that same result directory.
For rank-aligned RI stress results, use `human_negative_control_validated_summary.csv`
and `human_rank_alignment_quadrature_audit.json`. The initial 61-node stress
outputs are retained as a numerical audit trail and are superseded for those five
boundary fits. The rank-aligned standard deviation reaches its fixed upper bound;
its magnitude is a constrained stress result. Primary human and AI fits are unchanged.

The updated originality decision is in
`reports/synthetic_pairing_audit_20261109/originality_review_20261109.md`.
This case is not currently presented as an independent new method or a completed
JOCM submission.
''')
    plt.rcParams.update({'svg.fonttype':'none','font.size':10})
    fig,ax=plt.subplots(figsize=(8.3,4.2));labels=['MNL','Random service intercept','Outside component'];keys=['MNL_linear','RI_linear','spike_linear'];pos=np.arange(3)
    for k,condition in enumerate(['human_only','mixed_original_AI','mixed_repaired_AI']):
        subset=summary[summary.condition==condition].set_index('model').loc[keys]
        means=subset.mean_human_joint_nll.to_numpy();err=np.stack([means-subset.minimum.to_numpy(),subset.maximum.to_numpy()-means])
        ax.errorbar(pos+(k-1)*.16,means,yerr=err,fmt='o',capsize=4,label=['Human training only','Original AI pretests added','Re-paired AI: range over 10 controls'][k])
    ax.set_xticks(pos,labels);ax.set_ylabel('Held-out human joint NLL per choice');ax.set_title('The same human target; controlled AI answer pairing');ax.legend(fontsize=8);ax.set_ylim(1.025,1.073);fig.tight_layout();fig.savefig(pair/'human_target_pairing_control.svg');fig.savefig(pair/'human_target_pairing_control.png',dpi=180);plt.close(fig)
    print('Current human-primary and AI-pretest case reports written.',flush=True)


if __name__=='__main__':main()
