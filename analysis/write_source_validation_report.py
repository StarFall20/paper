"""Write a bounded research note from completed source-conditioned benchmarks."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from application_support_models import save, build
from source_conditioned_validation import enumerate_check
import dh36_observational_analysis as old


def table(frame):
    lines = ['| '+' | '.join(frame.columns)+' |', '| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False, name=None):
        lines.append('| '+' | '.join(f'{v:.6f}' if isinstance(v, (float, np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser()
    for name in ['input-dir','instrument','out']:
        p.add_argument('--'+name, type=Path, required=True)
    a=p.parse_args(); out=a.out
    scores=pd.read_csv(out/'source_stratified_prediction.csv')
    gains=pd.read_csv(out/'paired_descriptive_prediction_gains.csv')
    fits=json.loads((out/'model_fits.json').read_text())
    folds=json.loads((out/'fold_fit_checks.json').read_text())
    checks=json.loads((out/'numerical_checks.json').read_text())
    spec=json.loads(a.instrument.read_text()); d,provenance,*_=old.read_exports(a.input_dir,spec)
    forms,*_=build(spec,d)
    frozen={'V1':'dbe667ecc7511b69fcb5c36147f986499dbbbf23ec1d7f638eb137cfbb080e13','V2':'40b3cc7daee37b5ad0e1060b77c12cdccc9c5dff9cff1a57a75e32b8d5b7ce01'}
    assert {v['version']:v['sha256'] for v in provenance} == frozen
    checks['raw_hashes_match_frozen_exports']=True
    checks.pop('prediction_file_exported',None)
    checks['public_record_prediction_file_exported']=False
    checks['private_record_predictions_saved']=True
    for label in ['hurdle_blind_linear','hurdle_source_linear','hurdle_blind_cells','hurdle_source_cells']:
        x=forms['cells' if label.endswith('cells') else 'linear']; b=np.array(fits[label]['regular_component']['beta']); w=np.array(fits[label]['gate_weights_alpha_half'])
        _, reps=np.unique(x.reshape(len(x),-1),axis=0,return_index=True)
        checks[label+'_enumeration']=[{'design_number':design+1,'gate_group':group,**enumerate_check(x[index:index+1],b,w[group])} for design,index in enumerate(reps) for group in range(len(w))]
    checks['completed_fold_fits']=len(folds)
    checks['max_conditional_gradient_check_error']=max(v.get('gradient_check_error',0) for v in folds)
    checks['max_projected_gradient']=max(v['projected_gradient_max'] for v in folds)
    checks['fold_fit_bounds']=[{'model':v['model'],'fold':v['fold'],'indices':v['at_bound']} for v in folds if v['at_bound']]
    checks['full_RI_quadrature_61_vs_161_loglik_difference']=fits['RI_linear']['checks']['quadrature_61_vs_161_loglik_difference']
    save(out/'numerical_checks.json',checks)
    reference=pd.read_csv(Path(__file__).resolve().parents[1]/'reports/application_rebuild_20261109/panel_model_comparison.csv')
    for label in ['MNL_linear','RI_linear']:
        current=float(scores.loc[(scores.model==label)&(scores.population=='all_360'),'joint_nll_per_choice'].iloc[0])
        previous=float(reference.loc[reference.model==label,'fivefold_joint_nll_per_choice'].iloc[0]); assert abs(current-previous)<1e-9
    old_spike=pd.read_csv(Path(__file__).resolve().parents[1]/'reports/application_rebuild_20261109/source_boundary_comparison.csv')
    previous=float(old_spike.loc[(old_spike.model=='spike_linear')&(old_spike.population=='all_360'),'fivefold_joint_nll_per_choice'].iloc[0])
    current=float(scores.loc[(scores.model=='spike_linear')&(scores.population=='all_360'),'joint_nll_per_choice'].iloc[0])
    assert abs(current-previous)<1e-9
    checks['previous_MNL_RI_spike_pooled_scores_reproduced']=True
    save(out/'numerical_checks.json',checks)
    selected=['MNL_linear','RI_linear','spike_linear','source_ASC_MNL_linear','hurdle_blind_linear','hurdle_source_linear']
    pivot=scores[scores.model.isin(selected)].pivot(index='model',columns='population',values='joint_nll_per_choice').reindex(selected)
    pivot=pivot[['all_360','300_other_labels','60_AI_label']].reset_index()
    paired=gains[(gains.model=='hurdle_source_linear')&(gains.reference=='source_ASC_MNL_linear')]
    allgain=float(paired.loc[paired.population=='all_360','mean_joint_logscore_gain_per_choice'].iloc[0])
    aigain=float(paired.loc[paired.population=='60_AI_label','mean_joint_logscore_gain_per_choice'].iloc[0])
    attribution=(aigain/6)/allgain
    # Static scientific figure, using actual aggregate values only.
    plt.rcParams.update({'svg.fonttype':'none','font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.6),gridspec_kw={'width_ratios':[1,1.2]})
    colors=['#c8d8e8','#3274a1','#6baed6','#d58d38']
    parts=np.array([[300,0,0,0],[0,15,3,42]],float)
    shares=100*parts/parts.sum(axis=1,keepdims=True)
    left=np.zeros(2)
    for j,name in enumerate(['Nonconstant','All A','All B','All outside']):
        axes[0].barh([0,1],shares[:,j],left=left,color=colors[j],label=name)
        for k in range(2):
            if parts[k,j]>0: axes[0].text(left[k]+shares[k,j]/2,k,str(int(parts[k,j])),ha='center',va='center',fontsize=9)
        left+=shares[:,j]
    axes[0].set_yticks([0,1],['Other labels (n=300)','AI label (n=60)']);axes[0].set_xlabel('Share within source (%)');axes[0].set_title('Eight-answer patterns (counts labelled)');axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=4,fontsize=8)
    names=['Observed','Source-blind outside spike','Source-aware pattern hurdle']
    values=[]
    for group in ['300_other_labels','60_AI_label']:
        rows=scores[scores.population==group]
        values.append([float(rows.observed_all_outside.iloc[0]),float(rows.loc[rows.model=='spike_linear','predicted_all_outside'].iloc[0]),float(rows.loc[rows.model=='hurdle_source_linear','predicted_all_outside'].iloc[0])])
    positions=np.arange(2)
    for k,name in enumerate(names):
        bar=axes[1].bar(positions+(k-1)*.24,np.array(values)[:,k],width=.23,label=name,color=['#444444','#d58d38','#3274a1'][k])
        axes[1].bar_label(bar,fmt='%.1f',fontsize=8,padding=2)
    axes[1].set_xticks(positions,['Other labels (n=300)','AI label (n=60)']);axes[1].set_ylim(0,53);axes[1].set_ylabel('All-outside vectors');axes[1].set_title('Five-fold expected counts by source');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'source_conditional_calibration.svg');fig.savefig(out/'source_conditional_calibration.png',dpi=180);plt.close(fig)
    report=f'''# 来源条件验证与固定选项模式

2026年11月9日。本轮完成9个固定比较模型、45次整条答卷留出拟合，以及预先固定的两种平滑强度比较。研究目标保持为建立符合JOCM要求的独立贡献；本轮补齐一个会改变模型解释的对照。

## 新核实的完整模式

60条原标签“AI样本”的记录全部在8题中选择同一选项：15条全选A、3条全选B、42条全部不购买。其余300条全部至少换过一次选项。这比“42条全程不购买集中于该来源”更完整。固定选择本身不能判定真人身份、生成来源、有效性或心理机制。

## 总数吻合掩盖了什么

不区分来源的“不购买点质量”模型，五折共预测42.000条全程不购买，与实际42条接近。分来源后，它给其余300条预测35.316条，实际为0；给60条来源组预测6.684条，实际为42。整体计数吻合不代表来源内分布得到正确预测。

来源和固定选项模式的简单对照，分别预测0.621和40.420条。两项合计41.040条；采用固定平滑会留下少量对未观察类别的概率。这些都是每折训练后对留出记录的预测，不使用留出答卷推断类别后验。

![真实计数与分来源留出预测](source_conditional_calibration.svg)

## 同一批记录与同一组折内的模型比较

下表为每次选择归一化的8题联合负对数分数，越小越好。不同来源的列是不同评分人群，不能把它们的绝对数值当作同一难度任务。价格采用线性表示，另报自由价格—规则单元的固定敏感性。

{table(pivot)}

来源调整的普通模型与来源模式对照，在其余300条中的分数为1.032544和1.032465，差距约0.000080；描述性配对区间包含零。全样本的差距为0.120097，其中约{attribution*100:.2f}%的加权平均增益来自60条来源组。自由价格单元下，其余300条中来源模式对照的分数略差，差距同样接近零。此处不能把全样本排名解释成更准确地学习了属性权衡。

来源可见和来源不可见的模式对照使用同一条件选择部分；只有模式概率是否按来源估计发生改变。两者全样本联合分数相差0.056494。该匹配比较中，差距按构造完全来自来源信息，不能再作为属性效用模型提升的证据。线性与自由单元下差距相同也由这个构造解释。

## 对照的含义与计算检查

模式对照把完整答卷分为四种互斥结果：八题全A、全B、全不购买、以及至少改变一次选项。前3种采用固定向量概率；第4种采用以非固定向量为条件的MNL。训练折内按类别计数估计模式概率，主分析采用固定Dirichlet平滑0.5，1.0作为固定敏感性；没有按结果选择平滑参数。

类别在训练数据中是结果变量，预测阶段只输入菜单、拟合参数和已知来源标签。拟合非固定部分时使用其条件似然，而非直接删除固定选择后套用未经条件化的MNL。该模型用于诊断联合预测，参数不解释为总体人群的偏好或支付意愿。

对每种题目版本及每组模式权重，穷举3^8=6,561种回答，独立核对联合概率总和、边际概率及不购买次数分布。最大误差低于10^−10；有限差分与解析梯度一致。两份原始导出哈希与冻结记录一致。明细见`numerical_checks.json`；整条答卷的预测值和分折信息保留在私有目录。

这些折此前已用于本项目分析，属于探索性内部验证。配对区间是描述性结果，未覆盖重复使用折、模型选择或整个项目探索的影响。来源权重变化表是代数标准化，不代表真实人群构成或已识别的招募因果效应。

## 与最接近文献的实质关系

[Kazagli与de Lapparent（JOCM，2023）](https://doi.org/10.1016/j.jocm.2023.100413)已在有限混合框架中区分强偏好和固定选项规则，并考虑情境对规则采用概率的影响。已读该文模型部分及式(8)，本轮不把固定向量成分命名为新理论。

[Mariel与Artabe（2025）](https://doi.org/10.1016/j.reseneeco.2025.101496)已提出针对特定选择模式过度出现的模拟检验；已核对方法和结论，并阅读其[作者GitHub仓库](https://github.com/discretechoice/simulation-based-method)。本轮采用可复核的完整模式计数和来源内预测比较，没有声称复现其检验，也未将模式出现原因当作已识别。

[Alcorta与Mariel（JOCM，2026）](https://doi.org/10.1016/j.jocm.2026.100620)已讨论遗漏个体变量与方案截距交互的偏误。已核对出版方摘要。本轮加入来源与方案截距交互，是对照基线的补全，不承担方法创新。

## 投稿判断与下一项证据

[出版方的期刊介绍](https://shop.elsevier.com/journals/journal-of-choice-modelling/1755-5345)要求方法贡献或创新应用。来源条件验证和非交易选择属于该刊读者关心的问题，题目与方法标签本身尚不能建立独立增量。

当前已成立的具体发现是：这批记录中，整体“不购买”计数的准确预测伴随明显的来源内错配；固定选项模式与来源完全对应，简单来源模式对照解释了全样本的主要预测增益。这能支撑一份案例诊断研究稿。它仍不足以建立来源的因果作用、真人与AI选择差异、平台普遍质量结论或独立新理论。

来源服务说明仍是待核实事实：已读到两项各30份的AI样本任务和点数记录，研究者报告真人招募，但两者是否对应未确认。该事实影响案例的人群与应用解释；即使来源查明，也需证明这份案例相对已发非交易和模式诊断研究的独立实质价值。下一步随新增服务或招募证据修订案例主张，不按显著性继续增加背景交互或模型种类。
'''
    (out/'来源条件验证与创新边界.md').write_text(report)
    (out/'README.md').write_text('''# Source-conditioned validation of panel choices

Nine fixed exploratory benchmarks use the existing five whole-record folds.
The original source labels remain unchanged. All 60 AI-labelled vectors are
constant; all other 300 vectors are nonconstant. This fact does not certify
respondent origin or determine why a pattern occurs.

From the repository root, using the same scientific Python dependencies as
the application rebuild:

```sh
python analysis/source_conditioned_validation.py --input-dir "$PRIVATE_DIR" --instrument survey/public_dh36/instrument.json --folds "$PRIVATE_DIR/ideal_cv_folds_private.csv" --out reports/source_conditioned_validation_20261109 --private-out "$PRIVATE_DIR/source_conditioned_validation_20261109"
python analysis/write_source_validation_report.py --input-dir "$PRIVATE_DIR" --instrument survey/public_dh36/instrument.json --out reports/source_conditioned_validation_20261109
```

The historical fold filename does not certify ideal randomization. The
four-category hurdle is a classical diagnostic with an MNL component
conditioned on nonconstant responses. Training gate counts use fixed symmetric
Dirichlet smoothing of 0.5; 1.0 is a fixed sensitivity. Prediction inputs
exclude held-out answers. Do not interpret its category probabilities or
conditional utility coefficients as causal mechanisms or population WTP.

Source-specific and aggregate scores evaluate different questions. Source
weights in the standardization table are hypothetical. Paired intervals are
descriptive; reused folds do not provide selection-adjusted or independent
validation. Public outputs contain only aggregate results and fitted model
diagnostics. Individual predictions, assignments and raw records remain private.

This package is not a new-method claim or a declaration of submission readiness.
See the Chinese research assessment and the focused manuscript note.
''')
    note='''# Source-conditional evaluation of repeated health-service choices

*Exploratory research note, 9 November 2026. The recorded source-service definition remains unresolved. This draft makes a dataset-specific diagnostic claim and no new-method priority claim.*

## Abstract

Panel choice models can reproduce the aggregate frequency of persistent nonpurchase while allocating it to the wrong source group. We examine this distinction in 360 exported questionnaire records, each containing eight health-service choices. Sixty records with one original source label choose the same option throughout; the other 300 all change options at least once. A source-blind outside-component model predicts 42 all-outside vectors, matching the observed total. Its source-specific predictions are 35.32 for the group containing none and 6.68 for the group containing all 42. A transparent source-aware pattern benchmark reduces these predictions to 0.62 and 40.42. Its improvement over a source-adjusted ordinary model is concentrated in the 60-record group; joint scores in the other 300 records are nearly identical. These results distinguish aggregate calibration from source-conditional prediction in this dataset. They do not identify the cause of repeated choices or establish respondent origin.

## 1. Question

Repeated health-service choices combine attribute comparisons with persistent participation patterns. Their joint likelihood is sensitive to both. When source groups have different answer-vector distributions, a pooled model can improve its overall score by representing their composition. The practical question is where that improvement occurs and whether predicted persistent nonpurchase is allocated to the groups in which it is observed.

Non-trading models already represent constant choices and distinguish possible mechanisms. [Kazagli and de Lapparent (2023)](https://doi.org/10.1016/j.jocm.2023.100413) provides the relevant JOCM comparison. [Mariel and Artabe (2025)](https://doi.org/10.1016/j.reseneeco.2025.101496) tests excess frequencies of specific choice patterns. The present analysis uses these established concerns to evaluate source-conditional predictions in an existing questionnaire.

## 2. Records and comparisons

Two questionnaire versions each supply 180 records and eight choices among service A, service B and nonpurchase. Monthly fees and sharing-rule packages vary; other service-card features are represented by version-specific alternative intercepts. Different recruitment channels administered the two versions, with no retained allocation log. Version effects are channel/version associations.

The original source field labels 60 records as “AI样本”. Authenticated platform inspection confirms two 30-record task notices and matching AI-point usage. The researcher reports a human-recruitment description; its applicability to these tasks remains unverified. The analysis therefore uses source labels literally, without claiming a human-versus-generated comparison. There are 15 all-A, three all-B and 42 all-outside vectors in this group. All other 300 vectors are nonconstant. Constant choices alone establish neither invalidity nor their behavioral explanation.

Nine fixed benchmarks include ordinary MNL, a normal random service intercept, an outside-component mixture, source interactions with alternative intercepts and a four-category pattern hurdle. Linear fees and unrestricted observed fee-rule cells are fixed specification comparisons. The hurdle assigns probabilities to all-A, all-B, all-outside and nonconstant vectors. Its nonconstant component uses the correctly conditioned MNL likelihood. This is a predictive diagnostic; its conditional coefficients are not population preference or WTP estimates.

Five folds retain each record's eight choices together. Pattern probabilities are learned from training counts, with fixed symmetric Dirichlet smoothing of 0.5 and a sensitivity of 1.0. Held-out predictions use menus, fitted parameters and source metadata. Held-out answers enter scoring, with no posterior class assignment used for prediction. The folds were previously used during project exploration and provide internal validation.

## 3. Aggregate and source-specific evidence

The source-blind outside-component model predicts 42.00 all-outside vectors in total. It allocates 35.32 to the 300-record group, where none occur, and 6.68 to the 60-record group, where all 42 occur. Thus its accurate overall count coexists with substantial source-conditional errors.

The source-aware hurdle predicts 0.62 and 40.42 all-outside vectors in the two groups. Its expected total of 41.04 reflects the fixed smoothing of unobserved training categories. In the linear-fee comparison, joint negative log scores per choice are 0.956676 for the outside component and 0.877234 for the source-aware hurdle. Their marginal scores are 1.040234 and 0.992546. These scores address different distributions: complete vectors and individual choices.

Against the source-adjusted ordinary model, the hurdle's joint score changes from 0.997330 to 0.877234 in the pooled sample. Scores in the other 300 records are 1.032544 and 1.032465; their descriptive paired difference spans zero. Approximately 99.94% of the weighted pooled mean gain comes from the 60-record group. Unrestricted fee-rule cells give the same conclusion, with a near-zero difference in the other-source group.

The matched source-aware and source-blind hurdles share the same conditional utility component. Their pooled score difference, 0.056494, comes entirely from their different pattern gates by construction. It cannot be attributed to improved attribute-utility estimation. All probabilities and marginal predictions were checked against exhaustive enumeration of the 6,561 possible vectors for both versions and gate groups. The aggregate tables and numerical checks accompany the note.

## 4. Interpretation

Source-stratified evaluation changes the interpretation of the pooled predictive gain in these records. The known source field and constant-vector structure account for the main difference; a richer utility distribution is not required to reproduce it. A claim of stable preference types needs evidence beyond the pooled score and overall nonpurchase count.

The analysis concerns one questionnaire with a sharply separated source pattern. It does not identify a recruitment effect, label a record as generated, or establish how often this issue occurs in other studies. The task-specific service definition must determine the final population statement. Constant choices may arise for several reasons, and their frequency alone cannot resolve them. The reusable result is a fully specified case comparison showing how overall calibration and source-conditional allocation answer different empirical questions.

## Appendix: Conditional pattern likelihood

Let p_tj be the MNL probability for alternative j at task t. A constant-j vector has probability c_j=product_t p_tj. For a nonconstant vector y, the conditional probability is product_t p_t,y_t divided by (1−sum_j c_j). Conditional marginal probabilities are (p_tj−c_j)/(1−sum_j c_j). Training counts estimate the four pattern probabilities, either pooled or by source. This disjoint-support construction normalizes the entire vector distribution and preserves explicit prediction inputs.

The comparison uses standard conditioning and categorical smoothing. These calculations are not claimed as an independently new theorem. A publication contribution must come from a distinct, adequately documented empirical finding and its consequences for choice-model use.
'''
    (out/'focused_research_note.md').write_text(note)
    print(json.dumps({'fold_fits':len(folds),'source_group_contribution_fraction':attribution,'raw_hashes_match':True,'checks_and_report_written':True}))


if __name__=='__main__': main()
