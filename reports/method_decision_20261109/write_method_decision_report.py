"""Write the evidence-based research decision and a reproducible resource note."""
from pathlib import Path
import argparse,hashlib,json,shutil,zipfile
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def write_json(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args(); out=args.out
    support=pd.read_csv(out/'published_support_and_classical_equivalence.csv'); repair=pd.read_csv(out/'fixed_budget_repair_vs_classical.csv'); sims=pd.read_csv(out/'matched_classical_test_benchmark.csv')
    technical=json.loads((out/'audit_technical_verification.json').read_text()); kl=json.loads((out/'KL_and_scope_verification.json').read_text()); frozen=json.loads((out/'exploratory_freeze_manifest.json').read_text())
    assert len(support)==14 and len(repair)==14 and len(sims)==54
    assert (sims.framework_CMH_p_mismatches==0).all()
    assert max(support.classical_Schur_max_difference)<1e-12
    assert frozen['records']==360 and technical['real_archive_cases']==2
    total_replications=int(sims[sims.method=='fibre_full_profile_score'].replications.sum()); assert total_replications==9000
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'method-decision-20261109'})
    fig,ax=plt.subplots(figsize=(7.3,4.2)); route=support.query("dataset=='route'")
    for rep,label in [('absolute_menu_summary','Absolute candidate-menu summary'),('utility_difference_vector','Complete utility-difference vector')]:
        g=route.query('representation==@rep'); ax.plot(g.parameter,g.menu_contrast_rank,'o-',label=label)
    ax.set_xlabel('Analyst-defined value of time (CHF/hour)'); ax.set_ylabel('Finite-support menu contrast rank'); ax.set_title('Swiss released task support: two different target nulls'); ax.legend(fontsize=8); ax.grid(alpha=.15); fig.tight_layout(); fig.savefig(out/'representation_scope.png',dpi=180); fig.savefig(out/'representation_scope.svg',metadata={'Date':None}); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9.8,3.9),sharey=True)
    for ax,(dataset,g) in zip(axes,sims.query("method!='classical_full_profile_CMH'").groupby('dataset_source',sort=False)):
        for n in [200,400,800]:
            h=g.query('n_independent_single_choice_records==@n and method=="fibre_full_profile_score"'); ax.plot(h.eta_RMS_centered_utility_departure,h.rejection_rate,'o-',label=f'n={n}')
            lr=g.query('n_independent_single_choice_records==@n and method=="saturated_profile_LR"'); ax.scatter(lr.eta_RMS_centered_utility_departure,lr.rejection_rate,marker='x',color=ax.lines[-1].get_color(),s=35)
        ax.set_title(dataset+' diagnostic support'); ax.set_xlabel('RMS-normalized utility departure'); ax.set_ylim(0,1); ax.grid(alpha=.15)
    axes[0].set_ylabel('Conditional rejection proportion'); axes[0].legend(fontsize=8); fig.suptitle('Lines: fibre score = classical CMH exactly; x: saturated LR',fontsize=11); fig.tight_layout(); fig.savefig(out/'matched_comparator_power.png',dpi=180); fig.savefig(out/'matched_comparator_power.svg',metadata={'Date':None}); plt.close(fig)
    table=['| 公开案例与映射 | 原始不同菜单 | 非平凡纤维 | 菜单对比rank |','| --- | ---: | ---: | ---: |']
    for ds,par,rep,label in [('route',10,'absolute_menu_summary','瑞士路线；10 CHF/小时；绝对摘要'),('route',10,'utility_difference_vector','瑞士路线；10 CHF/小时；完整效用差向量'),('vehicle',60,'absolute_menu_summary','车辆；60个月总成本；绝对摘要'),('vehicle',60,'utility_difference_vector','车辆；60个月总成本；完整效用差向量')]:
        r=support.query('dataset==@ds and parameter==@par and representation==@rep').iloc[0]; table.append(f'| {label} | {int(r.unique_raw_menus)} | {int(r.non_singleton_fibres)} | {int(r.menu_contrast_rank)} |')
    max_matrix=support.classical_Schur_max_difference.max(); max_stat=max(v['max_score_CMH_statistic_difference'] for v in technical['simulations'].values())
    report=fr'''# 研究去留结论：停止当前新方法投稿主线，保留复现与设计审计成果

日期：2026年11月9日。

**最终决定：当前“有限纤维充分性新理论／新检验＋曝光证书＋修复算法”的独立方法增量未建立，不建议按这条主线向Journal of Choice Modelling投稿。现在不重发问卷，也不给出追加样本数。** 已按顺序完成探索资料封存、经典文献核对、真实公开设计支撑审计、固定预算修复比较和匹配模拟。新问卷与正式样本量设计的前置条件没有通过。

这个决定针对当前方法稿的贡献依据。它不表示选择实验设计审计没有实用价值，也不把不显著结果归因于收样者。

## 1. 现有360条资料已封存

两版原始工作簿和已有分析包在本地受控目录留存校验快照，原始文件未修改，公开材料只保留哈希与聚合结果。已有360条继续作为探索性和技术实现资料，来源与实际发放方式按原记录说明。理想分配情景的计算不能变成已核实的招募事实。

原分析及收尾核查固定在Git提交 `bbf3ceca96950c958ecb660b6a79a8497dfd71b8`。不显著结果、来源核查和模型识别限制均保留；本次没有重分组、换随机种子或替换主检验来争取显著性。

## 2. 当前核心对象可由经典模型完整表达

在声明的有限支撑上，任意φ响应面就是每个纤维任意一组分类概率。用纤维指示变量作饱和干扰参数后，经典加项模型的Fisher信息Schur补，逐项等于当前框架的纤维内信息。

\[
I_{{\rm fibre}}=\sum_z\sum_{{i\in F_z}}w_i(H_i-\bar H_z)^{{\mathsf T}}V_z(H_i-\bar H_z)
=I_{{\rm classical}}.
\]

相应rank是相同饱和原假设的失拟对比维数。相同方向集合和归一化下，最小特征值是经典最弱方向信息准则；剖面KL的二阶项也是同一个信息。详细推导见[classical_equivalence_proof.md](classical_equivalence_proof.md)。

这次比较保留了所有φ函数；没有把充分性原假设偷换为一个低维线性属性模型。分类响应、可行支撑和预先计算rank没有自动产生独立数学增量。连续情形中的条件方差表达也不能单靠标准干扰投影成为新理论。

文献核查使用了Goos等（2005）的模型敏感设计矩阵、Wiens（2010）的非线性响应KL与功效设计、Atkinson–Fedorov的模型判别先例，并核对了条件随机化、充分降维及近期JOCM邻近工作。阅读层级、差异和来源见[novelty_comparison.csv](novelty_comparison.csv)。这些邻近工作并非每一篇都与本文完全相同；本次否定新方法依据的关键证据是上面的同原假设等式及匹配计算。

## 3. 已审计两个真实公开SP档案，覆盖14种映射设置

瑞士公共交通路线档案来自388人的3,492次选择，包含3,417个不同的显示菜单。车辆档案来自Train作者软件包中的100人公开子集，共1,484次三选一任务，1,484个不同菜单。原始研究与官方字典均有来源；审计没有用他们的回答做新的行为拒绝检验。

候选映射由本次审计明确设定：路线采用每小时5、10、20、40 CHF的广义成本敏感性值；车辆采用48、60、84个月的未折现购置与运行总成本。这些值是可解释的分析设定，不宣称原作者使用了它们，也不宣称它们是真实偏好参数。

{chr(10).join(table)}

“绝对菜单摘要”与“完整候选效用差向量”给出不同的非参数原假设。后一种要求共同平移不改变选择响应，通常更严格；前者允许响应随绝对摘要水平变化。不能在看过结果后更换二者以争取更好的rank。即使固定为绝对摘要，路线案例在10 CHF/小时下也有17个非零对比，因此不能声称常用设计普遍结构性盲。

车辆案例在这些映射下没有非平凡纤维。这说明该已发布支撑不能检验对应的任意摘要响应面；不表示原车辆研究无效，也不表示其参数化模型不能接受其他形式的检验。两个案例不是行业代表性抽样，本次没有据此估计“多少比例D-efficient设计有问题”。原始设计生成算法也未被完整复刻。

![表示定义与对比rank](representation_scope.svg)

## 4. 修复有效，但经典准则给出同样结果

从每个公开案例构造16个二点诊断纤维，改变一个备选项的时间／费用分解或购置／运行成本分解。它们遵守明确的边际范围与数字步长，完整候选菜单保持不变。原研究的全部联合可行性约束和人类理解尚未复核，所以这些是计算探针，不是可以直接发布的问卷。

比较固定320次选择机会、每个纤维20次，不给修复方法额外预算。在七种分配比例下，候选模型原假设下的参数信息保持不变；当前纤维目标与经典残差信息一致。给定二点支持时，两者都在1∶1分配下得到最大局部信息。这是真实可执行的设计诊断，但在此条件下没有新的优化原则或方法优势。

14种公开支撑审计中，两个信息矩阵的最大元素差约为{max_matrix:.2e}。另外40格剖面KL计算验证其二阶对应关系；这没有把局部近似当成有限幅度下的全局等式。

## 5. 强基线比较没有提供额外检验能力

在基于公开属性生成的诊断支撑上，采用18个数据生成设置，每格500次模拟、499次固定边际条件抽样，共9,000次模拟。每个模拟记录只有一次选择，避免把它冒充重复作答问卷的样本量规划。

比较完整纤维score、完整剖面CMH形式与饱和剖面LR。这里的CMH形式把各纤维分量堆叠，允许各纤维有不同关联方向；它不是只检验共同关联方向的低维CMH。三者使用相同响应信息、备择空间、选择预算和条件校准。score与完整剖面CMH形式在全部9,000次模拟中的p值逐次相同，统计量最大差约{max_stat:.2e}；LR的有限样本拒绝率接近。全部零假设行、非天花板功效和Monte Carlo区间见[matched_classical_test_benchmark.csv](matched_classical_test_benchmark.csv)。

![匹配强基线比较](matched_comparator_power.svg)

η是按原假设响应度量归一化的效用偏离，不是百分点差异。这些曲线不能转换为新的人类DCE“需要收多少份”。机器学习和观测性模型的不同检验目标，也不能靠选择一个手写字典的失败来证明当前方法优势。

## 6. 权重、精度与有限支撑边界

严格等值纤维依赖声明的映射与记录精度。路线绝对摘要在10 CHF/小时下有17个菜单对比；把映射改为10.000001 CHF/小时后，本档案的3,417个菜单全部成为单点纤维。计算使用精确有理数，没有把四舍五入导致的近似相等当成精确相等。

这两种映射定义不同原假设。结果没有推翻原固定权重检验；它说明在应用中必须独立说明权重的含义与精度。若允许在一个权重区间任选映射、同时允许任意摘要响应面，固定有限目录往往可被某个一一映射索引，检验会失去内容。这个基本可识别性事实并非本次声称的新定理；它也不能被误用于否定连续总体中的单指标检验。

## 7. 已执行的改题处理

当前成果改为 **“离散选择实验设计可检验性审计：公开案例、经典等价关系与复现工具”**。输出改为可复现诊断资源，不再以新理论、新统计检验或新maximin原则组织论文。

已交付通用的菜单支撑审计入口 `design_support_audit.py`，允许输入任意声明的完整原始菜单、候选菜单、权重及等价表示，输出经典结构维数并明确区分结构可检验性、充分性与功效。随包给出合成教学例子和两个公开案例的复现脚本。具体技术说明稿见[design_audit_resource_note.md](design_audit_resource_note.md)。

这项改题已经落实为资源和说明稿；它目前不是另一个被证明达到JOCM创新门槛的新选题。若希望继续以独立方法论文投稿，需要新的实质问题或经过验证的独特操作，不能把本工具换个名称重新提出同样理论。

## 8. 新问卷和正式样本量的最终决定

按事先确定的顺序，新调查以方法增量成立为前置条件。本次该条件未通过，所以新问卷与新样本量步骤不启动。没有额外收样建议，没有复用旧设计的功效曲线，也没有发布计算探针。

未来重新进入人类研究，需要独立科学问题、有实质依据的预先固定摘要、多条独立原始变化方向、完整菜单保持、渠道内有日志的随机分配、研究负责人与联系安排、适用伦理程序和理解测试。届时再依据新设计及预定最小有意义效应进行受访者层面的功效计算。现在给出一个追加人数会绕过已经失败的贡献门槛。

## 9. 证据边界与历史材料

Scopus检索入口返回403，Google Scholar入口在本环境不可访问。本次已对可访问的一手论文、作者页面和官方数据进行定向核查，但不宣称数据库检索完备，也不保证全球优先权。

旧“第32轮达到投稿线”等文档是内部阶段判断，已被本次基于等价推导和真实支撑审计的决定覆盖；它们保留为历史记录，不作为独立同行评审。原360条分析保持探索性，不用其结果支持已完成真实随机实验的叙事。

## 主要一手来源

- [Goos等：模型稳健与模型敏感设计](https://webpages.math.luc.edu/~tobrien/research/GKOV.pdf)
- [Wiens：二元响应下的失拟检验与估计设计](https://sites.ualberta.ca/~dwiens/home%20page/pubs/binary%20lack%20of%20fit.pdf)
- [Apollo官方数据与字典](https://apollochoicemodelling.com/examples.html)
- [Axhausen等原研究](https://doi.org/10.1016/j.tranpol.2008.02.001)
- [Train官方代码与样本档案](https://eml.berkeley.edu/Software/abstracts/train1006mxlmsl.html)
- [Hoshino与Yanagi：暴露映射的条件随机化检验](https://doi.org/10.1002/jae.70076)

''' 
    (out/'最终研究去留报告.md').write_text(report)
    note='''# Auditing testability in released discrete-choice designs

Technical resource note — 9 November 2026

## Purpose

This resource checks the positive-mass support of a declared candidate menu representation before a new choice experiment is fielded. It implements classical finite-support lack-of-fit dimensions, projected categorical information and a matched-comparator check. No new general test or optimality principle is claimed.

## Demonstrated scope

Two released stated-choice archives are audited: 3,492 Swiss route-choice tasks (3,417 unique menus) and 1,484 three-alternative vehicle tasks. Four value-of-time settings and three ownership horizons, each under two declared representations, produce fourteen audits. The analyst-defined summaries are sensitivity cases, not original-author hypotheses. The original answer outcomes are not used to infer behavioral sufficiency or valid assignment.

At 10 CHF/hour, the route archive has 17 menu contrasts under an absolute summary and 2,157 under the complete utility-difference vector. These are different nonparametric nulls. The vehicle catalogue has singleton fibres in the reported settings. These facts do not invalidate either original study or establish a prevalence rate for efficient-design failures.

## Classical identity

For a saturated fibre response surface and a fixed local departure space, within-fibre categorical information equals the nuisance-parameter Schur complement of the corresponding added-term model. Rank, directional local power and the matched weakest-direction criterion follow from that matrix. Finite-amplitude profiled KL has the same local second-order coefficient.

## Generated diagnostic designs

Sixteen pairs per source archive preserve the complete declared candidate menu under specified marginal ranges and ticks. At a fixed choice budget, balanced two-point allocation maximizes both the fibre and classical information, with candidate parameter information unchanged under the null. Full joint feasibility and respondent comprehension were not established. These generated probes are not deployable questionnaires.

## Comparator evidence

Eighteen controlled settings use 500 replications and 499 fixed-margin reference draws each. The full-profile score and classical CMH produce identical p-values in all 9,000 replications; a saturated profile LR gives similar rejection rates. Simulated units each supply one choice. This benchmark is not a sample-size recommendation for a repeated-question human survey.

## Use

The generic entry point accepts complete raw menus, complete candidate menus and weights as JSON. It uses exact rational comparisons and reports positive-mass contrast dimensions. It never reports a positive rank as evidence of sufficiency, statistical power, original feasibility, actual random assignment or human provenance.

The public source datasets are obtained from their official sites and checked by hashes. Raw DH36 questionnaires remain private and exploratory. Research users should cite the original archives and classical method sources listed in the accompanying decision report.

## Publication status

This is a completed reproducibility and design-diagnostic resource. Independent methodological novelty and readiness for Journal of Choice Modelling submission have not been established. The previous new-method manuscript is retained as an archived working draft. No new questionnaire is recommended on the basis of the retired contribution claim.
'''
    (out/'design_audit_resource_note.md').write_text(note)
    pivot='''# 改题记录：贡献类型改为可复现设计诊断资源

## 已落实的选择

当前题目改为“离散选择实验设计可检验性审计：公开案例、经典等价关系与复现工具”。交付物是可运行的工具、公开案例审计和技术说明稿。它沿用经典统计结果，不宣称另创理论。

## 独立候选问题的结果级审查

| 候选方向 | 改变的研究对象或输出 | 本次结论 |
| --- | --- | --- |
| 完整纤维检验作为新方法 | 原统计量的表述 | 与相同原假设下的经典score/CMH一致，撤去新检验主张 |
| 最低成本曝光修复 | 设计分配规则 | 在检验的二点情形由经典均衡分配给出，未建立新优化原则 |
| 未知／学习权重下的可检验性 | 由固定摘要改为权重族 | 有真实范围问题；有限目录的一一映射现象是基本识别事实，尚无可确认的新方法贡献 |
| 偏好机制或政策决策损失 | 从一般充分性改为实质行为／决策问题 | 本次资料未预先识别相关机制或独立政策目标，不启动新调查来补叙事 |
| 设计可检验性诊断资源 | 从新检验改为可复现工具与案例 | 已实现；保留使用价值，公开范围和经典来源，暂不作为新的JOCM方法投稿 |

这些方向分别改变一个研究目标或约束，没有把几个既有理论拼接成新的理论名称。现有证据不支持给其中任何一个授予原创性或录用保证。

## 继续学术投稿的边界

继续以独立方法论文推进，需要能单独陈述并验证的新增操作或科学发现。仅增加缩写、定理包装、数据集数量或问卷人数，不改变此次等价关系。当前项目以资源方式收束；JOCM新方法主线停止。
'''
    (out/'research_pivot_decision.md').write_text(pivot)
    completion={'scope':'Complete the agreed sequential research decision; new survey and field power are conditional on methodological increment.','requirements':[{'step':1,'status':'completed','evidence':'Exploratory original and analysis snapshots verified; no source metadata changed.'},{'step':2,'status':'completed with negative independent-method decision','evidence':'Primary literature comparison; two published released SP supports; fourteen map cases; classical Schur/KL identity; fixed-budget repair; nine thousand matched replications.'},{'step':3,'status':'not triggered by failed prerequisite; decision completed','evidence':'No independent increment established; no additional field questionnaire published. Generated diagnostic probes explicitly not field validated.'},{'step':4,'status':'not triggered by failed prerequisite; decision completed','evidence':'No new human design proceeds, so no additional sample-size recommendation or reuse of old planning results.'}],'pivot_completed':'Generic classical-support audit tool and technical resource note delivered. Independent-new-method submission claim retired.','final_decision':'stop current new-method JOCM submission; preserve exploratory data; retain reproducibility resource; no new field spending on this claim','claim_of_publication_readiness':False,'all_required_decisions_resolved':True,'no_success_or_originality_guarantee':True}
    write_json(out/'sequential_completion_audit.json',completion)
    src=Path(__file__).parent
    for name in ['published_dce_decision_audit.py','classical_kl_scope_audit.py','design_support_audit.py','write_method_decision_report.py','fetch_published_dce_inputs.py']:
        if (src/name).exists(): shutil.copy2(src/name,out/name)
    import scipy,threadpoolctl
    versions={'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'threadpoolctl':threadpoolctl.__version__}
    (out/'requirements.txt').write_text(''.join(f'{k}=={v}\n' for k,v in versions.items()))
    readme='''# Reproducible DCE design audit and research decision, 2026-11-09

Read 最终研究去留报告.md, classical_equivalence_proof.md and survey_and_sample_decision.json. The current independent-new-method claim is not supported. The output is a completed classical-method diagnostic resource; it is not a submission-ready new theory or deployable new questionnaire.

## Reproduce

Use Python 3.12 and requirements.txt. Official public inputs are downloaded to a separate directory. No DH36 raw exports are included or needed to run the public design audit.

```sh
python fetch_published_dce_inputs.py --out-dir INPUTS
python published_dce_decision_audit.py --route-data INPUTS/apollo_swissRouteChoiceData.csv --vehicle-data INPUTS/vehicle_data.txt --out RESULTS --replications 500 --draws 499
python classical_kl_scope_audit.py --route-data INPUTS/apollo_swissRouteChoiceData.csv --vehicle-data INPUTS/vehicle_data.txt --out RESULTS
python design_support_audit.py --input teaching_example_design.json --out RESULTS/teaching_example_audit.json
```

The final decision report and bibliography are curated artifacts. The report-generation helper reads their frozen decision context and analysis outputs; run it against this complete result directory if rebuilding its figures. The figures compare different null representations explicitly and do not imply they are interchangeable.

All 9,000 simulation records are computer generated and each has one response. They are not respondents and are not field sample planning. Diagnostic probe feasibility is limited to the declared ticks and marginal bounds. Candidate representations and nominal weights are audit choices, not hypotheses attributed to the source authors. No original choice outcomes enter a new behavioral rejection claim.

## Generic support audit

Each JSON menu contains all alternatives, a declared candidate vector for each, and a nonnegative design weight. The representation is either absolute_menu_summary or utility_difference_vector (all non-reference candidate differences). Numeric values are compared as exact rational numbers. Repeated identical raw menus are merged; zero-mass support is omitted. A raw menu cannot have conflicting candidate values. Positive structural rank is not evidence that the representation is sufficient or that a statistical test has power.

## Evidence limits

The two data archives and their exact checksums are in published_data_sources.json. The release audits the displayed catalogue, not every original design-generation constraint. Scopus/Scholar access was unavailable as recorded in search_and_access_log.json. No exhaustive literature or global priority guarantee is made. Previous internal submission-threshold statements are archived judgments and are superseded by this decision.

Private DH36 raw snapshots and their exploratory analysis remain preserved; only their hashes are referenced. There are no new human records, contact details or private response vectors in this package.

SVG figures are the canonical figures in the repository. Optional PNG previews are included in the local ZIP but omitted from the canonical MANIFEST.sha256 and GitHub release directory; the same plotting helper recreates both formats.
'''
    (out/'README.md').write_text(readme)
    manifest=[]
    for p in sorted(out.iterdir()):
        if p.is_file() and p.suffix!='.png' and p.name not in ['MANIFEST.sha256','github_sync_receipt.json']: manifest.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name)
    (out/'MANIFEST.sha256').write_text('\n'.join(manifest)+'\n')
    archive=out.parent/'DCE_研究去留与复现审计_20261109.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file(): z.write(p,out.name+'/'+p.name)
    print('DECISION_REPORT_AND_RESOURCE_COMPLETE',archive,flush=True)

if __name__=='__main__': main()
