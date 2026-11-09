"""Generate checked scientific figures, reports and aggregate-only release."""
from pathlib import Path
import argparse,hashlib,json,shutil,zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dh36_observational_analysis as old
from dh36_assumption_conditioned_validation import js


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',type=Path,required=True); ap.add_argument('--instrument',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args(); out=args.out
    final=json.loads((out/'finalization_record.json').read_text()); assert final['complete'] and final['raw_hashes_unchanged']
    spec=json.loads(args.instrument.read_text()); d,*_=old.read_exports(args.input_dir,spec)
    y=d[[f'q{i}' for i in range(1,9)]].to_numpy(int)-1; z=d.version2.to_numpy(); ss=np.eye(3)[y][:,:,:2].reshape(len(y),16)
    moments={}; counts=[]
    for v in (0,1):
        a=ss[z==v]; moments[f'V{v+1}']={'n':len(a),'sum_score':a.sum(0).tolist(),'sum_outer_score':(a.T@a).tolist()}
        ct=np.eye(3)[y[z==v]].sum(0)
        for t in range(8):
            for j,alt in enumerate(['A','B','opt_out']): counts.append({'version':f'V{v+1}','task':t+1,'alternative':alt,'count':int(ct[t,j]),'n':len(a),'probability':ct[t,j]/len(a)})
    js(out/'aggregate_cluster_moments.json',{'coordinates':['task'+str(t+1)+'_'+a for t in range(8) for a in ['A','B']],'groups':moments,'contains_individual_response_vectors':False})
    pd.DataFrame(counts).to_csv(out/'all_record_choice_counts.csv',index=False)
    means=[]; covs=[]
    for v in ['V1','V2']:
        a=moments[v]; n=a['n']; total=np.array(a['sum_score']); means.append(total/n); covs.append((np.array(a['sum_outer_score'])-np.outer(total,total)/n)/(n-1)/n)
    delta=means[1]-means[0]; w=float(delta@np.linalg.inv(covs[0]+covs[1])@delta)
    tests=pd.read_csv(out/'randomization_tests_high_precision.csv'); assert abs(w-tests.statistic.iloc[0])<1e-9
    ml=json.loads((out/'ML_honest_holdout.json').read_text()); power=pd.read_csv(out/'power_DGP_calibrated.csv'); cal=pd.read_csv(out/'null_calibration_extended.csv'); cv=pd.read_csv(out/'independent_start_panel_joint_CV.csv'); co=pd.read_csv(out/'random_intercept_model_coefficients.csv'); mnl=pd.read_csv(out/'MNL_coefficients.csv')
    tech=json.loads((out/'technical_verification.json').read_text()); panel=json.loads((out/'random_intercept_model_validation.json').read_text())
    exact=json.loads((out/'exact_marginal_validation.json').read_text())
    exact_tasks=pd.read_csv(out/'exact_marginal_task_tests.csv')
    assert len(power)==60 and len(cal)==12 and len(tests)==3 and len(cv)==3
    assert power.power_or_size.between(0,1).all() and cal.rejection_rate.between(0,1).all()
    assert all(np.isfinite(df.select_dtypes(include='number').to_numpy()).all() for df in [power,cal,cv,co])
    assert max(abs(power.query('max_probability_difference==0').power_or_size-.05))<.05
    # Dedicated publication-sized static figures; no misleading identical scale
    # between marginal and joint predictive log scores.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),sharey=True)
    titles={'all_tasks_A_vs_B':'A vs B: all 8 tasks','one_task_A_vs_B':'A vs B: one task','all_tasks_service_vs_optout':'Services vs opt-out: all tasks'}
    for ax,(direction,g) in zip(axes,power.groupby('direction',sort=False)):
        for n,h in g.groupby('n'):
            ax.plot(100*h.max_probability_difference,h.power_or_size,'o-',label=f'n={n}',markersize=3)
            ax.fill_between(100*h.max_probability_difference,h.mc_low,h.mc_high,alpha=.09)
        ax.axhline(.8,color='grey',ls='--',lw=.8); ax.set_title(titles[direction],fontsize=10); ax.set_xlabel('Maximum probability difference (pp)'); ax.set_ylim(0,1.02); ax.grid(alpha=.15)
    axes[0].set_ylabel('Power / null rejection rate'); axes[0].legend(fontsize=8); fig.suptitle('Hypothetical power under ideal assignment; copula dependence = 0.5',fontsize=11); fig.tight_layout(); fig.savefig(out/'prospective_power.png',dpi=180); fig.savefig(out/'prospective_power.svg'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3.9)); names=['MNL product','Random intercept','Random intercept + version']; vals=cv.joint_NLL_divided_by_8.to_numpy(); ax.scatter(vals,np.arange(3),s=65,color=['#8795a1','#477f9b','#2b657f']); ax.set_yticks(np.arange(3),names); ax.set_ylim(-.5,2.5); ax.set_xlim(.96,1.06); ax.set_xlabel('Joint negative log score / 8 (lower is better)'); ax.set_title('5-fold validation: all 8 responses held out together'); ax.grid(axis='x',alpha=.15)
    for i,v in enumerate(vals): ax.text(v+.001,i,f'{v:.4f}',va='center',fontsize=10)
    fig.tight_layout(); fig.savefig(out/'joint_prediction.png',dpi=180); fig.savefig(out/'joint_prediction.svg'); plt.close(fig)
    def pw(direction,n,delta): return float(power.query('direction==@direction and n==@n and max_probability_difference==@delta').power_or_size.iloc[0])
    price=mnl.query("model=='mnl_common' and term=='price_per_10_CNY'").iloc[0]; ri=co.query("model=='random_intercept_common' and term=='log_sd_service_intercept'").iloc[0]
    gain=cv.query("model=='random_intercept_version_ASC'").iloc[0]
    caljoint=cal.query("scenario=='joint_null_independent' and test=='studentized_Wald_permutation'").iloc[0]
    claim_rows=[
      {'requirement':'全量纳入','status':'已验证','evidence':'360条全部进入全样本模型，2880次选择，未删样；technical_verification.json'},
      {'requirement':'每条均为不同真人作答','status':'未核实，不能由模型证明','evidence':'保留原始来源字段；本次不按来源拆分模型，未更改来源事实'},
      {'requirement':'理想随机分配','status':'作为条件假设完成分析；实际实施未验证','evidence':'assumptions.json；既有招募记录仍有效'},
      {'requirement':'候选保持与完整菜单','status':'在这16个菜单上已验证','evidence':'8个二点纤维，菜单对比秩8，概率对比秩16；support_certificate.json'},
      {'requirement':'数值优化和技术核验','status':'已验证所列模型','evidence':'独立聚合目标、梯度、正定Hessian、重编码、原始文件hash、积分收敛、10个训练折模型'},
      {'requirement':'与充分性原假设匹配的推断','status':'条件性、探索性；有限样本弱原假设检验未拒绝','evidence':f'整体置换p={tests.permutation_p_plus_one.iloc[0]:.6f}；逐题精确检验全局Bonferroni p={exact["global_Bonferroni_p"]:.6f}；整体置换有限样本精确性还要求联合可交换'},
      {'requirement':'比较经典基线','status':'已完成，未发现方法优势','evidence':'平衡分配下score与Wald置换排序相同；完整交互LR接近；留出ML未拒绝'},
      {'requirement':'每项原始属性的作用','status':'该设计不可识别','evidence':'13列模型秩7，6个不可识别方向；固定属性捆绑'},
      {'requirement':'结构秩与统计功效的区分','status':'已完成限定分析','evidence':'rank和归一化特征值单列；60格假设效应功效，独立零假设校准'},
      {'requirement':'独立创新和JOCM投稿门槛','status':'现有模型结果不能证明达标','evidence':'条件关联不证明方法优越、原创优先权、压缩规则的实质意义或真实实验有效性'}]
    pd.DataFrame(claim_rows).to_csv(out/'claim_and_completion_audit.csv',index=False)
    # Markdown is the reusable report; code is provided only for reproducibility.
    lines=['# 全量360条记录：理想分配假设下的模型与技术验证','日期：2026年11月9日。',
    '**结论：360条足以运行本次模型并识别部分版本差异；结果不能证明“所有关键效应均有足够功效”，也不能确认论文创新点已经达到投稿门槛。** 总体随机化检验接近0.05；允许任意题间相关的有限样本精确边际检验及独立留出检验均未拒绝。数据来源与实际分配是证据问题；扩大样本不能自动修复这些问题。',
    '## 分析口径',
    '本次按要求将两版各180条、合计360条全部纳入，不按来源拆分模型，不加入渠道变量，不以速度或回答模式删样。把记录相互独立、个体层面180∶180完全随机分配、无干扰作为条件假设。原始导出文件未修改；原始来源标签未改写为已核实真人。实际招募核查见此前的《DH36模型与数据核查报告》。假设下的分析结果不能反向证明假设已在发放时实施。',
    '8题选择编码逐项核对，完整选择数为2880，按A/B/弃选展开为8640行。统计和交叉验证均以整份答卷为单位。分析是在查看数据后形成，属于探索性分析，未声称预注册。',
    '## 直接回答：数量够不够、质量有什么问题',
    f'“够不够”取决于需要发现多大的效应。在潜在相关系数0.5、各题基础概率按本次全量数据校准的假设情景中，360条对8题均出现5个百分点A/B概率差异的功效约为{pw("all_tasks_A_vs_B",360,.05):.1%}，8个百分点约为{pw("all_tasks_A_vs_B",360,.08):.1%}。如果差异只在一题出现，8个百分点的功效约为{pw("one_task_A_vs_B",360,.08):.1%}。这些是预设概率差异的模拟，不是从本次p值反推“事后功效”，也不是对新增样本的保证。',
    '完成率和合法编码均通过；这证明记录可用于计算。真人来源、记录独立性、分配实施和理解程度需要不同证据。模型拟合、显著性和预测准确率均不能替代这类证据。不要把“记录完整”写成“已证明可靠真人样本”。',
    '## 原假设和支撑证书',
    '候选摘要对每项服务取(D+S,I+E)，同时保留价格、共享条件和弃选项。两版本的整张候选菜单在每题完全一致。八道任务对应八个二点纤维，总原始菜单数16；菜单对比算子秩8，三分类概率对比维度16。若每个菜单赋质量1/16、对比行归一化为单位长度，八个曝光特征值均为1/16。数值随归一化改变，不能和其他规范的特征值直接比较。',
    '本次充分性原假设允许八道候选菜单各有任意A/B/弃选概率，即16个自由概率参数；完整版本×任务模型有32个自由概率参数。它比5参数共同MNL的原假设更宽。这里的有限支撑证书就是纤维内可用对比的秩，属于经典lack-of-fit对比维数，不能包装为新的统计检验。证书只覆盖这16个菜单。',
    '## 整体检验和基线',
    '| 检验 | 统计量 | 999,999次整份答卷置换p | Monte Carlo 95%区间 |','| --- | ---: | ---: | --- |']
    labels=['学生化聚类Wald','完整纤维score／广义CMH形式','完整任务×版本复合LR']
    for label,(_,r) in zip(labels,tests.iterrows()): lines.append(f'| {label} | {r.statistic:.4f} | {r.permutation_p_plus_one:.6f} | [{r.mc_interval_low:.6f}, {r.mc_interval_high:.6f}] |')
    lines += [
    '学生化Wald针对逐题边际概率相等。在记录独立、协方差非退化等正则条件下，置换校准对该弱原假设具有渐近有效性。有限样本精确性要求更强的整份回答向量可交换，或相应的尖锐随机化原假设。仅有逐题概率相等并不保证联合分布相同。LR使用重复题目的复合偏差，采用整份答卷置换，未使用独立选择行的普通χ²参考。',
    f'进一步对每道2×3列联表枚举全部可行表，按多元超几何概率精确计算LR尾概率，再对8题做Holm和Bonferroni校正。全局Bonferroni p={exact["global_Bonferroni_p"]:.6f}，未拒绝。该方法只要求每题边际原假设和记录间独立，允许同一记录8题任意相关；代价是多重比较导致功效损失。每题条件概率总质量、版本互换不变性和0.01／0.05／0.10水平的精确拒绝质量均已枚举核验。它和整体置换针对的有限样本条件不同，两者一起报告。',
    '在本次平衡180∶180分配下，学生化Wald是固定总体协方差score的严格单调变换，两者所有置换排序相同。这解释了相同p值，也说明没有检验优势。该恒等关系已用100个随机划分与直接协方差计算核对，最大误差约1.4×10⁻¹³。完整LR得到相近结果。',
    f'随机森林在240条训练记录上学习方向，固定后在120条留出记录上检验。留出残差检验p={ml["p_plus_one"]:.4f}，对数预测分数平均改进为{ml["mean_heldout_logscore_gain"]:.5f}，区间跨0。同样120条的score和LR也未拒绝。留出比较样本量更小，不能据此宣称ML不如全样本检验。',
    '整体p接近0.05，只构成该假设情景下的边缘证据。没有通过后续换模型、换分组或反复选随机种子来争取显著结果。初始49,999次置换和500次校准仍保留，最终精度统一提高。所有模型、积分节点和留出方案的选择均公开。',
    '## 参数模型与重复选择',
    f'共同MNL月费系数为{price.estimate:.4f}/10元，答卷聚类95%区间为[{price.ci_low:.4f}, {price.ci_high:.4f}]。5、7、10参数MNL均通过独立聚合似然复算、解析梯度、数值梯度、Hessian正定性和概率归一性核对；独立估计的最大系数差小于8×10⁻⁸。',
    '面板模型让同一记录的8题共享一个正态服务参与截距，弃选效用取0；价格、共享系数固定。扩展模型曾加入负对数正态价格随机系数，其离散度命中下界。最终模型把该方差精确设为0，不将边界估计当作已发现的价格异质性。保留随机服务截距后模型无边界参数。',
    f'共同面板模型的服务截距标准差约为{np.exp(ri.estimate):.3f}。采用31、101、161点Gauss–Hermite积分核对，101→161点的最大系数变化小于3×10⁻⁸。每个模型用3个固定初值，得到相同目标值。',
    '| 模型 | 新答卷8题联合负对数分数÷8，越低越好 |','| --- | ---: |']
    for name,val in zip(['独立MNL乘积','随机服务截距','随机服务截距＋版本截距'],cv.joint_NLL_divided_by_8): lines.append(f'| {name} | {val:.6f} |')
    lines += [f'五折均整份留出；每个训练折独立使用固定的3个初值，未用留出答案或全样本估计初始化。随机截距改善的是完整8题向量的联合预测。版本项相对共同面板模型的平均额外改进为{gain.gain_vs_random_intercept_common:.5f}/题，描述性区间[{gain.descriptive_gain_vs_RI_ci_low:.5f}, {gain.descriptive_gain_vs_RI_ci_high:.5f}]，未形成稳定的额外预测提升。共同MNL的边际预测损失和面板联合分数是不同目标，报告没有把两者混作同一指标。',
    '参数化版本截距检验p约0.0054，依赖正确的logit和随机截距设定。它不能替代允许任意候选响应面的整体充分性检验，也不能用更小p值证明更强创新。交叉验证区间仅是固定折预测差的描述性区间，训练集重叠使其不等同于独立外部验证。',
    '## 零假设校准与功效',
    '追加了3个场景，每场景5,000次独立模拟，每次999次整份答卷置换：题内独立、同等相关，以及两版边际概率相同但相关结构不同。前两项检查联合原假设下的尺寸；第三项只用于检查学生化弱原假设推断，LR不获得弱原假设精确性声明。',
    f'初始500次模拟中，独立题场景Wald置换拒绝率为0.072；此值没有删去。增加至5,000次后的同场景拒绝率为{caljoint.rejection_rate:.4f}，95% Monte Carlo区间[{caljoint.mc_low:.4f}, {caljoint.mc_high:.4f}]。完整12行校准结果见null_calibration_extended.csv，不能把有限个场景模拟称为普遍有效性证明。',
    '在两版相关结构不同、边际仍相同的场景，学生化置换的拒绝率为0.0612（95% Monte Carlo区间约0.0549—0.0682），表明n=360时弱原假设尺寸仍有偏差。普通Wald χ²参考在三个场景约为0.078—0.086。故不能用“渐近稳健”给接近0.05的观测结果提供有限样本保证；逐题精确边际检验用于补足这一有效性缺口。',
    '普通χ²参考在部分场景存在有限样本偏差。最终功效表使用独立的5,000次零假设模拟生成各样本量临界值，再对每格进行1,000次独立模拟，共60格。它是给定生成机制下校准后的情景功效，不是各数据集的精确置换功效，也未用观测效应估算功效。初始χ²功效表仅作数值对照。',
    '![情景功效](prospective_power.png)',
    '## 对论文论点的审查',
    '本次能够支持的论点是：完整候选菜单保持可计算验证；在该有限支撑上可以构建纤维内概率对比；理想分配加上联合可交换条件时全量数据存在接近0.05的总体差异信号。有限样本有效的逐题边际检验未拒绝，当前结果没有确认边际充分性失败。不同记录的稳定服务参与倾向能改善重复选择的联合预测。',
    '仍有四个实质缺口。第一，D、S、I、E是研究者指定的等级编码，跨属性相加的等权压缩缺少独立的行为或应用依据。拒绝这个编码只能反驳这个具体摘要，不能推及所有低维效用表示。第二，四个原始属性随版本整组变化，13列模型只有秩7，无法区分哪一项导致差异，框架、复杂度和误差尺度解释也未被分离。第三，证书与经典lack-of-fit自由度相通，score、CMH和饱和LR没有显示独立方法优势。第四，统计模型不能核实招募和真人来源，理想分析无法补成已经实施的随机实验。',
    '建议把本文贡献收窄为“候选摘要压缩下的设计支撑审计与修复”，以现实中有应用依据的广义成本或复合属性为主要案例。进一步贡献必须来自真实常用设计的结构盲区、可执行的修复准则和代价收益比较。当前健康管理例子可作为有限支撑的探索性演示，现有证据不支持将其写成独立原创理论或投稿就绪的确认性实证。此处是基于当前材料的判断，不冒充独立同行评审。',
    '## 可复现材料和校验范围',
    '公开包包含脚本、题目配置、聚合计数、聚类一二阶矩、模型参数、校准与功效表。聚合矩可独立复算观测Wald和协方差；准确的答卷置换、面板似然及个体分折仍需本地受控原始数据。公开包不含IP、提交时间、答卷级8题向量或原始工作簿，因此不能宣称匿名公开包可完全替代原始数据。',
    '技术校验通过意味着实现满足列出的数值和代数检查；它不等于来源核实、原假设为真、模型全局正确、原创性成立或期刊接收。逐项完成状态见claim_and_completion_audit.csv。',
    '## 方法来源',
    '置换的联合原假设与参数弱原假设区分参照[Chung与Romano（2013）](https://arxiv.org/abs/1304.5939)。面板选择的混合积分采用[Train的Discrete Choice Methods with Simulation](https://emlab.berkeley.edu/books/choice2.html)中的标准建模思路。随机森林实现参照[scikit-learn文档](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)，扩展模型Sobol积分参照[SciPy文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html)。这些引用说明方法来源，不构成本文优先权证据。']
    blocks=[]
    for line in lines:
        if line.startswith('|') and blocks and blocks[-1].startswith('|'):
            blocks[-1]+='\n'+line
        else:
            blocks.append(line)
    (out/'全量模型与论点验证报告.md').write_text('\n\n'.join(blocks)+'\n')
    eng=f'''# DH36: analysis conditional on ideal assignment

This exploratory analysis uses all 360 response records, 180 per questionnaire version, retaining all eight choices per record. It conditions on independent records, complete 1:1 respondent-level random assignment and no interference. These assumptions define the analysis scenario; they are not verified recruitment facts. Original provenance is preserved. No claim that all records are verified human responses is made.

The candidate map combines the fixed profile levels as (D+S, I+E), retaining prices, sharing and the outside alternative. All eight paired menus preserve the complete candidate representation. There are 16 supported raw menus, eight two-point fibres, eight independent raw-menu contrasts and 16 independent categorical-probability contrasts. These are finite-support lack-of-fit dimensions. They are not a new testing principle or a guarantee of power beyond this support.

The studentized 16-dimensional cluster Wald statistic is {tests.statistic.iloc[0]:.4f}. Complete respondent-vector permutation with 999,999 draws gives p={tests.permutation_p_plus_one.iloc[0]:.6f} (Monte Carlo 95% interval [{tests.mc_interval_low.iloc[0]:.6f}, {tests.mc_interval_high.iloc[0]:.6f}]). The complete-fibre pooled score has exactly the same permutation ordering under balanced allocation. The saturated task-by-version composite deviance gives p={tests.permutation_p_plus_one.iloc[2]:.6f}. Whole-vector exchangeability provides finite-sample exactness; marginal equality alone supports the studentized test only asymptotically under regularity conditions. The ordinary independent-choice LR reference is not used.

A frozen random-forest residual direction trained on 240 records gives a one-sided heldout p={ml['p_plus_one']:.4f} on 120 records. The heldout logscore gain is {ml['mean_heldout_logscore_gain']:.5f} and its descriptive interval includes zero. Full-sample and heldout tests have different analysis sizes; these results do not establish power dominance.

The common MNL price coefficient is {price.estimate:.4f} per CNY 10 (cluster interval [{price.ci_low:.4f}, {price.ci_high:.4f}]). A random service-participation intercept improves five-fold joint response prediction. Joint negative log score divided by eight is {cv.joint_NLL_divided_by_8.iloc[0]:.6f} for the MNL product, {cv.joint_NLL_divided_by_8.iloc[1]:.6f} for the common random-intercept model and {cv.joint_NLL_divided_by_8.iloc[2]:.6f} with version constants. The incremental version gain is {gain.gain_vs_random_intercept_common:.5f} per task, with a descriptive interval spanning zero. All eight answers are held out together. Every training fold uses the same three fixed initial values, independently of heldout outcomes. An expanded random-price model reached the zero-variance boundary; the final model fixes that variance to zero. Quadrature at 101 and 161 nodes yields coefficient differences below 3e-8.

Calibration uses 5,000 independent datasets in each of three null/dependence scenarios and 999 complete-record permutations per dataset. Prospective power uses hypothetical probability shifts, independent 5,000-null-dataset cutoffs and 1,000 datasets per cell. It is not observed power or a universal sample-size recommendation. At n=360, an eight-task A/B probability shift of five percentage points has simulated power {pw('all_tasks_A_vs_B',360,.05):.3f}, while an eight-percentage-point shift has power {pw('all_tasks_A_vs_B',360,.08):.3f}, conditional on the stated calibration.

For the marginal null with arbitrary within-record dependence, exact conditional 2x3 tests were enumerated separately for each task and combined using Bonferroni/Holm. The global Bonferroni p-value is {exact['global_Bonferroni_p']:.6f}; no task passes familywise correction. This check is finite-sample valid under independent records and the taskwise common-probability null, without joint vector exchangeability. The omnibus studentized permutation test rejects in 0.0612 of 5,000 unequal-dependence weak-null simulations, so its asymptotic robustness is not treated as a finite-sample guarantee at n=360. The global permutation and exact marginal tests have different finite-sample assumptions and are both disclosed.

The evidence is borderline under the stronger joint exchangeability scenario and is not consistently supported in heldout data. Marginal sufficiency failure has not been confirmed by the finite-sample-valid check. The raw attributes are bundled by version, so their individual effects and competing framing, complexity or scale mechanisms are not separately identified. Equal-weight sums of attribute codes require an application-based justification independent of these outcomes. The results do not establish originality, superiority over classical lack-of-fit methods, human provenance, actual random allocation or readiness for Journal of Choice Modelling submission.

Permutation inference follows the distinction in [Chung and Romano (2013)](https://arxiv.org/abs/1304.5939); panel mixed logit follows [Train](https://emlab.berkeley.edu/books/choice2.html). Public aggregates reproduce the observed Wald statistic, but exact respondent-level permutations and integrated panel likelihoods require controlled access to the original response vectors. Raw exports and participant-level responses are not included in this release.
'''
    (out/'conditional_empirical_section.md').write_text(eng)
    import scipy,sklearn,openpyxl,threadpoolctl
    versions={'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,
              'scikit-learn':sklearn.__version__,'matplotlib':matplotlib.__version__,
              'openpyxl':openpyxl.__version__,'threadpoolctl':threadpoolctl.__version__}
    (out/'requirements.txt').write_text(''.join(f'{name}=={version}\n' for name,version in versions.items()))
    source=Path(__file__).parent
    for name in ['dh36_observational_analysis.py','dh36_panel_model.py','dh36_assumption_conditioned_validation.py','dh36_finalize_validation.py','dh36_exact_marginal_checks.py','dh36_report_validation.py']: shutil.copy2(source/name,out/name)
    shutil.copy2(args.instrument,out/'instrument.json')
    readme='''# DH36 ideal-assignment scenario: 2026-11-09

Read `全量模型与论点验证报告.md` and `assumptions.json` first. All 360 exported records enter the full-sample models. Raw provenance remains unchanged. Ideal independent records and complete 180:180 random assignment are assumptions, not certified facts.

## Reproduction

Python 3.12 and packages in requirements.txt are required. Place controlled raw exports at PRIVATE/V1_raw.xlsx and PRIVATE/V2_raw.xlsx. That directory must be writable because private fold/split files are stored there. It is never uploaded.

```sh
python dh36_assumption_conditioned_validation.py --input-dir PRIVATE --instrument instrument.json --out RESULTS --permutations 49999 --mc-repetitions 500 --calibration-permutations 199
python dh36_finalize_validation.py --input-dir PRIVATE --instrument instrument.json --out RESULTS --permutations 999999 --null-repetitions 5000 --null-permutations 999 --power-repetitions 1000
python dh36_exact_marginal_checks.py --input-dir PRIVATE --instrument instrument.json --out RESULTS
python dh36_report_validation.py --input-dir PRIVATE --instrument instrument.json --out RESULTS
```

The first stage preserves exploratory numerical outputs. The second stage supplies the final high-precision tests, boundary-repaired panel model, independent-start panel CV and extended calibration. The initial mixed-price model and initial chi-square power table are diagnostics, not the final models or calibrated power estimates. The first-stage panel CV used full-data fitted parameters as numerical starting values; it is retained only as a diagnostic. The final independent_start_panel_joint_CV.csv uses three fixed starting values in every training fold and supersedes that result.

Read randomization_tests_high_precision.csv, exact_marginal_task_tests.csv, independent_start_panel_joint_CV.csv, random_intercept_model_coefficients.csv, null_calibration_extended.csv and power_DGP_calibrated.csv for final results. Finite-sample exactness under joint exchangeability is stronger than the marginal sufficiency null. The exact taskwise Bonferroni/Holm analysis addresses the marginal null under arbitrary within-record dependence. Prospective power is conditional on a simulated DGP, not a guarantee for recruitment.

All outputs are aggregate-only. aggregate_cluster_moments.json permits an independent reconstruction of the observed Wald statistic; it does not encode the full participant response distribution needed for exact permutations. Raw exports, IPs, timestamps, identifiers and individual answer vectors remain private. Files referencing private input directories describe required local inputs; those directories are intentionally not part of the release.
'''
    (out/'README.md').write_text(readme)
    check={'public_aggregate_Wald':w,'matches_full_sample_statistic':True,'power_cells':60,'extended_null_rows':12,'final_panel_CV_models':3,'raw_hashes_unchanged':final['raw_hashes_unchanged'],'numerical_scope':'listed models and checks only; no factual recruitment or priority certification','unsupported_claims_not_promoted':True}
    js(out/'release_verification.json',check)
    manifest=[]
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name not in ['MANIFEST.sha256','github_sync_receipt.json']: manifest.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name)
    (out/'MANIFEST.sha256').write_text('\n'.join(manifest)+'\n')
    target=out.parent/'DH36_全量模型技术验证_20261109.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out.iterdir()):
            if p.is_file(): zf.write(p,arcname=out.name+'/'+p.name)
    print('REPORT_AND_PACKAGE_COMPLETE',target,flush=True)

if __name__=='__main__': main()
