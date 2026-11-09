"""Create aggregate questionnaire-reuse outputs from the executed analyses."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ANALYSIS=Path(__file__).resolve().parent

def table(df):
    cols=list(df.columns); rows=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for row in df.itertuples(index=False,name=None): rows.append('| '+' | '.join(f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',type=Path,required=True); ap.add_argument('--instrument',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args(); OUT=a.out
    pooled=pd.read_csv(OUT/'panel_model_comparison.csv'); follow=pd.read_csv(OUT/'source_boundary_comparison.csv')
    checks=json.loads((OUT/'panel_model_fits.json').read_text()); sourcechecks=json.loads((OUT/'source_boundary_model_fits.json').read_text())
    c=pd.concat([pd.read_csv(OUT/'panel_same_price_rule_contrasts.csv'),pd.read_csv(OUT/'source_boundary_same_price_contrasts.csv')],ignore_index=True)
    c['population']=c.population.replace({'300_records_without_AI_source_label':'300_other_source_labels'})
    c.to_csv(OUT/'consolidated_same_price_contrasts.csv',index=False)
    pp=pooled[['model','parameters','fivefold_joint_nll_per_choice','fivefold_predicted_all_outside']]
    ss=follow[['population','model','fivefold_joint_nll_per_choice','observed_all_outside']]
    cc=c[['population','model','utility_difference','ci_low','ci_high']]
    # Publication-style figures: actual support and observed aggregate counts.
    support=pd.read_csv(OUT/'exact_fee_sharing_support.csv'); prices=sorted(support.price.unique()); rules=['G0','G1','G2']
    mat=np.array([[len(support[(support.sharing==g)&(support.price==p)]) for p in prices] for g in rules])
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),gridspec_kw={'width_ratios':[1.4,1]})
    ax=axes[0]; ax.imshow(mat>0,cmap='Blues',vmin=0,vmax=1,aspect='auto'); ax.set_xticks(range(len(prices)),prices); ax.set_yticks(range(3),rules); ax.set_xlabel('Monthly fee (CNY)'); ax.set_title('Observed price-sharing cells')
    for i in range(3):
        for j in range(len(prices)):
            if mat[i,j]: ax.text(j,i,str(mat[i,j]),ha='center',va='center',color='white',fontweight='bold')
    ax=axes[1]; ax.bar([0,1],[42/60,0],color=['#377eb8','#a6bddb']); ax.set_xticks([0,1],['Export label: AI samples\n(n = 60)','Other source labels\n(n = 300)']); ax.set_ylim(0,1); ax.set_ylabel('Fraction choosing outside in all 8 tasks'); ax.set_title('Observed source concentration'); ax.text(0,.73,'42 / 60',ha='center'); ax.text(1,.04,'0 / 300',ha='center')
    fig.text(.02,.02,'Cell numbers: card appearances per form. Source labels retained; task-specific respondent provenance unresolved.',fontsize=8)
    fig.tight_layout(rect=[0,.06,1,1]); fig.savefig(OUT/'support_and_source_audit.svg'); fig.savefig(OUT/'support_and_source_audit.png',dpi=170); plt.close(fig)
    report=f'''# 问卷保留与模型重建结果

2026年11月9日。原始360条记录、两版问卷内容、历次代码和结果全部保留。当前可以形成探索性应用实证及可复核的设计审计；独立创新与正式投稿条件尚未同时得到支持。

## 现在采用的研究问题

付费健康管理服务中，数据共享规则与服务选择如何关联？“由用户决定是否共享、共享给谁”与“去除身份信息后提供给科研机构”在模型调整后的同一月费下如何比较？持续不购买记录和招募来源如何影响这一比较？

这个问题直接使用原有8道选择题和已有背景题。两版各180条，共360条记录、2,880次选择。每条记录的8次选择一起进入估计和验证，不能当作2,880个独立受访者。优先级题用于描述，不作随机机制或外生中介。

## 来源任务已核验，具体服务含义待核对

你反馈“平台说明为真人招募答卷”。已登录平台的系统通知确认两版各有一项30份的“AI样本”任务；三页点数记录共60条，两版各30条，每条消耗0.5点。该真人招募说明是否对应这两项任务尚待核实。主分析保留全部360条，原始导出中的60条“AI样本”标签保持不变；来源名称、点数记录和回答模式均不能单独核验真人身份或生成来源。详见[来源核验更新](../platform_source_verification_20261109/问卷星来源核验更新.md)。

42条“8题全部不购买”记录全部属于这60条来源组（版本1为22/30，版本2为20/30）；其余300条为0/300。这个集中现象要求来源敏感性分析。它本身不能证明答卷无效、生成来源或某个消费者群体的动机。请保留平台服务说明、订单或客服解释，供以后准确填写招募方法使用，无需提供参与者身份信息。

实际发放时，不同渠道分别填写一个版本，未保存分配记录。应用稿采用版本／渠道关联解释，不写成已核验的两组随机实验，也不作原来的充分性因果检验。

## 价格和规则实际能比较到哪里

不向外部共享（G0）只与79、109元共同出现；去标识后向科研机构共享（G1）与69、89、99元共同出现；用户决定的规则包（G2）与59、89、119元共同出现。G1与G2在89元有共同价格，G0与其他规则没有共同价格。

自由价格水平加规则主效应共有12列、有效秩11；其中存在一条精确的混淆方向。这里的价格—规则支撑有8条边、2个连通分量且无环，主效应模型可表示全部已观察单元，不能用这些单元检验一般价格—规则交互。原来四个服务卡属性的变动仍与版本／方案绑定，不能分开估计其因果贡献。

主对比改为G2与G1在89元的效用差。它由跨题结构模型识别，仍要求方案／版本截距及效用跨题稳定；没有一道任务把这两个89元方案直接同时对置，不能称为直接随机同价实验。删去第2题或第5题后，这个自由单元对比即失去识别。约15元的原支付意愿保留为线性价格条件下的附加估计，不作为设计本身支持的确定金额。

## 已完成的模型比较

主比较冻结后运行了线性价格、二次价格、自由价格—规则单元、随机服务截距、两类参与倾向、两类尺度和两类偏好模型。自由单元模型另作条件A/B估计。两类模型到达全程不购买边界后，增补经典的全程不购买点质量模型，避免用任意巨大截距冒充稳定偏好类型。该增补是看过结果后的探索性分析修订。

所有验证沿用同一组按整条答卷划分的五折。下表的联合负对数预测分数越小越好；它评价8次回答的联合分布，不能等同于下一次选择的边际预测表现。这些折已经用于前期探索，属于内部验证。

{table(pp)}

全部360条中，全程不购买成分几乎解释了复杂偏好混合模型的全部改善。偏好两类与参与两类的联合分数几乎相同，前者还存在近零曲率和参数边界，不能命名为稳定的人群偏好类型。尺度两类的最优解到达边界，也不作为行为发现。

{table(ss)}

其余300条的敏感性分析中，普通模型、随机截距和两类模型没有显示更复杂模型的预测改善。不同样本总体的绝对分数不可横向当作优劣排名；这里只在同一总体内比较模型。全程不购买模式的集中来源是解释模型差异必须保留的证据。

## 同价规则比较的证据强度

下表为G2−G1在89元的效用差及记录聚类区间。不同模型的效用尺度可能变化，系数大小不能直接跨模型比较；判断重点是方向、区间与假设。多个模型和对比均属探索性，未作选择后推断或多重检验保证。

{table(cc)}

估计方向均为正；处理持续参与差异的全样本模型以及条件A/B模型的区间包含零。其余300条的自由单元普通／随机截距模型区间不含零，但条件A/B区间仍包含零。不能挑选某个显著模型作为结论。现阶段可报告正向关联信号和模型依赖性，不能证明用户控制的纯机制，也不能宣布稳定的金额溢价。

## 问卷内容如何完整保留

| 已有内容／成果 | 当前用途 | 写作处理 |
| --- | --- | --- |
| 8道A/B/不购买题 | 选择概率、价格和共享规则包的探索性关联 | 全部进入主分析，按记录聚类 |
| 全程不购买答题模式 | 参与结构与来源敏感性 | 保留全部记录，不按结果删答卷 |
| 应用使用、付费经历 | 样本描述与预先明确的探索性分层 | 无人口代表性或因果机制声明 |
| 服务卡固定属性 | 说明方案组合与识别条件 | 保留卡片，不拆出单独属性效应 |
| 两版问卷 | 版本／渠道调整与分组描述 | 不作已核验的随机版本效应 |
| 旧理论、经典等价证明、模拟代码 | 设计审计和复现资源 | 不重新命名为原创理论 |

## 稿件如何推进

应用稿主线按“服务选择问题—真实价格支撑—参与和来源处理—同价规则比较—业务解释”组织。最接近文献已研究健康应用价格／隐私、健康数据治理和偏好分群，因此这三个标签本身不能承担创新点。独立贡献仍需一个有价值且由证据稳定支持的具体结论；增加模型数量不能代替它。

先完成可核对的应用实证稿，整理平台招募说明和实际适用的研究／伦理说明。现有答卷不必重新发放。是否补充验证与怎样修改题目，应由尚待证明的具体问题决定；目前不建议按原设计继续购买相同答卷来解决创新或识别问题。

同行研究对应关系见[最接近文献核查](closest_prior_work.md)。当前研究稿见[应用实证初稿](application_evidence_draft.md)。数值与图表均来自本轮实际运行，未把模型收敛或自评当作期刊录用证据。
'''
    (OUT/'问卷保留与模型重建结果.md').write_text(report)
    (OUT/'README.md').write_text('''# Exploratory application rebuild — 2026-11-09

This package preserves the questionnaire and evaluates an applied question. It makes no new-method, verified-randomization or submission-readiness claim. Recruitment is recorded as the researcher's report of a platform human-recruitment description. Authenticated inspection confirms two 30-record AI-sample task notices and 60 matching point-log entries; the task-specific respondent provenance remains unresolved. See the [source-verification update](../platform_source_verification_20261109/README.md). Original source labels remain in the private exports. Public files contain aggregate results and code only.

Run from repository root with scientific Python dependencies numpy, pandas, scipy, matplotlib, openpyxl, threadpoolctl. Raw exports and fold assignments remain private. `PRIVATE_DIR` must contain V1_raw.xlsx, V2_raw.xlsx and ideal_cv_folds_private.csv. The historic fold filename does not certify ideal randomization. A private access holder can reproduce the estimates; an independent public download cannot recover private records or certify respondent provenance.

```sh
python analysis/application_support_models.py --input-dir "$PRIVATE_DIR" --instrument survey/public_dh36/instrument.json --out reports/application_rebuild_20261109
python analysis/application_panel_comparison.py --input-dir "$PRIVATE_DIR" --instrument survey/public_dh36/instrument.json --folds "$PRIVATE_DIR/ideal_cv_folds_private.csv" --out reports/application_rebuild_20261109
python analysis/application_source_diagnostic.py --input-dir "$PRIVATE_DIR" --instrument survey/public_dh36/instrument.json --folds "$PRIVATE_DIR/ideal_cv_folds_private.csv" --out reports/application_rebuild_20261109
python analysis/write_application_rebuild_report.py --input-dir "$PRIVATE_DIR" --instrument survey/public_dh36/instrument.json --out reports/application_rebuild_20261109
```

The second command implements the fixed eight-model family; the third implements the documented boundary/source follow-up. Five folds keep each record's eight choices together. Joint and marginal prediction targets are separate. Mixture prediction uses population weights for held-out records. Hessian/boundary diagnostics are retained. Classical full-outside mixtures are not claimed as independent innovation.

All p-values/intervals are exploratory and model conditional. Pooled results are accompanied by a 300-record source sensitivity, not a silent exclusion. The latter sample is defined by original source labels and is not independently verified as representative.

The figure's “AI samples” text is a literal translation of the original export label, not a finding of machine generation. The researcher reports platform recruitment of humans.
''')
    # Verify public table totals and likelihood computations against saved fits.
    import sys
    sys.path.insert(0,str(ANALYSIS))
    from application_panel_comparison import Panel
    from application_support_models import build,save
    import dh36_observational_analysis as old
    spec=json.loads(a.instrument.read_text()); d,provenance,*_=old.read_exports(a.input_dir,spec); mm,*_=build(spec,d); y=d[[f'q{i}' for i in range(1,9)]].to_numpy(int)-1
    descriptions=[]
    for population,use in [('all_360',np.ones(len(d),bool)),('300_other_source_labels',~d.ai_label.to_numpy()),('60_original_AI_source_labels',d.ai_label.to_numpy())]:
        for variable in [f'q{i}' for i in range(1,9)]+['app_use','paid_service','priority']:
            n=int(use.sum())
            for code,count in d.loc[use,variable].value_counts().sort_index().items(): descriptions.append({'population':population,'variable':variable,'response_code':int(code),'count':int(count),'denominator':n,'fraction':float(count/n)})
    pd.DataFrame(descriptions).to_csv(OUT/'choice_and_background_counts.csv',index=False)
    verify={}
    for label,form,kind in [('MNL_linear','linear','MNL'),('RI_linear','linear','RI'),('RI_cells','cells','RI'),('participation_2class','linear','participation'),('scale_2class','linear','scale'),('taste_2class','linear','taste')]:
        ll=-Panel(mm[form],y,kind).eval(np.array(checks[label]['beta']))[0]; err=abs(ll-checks[label]['checks']['loglik']); assert err<1e-7; verify[label+'_saved_likelihood_error']=err
    assert int((y==2).all(1).sum())==42 and not (y[~d.ai_label.to_numpy()]==2).all(1).any()
    verify['raw_file_hashes']=[{'version':v['version'],'sha256':v['sha256']} for v in provenance]
    verify['records']=len(y); verify['choice_rows']=y.size; verify['raw_and_person_level_data_exported']=False
    verify['source_description']='Researcher reports human recruitment; authenticated task notices and AI-point logs confirm the two 30-record tasks, but their task-specific respondent provenance remains unresolved. Original AI样本 labels retained.'
    save(OUT/'numerical_validation_summary.json',verify)
    print('Report, figures and aggregate validation written.',flush=True)

if __name__=='__main__': main()
