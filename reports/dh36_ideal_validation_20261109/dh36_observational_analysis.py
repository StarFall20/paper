"""Analyse DH36 exports while retaining provenance and respondent clusters.

Recruitment channels and questionnaire versions coincide. Results describe
cohort associations and predictive fit; they are not randomized sufficiency
tests. AI-labelled records remain labelled in every analysis population.
Raw and participant-level outputs are written only to the private directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import chi2, norm


MODEL_NAMES = {
    'mnl_common': '共同价格与共享参数',
    'mnl_cohort_intercepts': '加入版本／渠道截距',
    'mnl_cohort_all': '两组分别估计参数',
    'task_common': '逐题共同概率',
    'task_cohort': '逐题分组概率',
}


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def read_exports(input_dir, spec):
    people, provenance, audit = [], [], []
    duplicate_flags, non_ai_ips = [], {}
    for version, activity in [('V1', 390522225), ('V2', 390535906)]:
        path = input_dir/f'{version}_raw.xlsx'
        raw = pd.read_excel(path)
        cols = list(raw.columns)
        questions = [next(c for c in cols if re.match(rf'^第{t}题\s', str(c))) for t in range(1, 9)]
        common = [next(c for c in cols if re.match(rf'^{q}\.\s', str(c))) for q in (9, 10, 11)]
        d = pd.DataFrame({
            'participant_id': version+'_'+raw['序号'].astype(str),
            'version': version, 'version2': int(version=='V2'),
            'export_row_id': raw['序号'],
            'submitted_at': pd.to_datetime(raw['提交答卷时间']),
            'duration_seconds': pd.to_numeric(raw['所用时间'].astype(str).str.extract(r'(\d+)')[0]),
            'platform_source': raw['来源'].fillna('未提供').astype(str),
            'platform_source_detail': raw['来源详情'].fillna('未提供').astype(str),
        })
        for t, c in enumerate(questions, 1):
            d[f'q{t}'] = pd.to_numeric(raw[c], errors='raise')
            if not d[f'q{t}'].isin([1, 2, 3]).all():
                raise ValueError(f'Invalid choice code in {version} q{t}')
        for key, c in zip(['app_use', 'paid_service', 'priority'], common):
            d[key] = pd.to_numeric(raw[c], errors='raise')
            allowed = range(1, 9) if key=='priority' else range(1, 6)
            if not d[key].isin(allowed).all():
                raise ValueError(f'Invalid background code: {version} {key}')
        d['ai_label'] = d.platform_source.str.contains('AI样本', regex=False)
        d['before_bulk_collection_date'] = d.submitted_at.dt.date.astype(str) < '2026-11-08'
        d['under_60_seconds'] = d.duration_seconds < 60
        d['one_choice_repeated'] = d[[f'q{i}' for i in range(1, 9)]].nunique(axis=1)==1
        # IP addresses never leave the original private exports.
        ip = raw['来自IP'].fillna('').astype(str).str.extract(r'(\d+\.\d+\.\d+\.\d+)')[0]
        nonmissing = ip.notna()
        duplicate_ip_rows = int((ip.duplicated(keep=False) & nonmissing).sum())
        for label, selection in [('all',np.ones(len(d),bool)),('non_AI',~d.ai_label)]:
            values = ip[selection].dropna()
            duplicate_flags.append({'version':version,'population':label,
                                    'rows_with_duplicate_IP':int(values.duplicated(keep=False).sum()),
                                    'records':int(np.sum(selection))})
        non_ai_ips[version] = set(ip[~d.ai_label].dropna())
        if d.participant_id.duplicated().any():
            raise ValueError('Duplicate export row identifiers')
        people.append(d)
        provenance.append({'version': version, 'activity_id': activity,
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'rows': len(raw), 'columns': len(cols)})
        for label, g in d.groupby('platform_source', dropna=False):
            audit.append({'version': version, 'platform_source': label, 'records': len(g),
                          'duration_min': int(g.duration_seconds.min()),
                          'duration_median': float(g.duration_seconds.median()),
                          'duration_max': int(g.duration_seconds.max()),
                          'under_60_seconds': int(g.under_60_seconds.sum()),
                          'repeated_one_choice': int(g.one_choice_repeated.sum())})
        provenance[-1]['duplicate_ip_rows_flag_only'] = duplicate_ip_rows
    duplicates={'within_form':duplicate_flags,
                'cross_form_non_AI_IP_overlap':len(non_ai_ips['V1']&non_ai_ips['V2']),
                'IP_matches_are_flags_not_person_identity_or_exclusion_rules':True}
    return pd.concat(people, ignore_index=True), provenance, pd.DataFrame(audit), duplicates


def menu_arrays(spec):
    base = np.zeros((8, 3, 5))
    for t, task in enumerate(spec['tasks']):
        for j in range(2):
            base[t, j, j] = 1
            base[t, j, 2] = (task['prices'][j]-89)/10
            base[t, j, 3] = task['sharing'][j]=='G1'
            base[t, j, 4] = task['sharing'][j]=='G2'
    for j, alt in enumerate(['A', 'B']):
        profiles = [spec['profiles'][v][alt] for v in ['V1', 'V2']]
        sums = [(int(c[0][1:])+int(c[1][1:]), int(c[2][1:])+int(c[3][1:])) for c in profiles]
        assert sums[0]==sums[1]
    return base


def design(d, base, name):
    n = len(d)
    x = np.broadcast_to(base, (n, 8, 3, 5)).copy().reshape(n*8, 3, 5)
    if name == 'mnl_common':
        names = ['ASC_A', 'ASC_B', 'price_per_10_CNY', 'G1_vs_G0', 'G2_vs_G0']
    elif name in ['mnl_cohort_intercepts', 'mnl_cohort_all']:
        sub = x[:, :, :2] if name=='mnl_cohort_intercepts' else x
        v = np.repeat(d.version2.to_numpy(), 8)[:, None, None]
        x = np.concatenate([x, sub*v], axis=2)
        names = ['ASC_A', 'ASC_B', 'price_per_10_CNY', 'G1_vs_G0', 'G2_vs_G0']
        names += ['V2_x_'+s for s in names[:sub.shape[-1]]]
    else:
        raise ValueError(name)
    return x, names


def objective(b, x, y):
    u = np.einsum('rjp,p->rj', x, b)
    lp = u-logsumexp(u, axis=1, keepdims=True)
    p = np.exp(lp)
    ex = np.einsum('rj,rjp->rp', p, x)
    score = x[np.arange(len(y)), y]-ex
    return -float(lp[np.arange(len(y)), y].sum()), -score.sum(axis=0)


def fit_mnl(d, base, name, validation=False):
    x, names = design(d, base, name)
    y = d[[f'q{i}' for i in range(1, 9)]].to_numpy(int).reshape(-1)-1
    rank = int(np.linalg.matrix_rank(x.reshape(-1, x.shape[-1])))
    if rank != x.shape[-1]:
        raise ValueError(f'Rank deficient {name}: {rank}/{x.shape[-1]}')
    if validation:
        b0 = np.linspace(-.08, .08, x.shape[-1]); eps = 1e-5
        fd = np.array([(objective(b0+np.eye(len(b0))[j]*eps, x, y)[0]-objective(b0-np.eye(len(b0))[j]*eps, x, y)[0])/(2*eps) for j in range(len(b0))])
        if np.max(np.abs(fd-objective(b0, x, y)[1]))>1e-5:
            raise ValueError('Analytical gradient failed independent finite difference check')
    opt = minimize(objective, np.zeros(x.shape[-1]), args=(x, y), jac=True,
                   method='BFGS', options={'gtol': 1e-7, 'maxiter': 1500})
    # A few exact Newton steps remove BFGS precision-loss residuals without
    # relaxing the convergence tolerance or changing the likelihood.
    refinements = 0
    for _ in range(6):
        value, g = objective(opt.x, x, y)
        if np.max(np.abs(g))<=1e-7:
            break
        u0 = np.einsum('rjp,p->rj', x, opt.x)
        p0 = np.exp(u0-logsumexp(u0, axis=1, keepdims=True))
        ex0 = np.einsum('rj,rjp->rp', p0, x)
        h0 = np.einsum('rj,rjp,rjq->pq', p0, x, x)-ex0.T@ex0
        step = np.linalg.solve(h0, g)
        for fraction in [1., .5, .25, .125, .0625]:
            candidate = opt.x-fraction*step
            if objective(candidate, x, y)[0]<=value+1e-9:
                opt.x = candidate; refinements += 1
                break
        else:
            break
    opt.fun = objective(opt.x, x, y)[0]
    grad = float(np.max(np.abs(objective(opt.x, x, y)[1])))
    if grad>1e-5:
        raise ValueError(f'Optimizer gradient too large: {name} {grad}')
    u = np.einsum('rjp,p->rj', x, opt.x)
    p = np.exp(u-logsumexp(u, axis=1, keepdims=True))
    ex = np.einsum('rj,rjp->rp', p, x)
    h = np.einsum('rj,rjp,rjq->pq', p, x, x)-ex.T@ex
    h_inv = np.linalg.inv(h)
    score_rows = x[np.arange(len(y)), y]-ex
    clusters = score_rows.reshape(len(d), 8, -1).sum(axis=1)
    correction = len(d)/(len(d)-1)*(len(y)-1)/(len(y)-len(names))
    cov = correction*h_inv@(clusters.T@clusters)@h_inv
    se = np.sqrt(np.diag(cov))
    rows = [{'term': s, 'estimate': float(b), 'cluster_se': float(e),
             'ci_low': float(b-1.95996398454*e), 'ci_high': float(b+1.95996398454*e),
             'p_value_association_only': float(2*norm.sf(abs(b/e)))}
            for s, b, e in zip(names, opt.x, se)]
    test = None
    if len(names)>5:
        delta = opt.x[5:]; dcov = cov[5:, 5:]
        w = float(delta@np.linalg.pinv(dcov)@delta)
        test = {'wald': w, 'df': len(delta), 'p_value_association_only': float(chi2.sf(w, len(delta)))}
    return {'beta': opt.x, 'cov': cov, 'parameters': rows, 'test': test,
            'rank': rank, 'gradient_max': grad,
            'solver_success': bool(opt.success or (refinements>0 and grad<=1e-7)),
            'newton_refinements': refinements,
            'composite_log_likelihood': -float(opt.fun), 'probabilities': p.reshape(len(d), 8, 3)}


def contrasts(d, repetitions, seed):
    y = d[[f'q{i}' for i in range(1, 9)]].to_numpy(int)-1
    scores = (y[:, :, None]==np.arange(3)[None, None, :]).astype(float).reshape(len(d), 24)
    z = d.version2.to_numpy()
    a, b = scores[z==0], scores[z==1]
    delta = b.mean(axis=0)-a.mean(axis=0)
    covariance = np.cov(a, rowvar=False, ddof=1)/len(a)+np.cov(b, rowvar=False, ddof=1)/len(b)
    se = np.sqrt(np.diag(covariance))
    keep = np.array([t*3+j for t in range(8) for j in range(2)])
    cov16 = covariance[np.ix_(keep, keep)]
    inv16 = np.linalg.pinv(cov16)
    rank = int(np.linalg.matrix_rank(cov16))
    w = float(delta[keep]@inv16@delta[keep])
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(0, repetitions, 500):
        count = min(500, repetitions-len(boot)*500)
        wa = rng.multinomial(len(a), np.ones(len(a))/len(a), size=count)
        wb = rng.multinomial(len(b), np.ones(len(b))/len(b), size=count)
        boot.append(wb@b/len(b)-wa@a/len(a)-delta)
    errors = np.concatenate(boot, axis=0)
    max_t = np.max(np.abs(errors/np.maximum(se, 1e-12)), axis=1)
    critical = float(np.quantile(max_t, .95))
    bw = np.einsum('bi,ij,bj->b', errors[:, keep], inv16, errors[:, keep])
    bootstrap_p = float((1+np.sum(bw>=w))/(1+repetitions))
    rows, task_p = [], []
    for t in range(8):
        k = [t*3, t*3+1]
        dt = delta[k]; ct = covariance[np.ix_(k, k)]
        wt = float(dt@np.linalg.pinv(ct)@dt)
        task_p.append(float(chi2.sf(wt, 2)))
        for j, alt in enumerate(['A', 'B', 'opt_out']):
            k = t*3+j
            rows.append({'task': t+1, 'alternative': alt,
                         'n_V1': len(a), 'n_V2': len(b),
                         'count_V1': int(a[:, k].sum()), 'count_V2': int(b[:, k].sum()),
                         'prob_V1': float(a[:, k].mean()), 'prob_V2': float(b[:, k].mean()),
                         'difference_V2_minus_V1': float(delta[k]),
                         'se': float(se[k]), 'ci95_low': float(delta[k]-1.96*se[k]),
                         'ci95_high': float(delta[k]+1.96*se[k]),
                         'simultaneous_low': float(delta[k]-critical*se[k]),
                         'simultaneous_high': float(delta[k]+critical*se[k])})
    order = np.argsort(task_p); adjusted = np.zeros(8); prior = 0.
    for pos, idx in enumerate(order):
        prior = max(prior, min(1., (8-pos)*task_p[idx])); adjusted[idx] = prior
    for row in rows:
        row['task_p_association_only'] = task_p[row['task']-1]
        row['task_p_holm_8'] = float(adjusted[row['task']-1])
    return pd.DataFrame(rows), {'wald': w, 'df': rank,
            'p_asymptotic_association_only': float(chi2.sf(w, rank)),
            'p_centered_cluster_bootstrap_association_only': bootstrap_p,
            'bootstrap_repetitions': repetitions, 'simultaneous_95_critical': critical,
            'n_V1': len(a), 'n_V2': len(b)}


def predictive_validation(d, base, seed):
    rng = np.random.default_rng(seed)
    folds = np.zeros(len(d), int)
    for v in [0, 1]:
        idx = np.where(d.version2.to_numpy()==v)[0]; rng.shuffle(idx)
        for k, part in enumerate(np.array_split(idx, 5)):
            folds[part] = k
    y_all = d[[f'q{i}' for i in range(1, 9)]].to_numpy(int)-1
    losses = {name: np.zeros(len(d)) for name in MODEL_NAMES}
    for k in range(5):
        train, test = folds!=k, folds==k
        dt, dv = d.loc[train], d.loc[test]
        yt, yv = y_all[train], y_all[test]
        assert set(dt.participant_id).isdisjoint(dv.participant_id)
        for name in ['mnl_common', 'mnl_cohort_intercepts', 'mnl_cohort_all']:
            fit = fit_mnl(dt, base, name)
            x, _ = design(dv, base, name)
            u = np.einsum('rjp,p->rj', x, fit['beta'])
            lp = (u-logsumexp(u, axis=1, keepdims=True)).reshape(len(dv), 8, 3)
            losses[name][test] = -np.take_along_axis(lp, yv[:, :, None], axis=2)[:, :, 0].mean(axis=1)
        for name in ['task_common', 'task_cohort']:
            probabilities = np.zeros((len(dv), 8, 3))
            for group in ([0] if name=='task_common' else [0, 1]):
                sub = np.ones(len(dt), bool) if name=='task_common' else dt.version2.to_numpy()==group
                counts = (yt[sub, :, None]==np.arange(3)[None, None, :]).sum(axis=0)
                p = (counts+.5)/(len(yt[sub])+1.5)
                selection = np.ones(len(dv), bool) if name=='task_common' else dv.version2.to_numpy()==group
                probabilities[selection] = p
            losses[name][test] = -np.log(np.take_along_axis(probabilities, yv[:, :, None], axis=2)[:, :, 0]).mean(axis=1)
    rows = []
    for name in MODEL_NAMES:
        diff = losses[name]-losses['mnl_common']
        e = float(diff.std(ddof=1)/np.sqrt(len(diff)))
        rows.append({'model': name, 'label': MODEL_NAMES[name],
                     'cv_mean_log_loss': float(losses[name].mean()),
                     'difference_vs_common_MNL': float(diff.mean()),
                     'paired_ci_low': float(diff.mean()-1.96*e),
                     'paired_ci_high': float(diff.mean()+1.96*e),
                     'folds': 5, 'split_unit': 'respondent', 'n': len(d)})
    return pd.DataFrame(rows), folds


def rank_audit(spec):
    rows = []
    for v in [0, 1]:
        for task in spec['tasks']:
            for j, alt in enumerate(['A', 'B']):
                d, s, i, e = [int(c[1:]) for c in spec['profiles']['V'+str(v+1)][alt]]
                rows.append([j==0, j==1, v*(j==0), v*(j==1),
                             d==1, d==2, s==2, i==1, e==1, e==2,
                             (task['prices'][j]-89)/10, task['sharing'][j]=='G1', task['sharing'][j]=='G2'])
    x = np.array(rows, float)
    return {'raw_model_columns': x.shape[1], 'raw_model_rank': int(np.linalg.matrix_rank(x)),
            'unidentified_dimensions': int(x.shape[1]-np.linalg.matrix_rank(x)),
            'channel_version_columns': 2, 'channel_version_rank': 1,
            'interpretation': 'Within each version/alternative the four raw profile attributes are fixed. They are absorbed by cohort-specific alternative constants. Recruitment channel and version are identical indicators.'}


def table(frame, columns, formats=None):
    formats = formats or {}
    result = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for _, row in frame.iterrows():
        result.append('| '+' | '.join(formats[c](row[c]) if c in formats else str(row[c]) for c in columns)+' |')
    return '\n'.join(result)


def write_report(out, results, audit, rank, provenance):
    f = results['all_records']; h = results['non_AI_labelled']; c = results['non_AI_bulk_date']
    def para(r):
        return f"V1={r['test']['n_V1']}，V2={r['test']['n_V2']}；总体Wald={r['test']['wald']:.3f}，自由度{r['test']['df']}，渐近p={r['test']['p_asymptotic_association_only']:.4f}，受访者层面中心化bootstrap p={r['test']['p_centered_cluster_bootstrap_association_only']:.4f}。"
    source_profile = pd.read_csv(out/'source_choice_profile.csv')
    b_f = f['coefficients'].query("model=='mnl_common' and term=='price_per_10_CNY'").iloc[0]
    b_h = h['coefficients'].query("model=='mnl_common' and term=='price_per_10_CNY'").iloc[0]
    lines = ['# DH36问卷数据核查与模型结果', '日期：2026年11月8日。',
        '已完成三种MNL参数模型、两个逐题概率基线、受访者分组5折验证和4,999次受访者层面bootstrap。全量360条包含60条AI标记记录。总体组间比较在全量数据中的bootstrap p约0.031，排除AI标记后约0.099，结论对来源处理敏感。渠道与版本完全重合，当前结果不能作为随机化充分性检验。',
        f"共同MNL的月费系数在全量数据中为{b_f['estimate']:.4f}（每10元，95%区间{b_f['ci_low']:.4f}至{b_f['ci_high']:.4f}），在非AI标记样本中为{b_h['estimate']:.4f}（95%区间{b_h['ci_low']:.4f}至{b_h['ci_high']:.4f}）。两种数据口径下，较低月费均与更高选择概率相关。加入版本／渠道交互在本次受访者分组验证中没有改善平均预测损失。",
        '## 本次数据与结论范围',
        '问卷星两版各180条，共360条。来源字段中两版各30条标为“AI样本”，合计60条。全量模型保留360条；另报告不含AI标记记录的300条，以及排除一份较早记录后的299条敏感性分析。来源标签保持原样，未将AI标记记录写成已核实的真人答卷。',
        '研究者确认：不同招募渠道分别只填写一个版本；没有分配记录；发放期间未修改问卷。渠道与版本完全重合。任何版本差异同时包含招募群体、渠道、效用、尺度和其他未测因素。当前数据不能作为已实施的候选保持随机实验，也不能用拒绝版本不变来确认候选摘要不充分。标准误条件于这两个已观察渠道，不能估计跨渠道总体的不确定性。',
        '以下均为观察性、探索性分析。分析方案未外部预注册；AI来源、速度和早期答卷标记在查看导出文件时发现。未以显著性、是否一直选同一项或是否选择弃选作为删样规则。',
        '以答卷序号作为聚类标识。全量模型的标准误和bootstrap额外依赖不同记录之间相互独立；AI标记记录的生成机制尚未核实，这一条件也未得到验证。因此，全量p值仅作混合来源数据的数值敏感性结果。平台序号与IP也不能完全证实每条记录对应不同的真实参与者。',
        '## 数据核查',
        table(audit, ['version', 'platform_source', 'records', 'duration_min', 'duration_median', 'duration_max', 'under_60_seconds']),
        '不同来源的选择分布如下。分母为各来源记录数乘以8题，仅作描述，不将同一人的8题当成独立样本：',
        table(source_profile, ['version','platform_source','records','choice_rows','share_A','share_B','share_opt_out'],
              {x:lambda z:f'{z:.1%}' for x in ['share_A','share_B','share_opt_out']}),
        '11题均有填写，八道选择题的合法编码为1=A、2=B、3=都不选择。原始文件未更改。平台IP仅用于本地重复标记，未进入公开结果或模型。',
        '版本2有一条2026年11月3日提交、用时28秒的记录，其余记录为11月8日提交。研究者尚未确认它是否试填；全量与非AI模型保留该记录，299条敏感性模型按提交日期排除它，不能将这一步描述为事先规定的排除规则。',
        '## 模型构造与识别',
        r'对每名受访者的8次回答共同保留。取弃选效用为0，服务效用模型为 $U_{itj}=\alpha_A1(j=A)+\alpha_B1(j=B)+\beta_p(p_{tj}-89)/10+\gamma_1 1(G_{tj}=G1)+\gamma_2 1(G_{tj}=G2)$。',
        r'边际选择概率为 $P(Y_{it}=j)=\exp(U_{itj})/\sum_{k\in\{A,B,0\}}\exp(U_{itk})$，尺度归一化。同一人的八次选择通过聚类稳健推断处理关联。',
        '模型1共同估计上述5个参数。模型2增加两个版本／渠道截距，共7个参数。模型3允许5个参数均随版本／渠道变化，共10个参数。ASC吸收各版本固定服务组合与其他未观测组别差异；不得把某个ASC或版本交互解释为单独的数据保存、咨询、提醒或效果研究属性的效应。',
        f"完整原始属性模型共有{rank['raw_model_columns']}列，有效秩为{rank['raw_model_rank']}，缺少{rank['unidentified_dimensions']}个独立方向。将四项固定属性和版本特定服务常数同时放入模型，无法分别估计这些属性的参数。渠道与版本两列的秩为1，加入渠道控制变量无法分离两者影响。",
        'MNL按每题的边际选择概率构造复合似然，以受访者汇总得分计算聚类稳健协方差。它没有为同一人的八次回答建立独立的联合似然，不报告将2,880次选择当独立受访者得到的LR显著性或AIC/BIC。',
        '另用逐题共同概率模型与逐题分组概率模型作灵活基线。8题×A/B共16维概率差采用两组均值协方差与受访者层面bootstrap，24个类别差的区间使用联合max-t校正。单题检验的p值按8题Holm校正。它们检验观察群体的概率差，均不恢复随机化。',
        '## 总体组间比较',
        '**全量360条（包含AI标记记录）：** '+para(f),
        '**非AI标记300条：** '+para(h),
        '**非AI标记且11月8日提交299条：** '+para(c),
        '这些p值描述当前两个样本的关联。在缺少渠道重叠与分配记录的情况下，p值大小不能识别被粗化丢弃的属性信息。',
        '## 全量模型参数',
        table(f['coefficients'], ['model', 'term', 'estimate', 'cluster_se', 'ci_low', 'ci_high'],
              {x:lambda z:f'{z:.4f}' for x in ['estimate','cluster_se','ci_low','ci_high']}),
        '价格系数对应每月费用增加10元。G1和G2以G0“不向外部机构共享”为参照。数据保存、咨询、提醒和效果研究的独立系数没有被估计。',
        '## 受访者分组交叉验证',
        '同一受访者的8题始终进入同一折。采用按版本分层的5折验证，预测目标仅限于这两个已观察渠道、这八个菜单下的新记录。逐题概率基线使用每格0.5平滑。数值为每道选择题平均负对数概率，越低表示预测更好。差值区间以受访者的配对损失计算，属于给定一次训练划分的近似描述，不包括重新训练模型的全部不确定性。',
        '**全量360条：**',
        table(f['cv'], ['label','cv_mean_log_loss','difference_vs_common_MNL','paired_ci_low','paired_ci_high'],
              {x:lambda z:f'{z:.5f}' for x in ['cv_mean_log_loss','difference_vs_common_MNL','paired_ci_low','paired_ci_high']}),
        '**非AI标记300条：**',
        table(h['cv'], ['label','cv_mean_log_loss','difference_vs_common_MNL','paired_ci_low','paired_ci_high'],
              {x:lambda z:f'{z:.5f}' for x in ['cv_mean_log_loss','difference_vs_common_MNL','paired_ci_low','paired_ci_high']}),
        '## 逐题概率差',
        '图中差值为版本2减版本1，单位为百分点。区间按同一人的8题重抽样，并对24个类别差同时校正。',
        '![全量与非AI标记样本的逐题概率差](choice_contrasts.png)',
        '## 样本描述与模型检验的边界',
        '本次无需Cronbach α、因子分析或将选择题当成Likert量表求总分。各任务是不同菜单的选择观测。第11题是完成选择后的侧重自述，不作为主模型的混杂控制。过去一年应用使用和付费经历只作样本描述。',
        '已对数据完整性、合法编码、菜单配对、设计秩、优化梯度、受访者切分和聚类协方差作检查。原始IP、时间戳、逐人答卷和逐人预测均仅在本地私有目录。',
        '## 后续研究安排',
        '若要完成充分性的随机化实证，需要在同一招募渠道内把新受访者随机分到两个版本，保存分配机制与匿名记录。采用多个渠道时，每个渠道都应包含两版，并按渠道内的实际随机化规则分析。继续在当前两类渠道分别只补各自版本，不能解除渠道与版本的混淆。',
        '当前数据可作为观察性实施示例与模型诊断，论文的随机实验部分仍需新的可追溯分配证据。AI标记记录可以用于独立的合成数据对照，数量应单列。',
        '## 方法参考',
        '- ISPOR统计分析建议：https://www.valueinhealthjournal.com/article/S1098-3015%2816%2930291-1/fulltext',
        '- ISPOR实验设计建议：https://www.valueinhealthjournal.com/article/S1098-3015%2812%2904162-9/fulltext',
        '- 聚类稳健协方差参考：https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_cluster.html',
        '## 来源校验',
        '导出日期2026年11月8日；版本由问卷编号映射。来源文件逐字节校验值：',
        '\n'.join(f"- {p['version']}，问卷{p['activity_id']}，{p['rows']}行，SHA256 `{p['sha256']}`。" for p in provenance)]
    (out/'DH36_模型与数据核查报告.md').write_text('\n\n'.join(lines)+'\n')


def plots(out, results):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.font_manager import FontProperties
    matplotlib.rcParams['svg.fonttype']='none'
    font = FontProperties(fname='/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    fig, axes = plt.subplots(1, 2, figsize=(13, 10), sharey=True)
    for ax, key, title in zip(axes, ['all_records','non_AI_labelled'], ['全量360条（含60条AI标记）','非AI标记300条']):
        d = results[key]['contrasts']
        for i, row in d.iterrows():
            colour = ['#2463a5','#cf7d18','#607d45'][i%3]
            ax.plot([row.simultaneous_low*100, row.simultaneous_high*100], [i,i], color=colour, lw=1.8)
            ax.scatter([row.difference_V2_minus_V1*100], [i], color=colour, s=24, zorder=3)
        ax.axvline(0, color='#777777', lw=.8)
        ax.set_title(title, fontproperties=font, fontsize=14)
        ax.set_xlabel('版本2 − 版本1（百分点，联合95%区间）', fontproperties=font, fontsize=11)
        ax.grid(axis='x', alpha=.18);ax.set_xlim(-40,40)
    axes[0].set_yticks(range(24), [f"T{t+1:02d}  {a}" for t in range(8) for a in ['A','B','弃选']], fontproperties=font)
    axes[0].invert_yaxis()
    fig.suptitle('观察群体的选择概率差', fontproperties=font, fontsize=17)
    fig.text(.5,.012,'渠道与版本完全重合；图中差异不能解释为随机分配的因果效应。',ha='center',fontproperties=font,fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.96));fig.savefig(out/'choice_contrasts.png',dpi=180)
    fig.savefig(out/'choice_contrasts.svg');plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bootstrap-reps',type=int,default=4999)
    args=parser.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(args.input_dir/'matplotlib_cache'))
    spec=json.loads(args.spec.read_text());base=menu_arrays(spec)
    people,provenance,audit,duplicates=read_exports(args.input_dir,spec)
    people.to_csv(args.input_dir/'responses_collected_prepared.csv',index=False)
    save_json(out/'metadata_duplicate_flags.json',duplicates)
    audit.to_csv(out/'sample_source_audit.csv',index=False)
    source_profile=[]
    for (version,source),group in people.groupby(['version','platform_source']):
        values=group[[f'q{i}' for i in range(1,9)]].to_numpy().flatten()
        source_profile.append({'version':version,'platform_source':source,'records':len(group),'choice_rows':len(values),
                               'share_A':float(np.mean(values==1)),'share_B':float(np.mean(values==2)),
                               'share_opt_out':float(np.mean(values==3))})
    pd.DataFrame(source_profile).to_csv(out/'source_choice_profile.csv',index=False)
    background=[]
    for subset,mask in [('all_records',np.ones(len(people),bool)),('non_AI_labelled',~people.ai_label)]:
        for v in ['V1','V2']:
            dv=people.loc[mask & (people.version==v)]
            for field in ['app_use','paid_service','priority']:
                for value,count in dv[field].value_counts().sort_index().items():
                    background.append({'population':subset,'version':v,'field':field,'value':int(value),'count':int(count),'n':len(dv),'proportion':float(count/len(dv))})
    pd.DataFrame(background).to_csv(out/'background_counts.csv',index=False)
    results={};global_rows=[];all_coefficients=[];all_cv=[];diagnostics=[]
    masks={'all_records':np.ones(len(people),bool),'non_AI_labelled':~people.ai_label,
           'non_AI_bulk_date':(~people.ai_label)&(~people.before_bulk_collection_date)}
    for k,(population,mask) in enumerate(masks.items()):
        d=people.loc[mask].reset_index(drop=True)
        contrast,test=contrasts(d,args.bootstrap_reps,20261108+k)
        contrast.to_csv(out/f'{population}_contrasts.csv',index=False)
        global_rows.append({'population':population,**test})
        coefficients=[]
        for model in ['mnl_common','mnl_cohort_intercepts','mnl_cohort_all']:
            fit=fit_mnl(d,base,model,validation=True)
            for row in fit['parameters']:coefficients.append({'population':population,'model':model,**row})
            diagnostics.append({'population':population,'model':model,'n_respondents':len(d),'tasks':len(d)*8,
                                'parameters':len(fit['beta']),'rank':fit['rank'],'gradient_max':fit['gradient_max'],
                                'solver_success':fit['solver_success'],'composite_log_likelihood':fit['composite_log_likelihood'],
                                'cohort_wald_p_association_only':None if fit['test'] is None else fit['test']['p_value_association_only']})
        coefficients=pd.DataFrame(coefficients);all_coefficients.append(coefficients)
        cv,folds=predictive_validation(d,base,20261108)
        cv.insert(0,'population',population);all_cv.append(cv)
        pd.DataFrame({'participant_id':d.participant_id,'fold':folds}).to_csv(args.input_dir/f'{population}_cv_folds.csv',index=False)
        results[population]={'contrasts':contrast,'test':test,'coefficients':coefficients,'cv':cv}
        print(json.dumps({'population':population,**test},ensure_ascii=False),flush=True)
    pd.DataFrame(global_rows).to_csv(out/'global_cohort_association.csv',index=False)
    pd.concat(all_coefficients).to_csv(out/'mnl_cluster_parameters.csv',index=False)
    pd.concat(all_cv).to_csv(out/'respondent_grouped_cv.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'model_diagnostics.csv',index=False)
    rank=rank_audit(spec);save_json(out/'identification_audit.json',rank)
    save_json(out/'data_provenance.json',{'export_date':'2026-11-08','raw_exports':provenance,
        'raw_records':len(people),'ai_labelled_records':int(people.ai_label.sum()),
        'non_AI_labelled_records':int((~people.ai_label).sum()),'early_records_flag_only':int(people.before_bulk_collection_date.sum()),
        'recruitment':'Different channels each received only one version; no allocation record. Researcher reports no content changes.',
        'full_source_analysis_retains_AI_labels':True,'verified_human_status_of_AI_labelled_records':False,
        'randomized_sufficiency_claim_supported':False,'individual_records_publicly_shared':False})
    plots(out,results);write_report(out,results,audit,rank,provenance)
    print('Analysis completed. Raw files and participant-level records remain private.',flush=True)


if __name__=='__main__':main()
