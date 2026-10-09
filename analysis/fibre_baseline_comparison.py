"""Conditional-sufficiency baselines on finite fibres.

All tests condition on fibre membership, profile counts and fibre outcome
margins. Uniform assignment within a fibre justifies a multivariate
hypergeometric reference. The support-complete score and block-omnibus
CMH are algebraically identical on this saturated finite support; no new
inferential family is claimed. Complete-profile LR uses unrestricted fibre
intercepts under the null. ML uses an honest 70/30 split and freezes the
learner before randomizing held-out outcomes within fibres.
"""
from __future__ import annotations
import argparse, csv, os
import numpy as np
from scipy.special import xlogy
from sklearn.ensemble import RandomForestClassifier
from support_complete_fibre_benchmark import generate
from support_complete_fibre_design import candidate_values


def fibre_tables(points, fibre, profile, y):
    z = candidate_values(points)
    tables = []
    for g in np.unique(fibre):
        ids = np.flatnonzero(z == g)
        mask = fibre == g
        n = np.array([np.sum(mask & (profile == j)) for j in ids], dtype=int)
        a = np.array([np.sum(y[mask & (profile == j)]) for j in ids], dtype=int)
        tables.append((ids, n, a))
    return tables


def table_statistics(n, a):
    """Block-omnibus generalized CMH and G-squared; accepts batched a."""
    n = np.asarray(n, dtype=float)
    a = np.atleast_2d(a).astype(float)
    N = n.sum(); m = a.sum(axis=1, keepdims=True)
    valid = (m[:, 0] > 0) & (m[:, 0] < N) & (N > 1)
    if not np.any(valid): return np.zeros(len(a)), np.zeros(len(a))
    p = m / max(N, 1); expected = n[None, :] * p
    denom = n[None, :] * p * (1-p)
    pearson = np.sum(np.divide((a-expected)**2, denom,
                              out=np.zeros_like(a), where=denom>0), axis=1)
    # The conditional covariance is m(N-m)/(N-1)*(diag(n/N)-ww').
    # On the zero-sum contrast space its inverse gives (N-1)/N*Pearson.
    cmh = pearson * (N-1)/max(N,1)
    failures = n[None, :] - a
    expected_fail = n[None, :] * (1-p)
    ratio1 = np.divide(a, expected, out=np.ones_like(a), where=expected>0)
    ratio0 = np.divide(failures, expected_fail, out=np.ones_like(a), where=expected_fail>0)
    lr = 2*np.sum(xlogy(a, ratio1) + xlogy(failures, ratio0), axis=1)
    cmh[~valid] = 0; lr[~valid] = 0
    return cmh, np.maximum(lr,0)


def conditional_tests(points, fibre, profile, y, reps=499, seed=0):
    rng = np.random.default_rng(seed)
    score = 0.; lr = 0.; score_ref = np.zeros(reps); lr_ref = np.zeros(reps)
    for _, n, a in fibre_tables(points, fibre, profile, y):
        sc, dev = table_statistics(n, a); score += sc[0]; lr += dev[0]
        draws = rng.multivariate_hypergeometric(n, int(a.sum()), size=reps)
        sc, dev = table_statistics(n, draws); score_ref += sc; lr_ref += dev
    ps = (1 + np.sum(score_ref >= score-1e-12))/(reps+1)
    pl = (1 + np.sum(lr_ref >= lr-1e-12))/(reps+1)
    return {'support_complete':float(ps), 'generalized_cmh':float(ps),
            'complete_fibre_lr':float(pl)}


def ml_residual_pvalue(points, fibre, profile, y, reps=499, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y)); cut = int(.70*len(y))
    train, test = order[:cut], order[cut:]
    clf = RandomForestClassifier(n_estimators=80, max_depth=5,
            min_samples_leaf=8, random_state=seed, n_jobs=1)
    x = np.column_stack((points, candidate_values(points)))
    clf.fit(x[profile[train]], y[train])
    # Cache predictions for all finite profiles. Only held-out outcomes vary.
    predictions = clf.predict_proba(x)
    prob = (predictions[:, list(clf.classes_).index(1)] if 1 in clf.classes_
            else np.zeros(len(points)))
    prob = np.clip(prob,1e-6,1-1e-6)
    observed = 0.; reference = np.zeros(reps)
    for ids, n, a in fibre_tables(points, fibre[test], profile[test], y[test]):
        logits = np.log(prob[ids]/(1-prob[ids]))
        # The fibre-only residual log score and failure log terms are constant
        # under the fixed-margin reference, so only this term varies.
        observed += a @ logits
        draws = rng.multivariate_hypergeometric(n, int(a.sum()), size=reps)
        reference += draws @ logits
    return float((1+np.sum(reference >= observed-1e-12))/(reps+1))


def run(reps=500, respondents=400, randomization_reps=499,
        ml_randomization_reps=499, eta=.45,
        out='results/fibre_baseline_comparison.csv'):
    rows=[]
    for condition in ('null','aligned','interaction','hidden_quadratic','hidden_orthogonal'):
        for rep in range(reps):
            pts,fibre,profile,y=generate(20261007+rep,respondents,condition,eta)
            pvals=conditional_tests(pts,fibre,profile,y,randomization_reps,20262000+rep)
            pvals['ml_residual']=ml_residual_pvalue(pts,fibre,profile,y,ml_randomization_reps,20265000+rep)
            for method,p in pvals.items():
                rows.append(dict(condition=condition,method=method,rep=rep,
                    respondents=respondents,eta=eta,randomization_reps=(ml_randomization_reps if method=='ml_residual' else randomization_reps),
                    pvalue=p,reject=int(p<=.05)))
        print(condition,{m:round(np.mean([r['reject'] for r in rows if r['condition']==condition and r['method']==m]),3) for m in pvals},flush=True)
    os.makedirs(os.path.dirname(out) or '.',exist_ok=True)
    with open(out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('wrote',len(rows),'rows to',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reps',type=int,default=500);p.add_argument('--respondents',type=int,default=400);p.add_argument('--randomization-reps',type=int,default=499);p.add_argument('--ml-randomization-reps',type=int,default=499);p.add_argument('--eta',type=float,default=.45);p.add_argument('--out',default='results/fibre_baseline_comparison.csv');run(**vars(p.parse_args()))
