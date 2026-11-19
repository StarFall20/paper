"""Fit the declared LLM, support and matched-prompt comparators."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from source_data import ROOT,cases,dependencies


def moments(x,mass):
    return np.einsum('nja,nj->a',x,mass)/len(x)


def objective(x,mass,ridge=0):
    return x,np.ones(x.shape[:2],bool),np.full(len(x),1/len(x)),moments(x,mass),ridge


def correction(case,power):
    l,y,gl,_=case.label
    u,gu,_=case.aux
    b=moments(l,y)+power*(moments(u,gu)-moments(l,gl))
    x=np.concatenate((l,u))
    w=np.r_[np.full(len(l),(1-power)/len(l)),np.full(len(u),power/len(u))]
    active=w>0
    return x[active],np.ones(x[active].shape[:2],bool),w[active],b,0


def metric(case,beta=None):
    x,y,p,ids=case.eval
    if beta is None:
        lp=np.log(p)
    else:
        u=x@beta
        lp=u-logsumexp(u,axis=1,keepdims=True)
        p=np.exp(lp)
    err=np.sum(-lp*y,axis=1)
    brier=np.square(p-y).sum(axis=1)
    return [float(np.mean([v[ids==j].mean() for j in np.unique(ids)])) for v in (err,brier)]


def row(case,method,state='native',beta=None,score=None,power=None,reason=None):
    is_forecast=beta is not None or method=='raw_LLM'
    loss,brier=metric(case,beta) if is_forecast else (None,None)
    return dict(context=case.key[0],rep=case.key[1],coverage=case.key[2],label_people=case.key[3],
                label_fold=case.key[4],eval_fold=case.key[5],prompt_people=case.prompt_people,
                prompt_menus=case.prompt_menus,method=method,state=state,score=score,power=power,
                reason=reason,loss=loss,brier=brier,beta=None if beta is None else beta.tolist())


def support(backend,inputs):
    x,availability,w,b,_=inputs
    report=backend.support.support(x,availability,w,b)
    return report[0]


def fit_required(backend,inputs,beta,accuracy):
    value,score,ok=backend.forecast.fit_model(inputs,beta,accuracy)
    assert ok,score
    return value,score


def policy(case,backend,name,power,eligible,pilot,accuracy,reason):
    if eligible:
        try:
            val,score,ok=backend.forecast.fit_model(correction(case,power),pilot[0],accuracy)
        except np.linalg.LinAlgError:
            ok=False
        if ok:
            return row(case,name,beta=val,score=score,power=power)
        reason='numeric_fit_unavailable'
    return row(case,name,'labelled_ridge_fallback',*pilot,power=power,reason=reason)


def run_case(case,plan,backend):
    x,y,gl,lids=case.label
    u,gu,uids=case.aux
    tol=plan['tolerance']['support']
    acc=plan['tolerance']['score']
    p=x.shape[-1]
    pilot=fit_required(backend,objective(x,y,.01),np.zeros(p),acc)
    data=[row(case,'labelled_ridge','native',*pilot)]
    ordinary=support(backend,objective(x,y))
    if ordinary=='interior':
        unpenalized=fit_required(backend,objective(x,y),pilot[0],acc)
        data.append(row(case,'labelled_MLE','native',*unpenalized))
    else:
        data.append(row(case,'labelled_MLE','unavailable',reason=ordinary))
    combined=np.concatenate([x,case.prompt[0]])
    choices=np.concatenate([y,case.prompt[1]])
    p_ridge=fit_required(backend,objective(combined,choices,.01),pilot[0],acc)
    data.append(row(case,'prompt_matched_ridge','native',*p_ridge))
    matched=support(backend,objective(combined,choices))
    if matched=='interior':
        p_mle=fit_required(backend,objective(combined,choices),p_ridge[0],acc)
        data.append(row(case,'prompt_matched_MLE','native',*p_mle))
    else:
        data.append(row(case,'prompt_matched_MLE','matched_ridge_fallback',*p_ridge,reason=matched))
    data.append(row(case,'raw_LLM','frozen'))
    cap,_,_=backend.certificate.labelled_depth(x,np.ones(x.shape[:2],bool),y.argmax(axis=1),gl,tol)
    assert -tol<=cap<=1+tol, (case.key,cap)
    # The LP can return signed roundoff at a mathematically zero endpoint.
    cap=0.0 if cap<=tol else min(cap,1.0)
    assert ordinary=='rank_deficient' or ((cap>0)==(ordinary=='interior')), (case.key,cap,ordinary)
    pp=cap/2
    if cap>tol:
        assert support(backend,correction(case,pp))=='interior'
    data.append(policy(case,backend,'safe_half',pp,cap>tol,pilot,acc,'zero_safe_bound'))
    unit=support(backend,correction(case,1))
    data.append(policy(case,backend,'fixed_unit',1.,unit=='interior',pilot,acc,unit))
    released=backend.external.correction_power(uids,gu,lids,gl,y)
    published=support(backend,correction(case,released))
    data.append(policy(case,backend,'released_rule',released,published=='interior',pilot,acc,published))
    return data


def main(args):
    plan,backend=dependencies(args.dependencies)
    result=[]
    for case in cases(args,plan,backend):
        result.extend(run_case(case,plan,backend))
    assert len(result)==(120+360)*8
    args.out.mkdir(parents=True)
    (args.out/'fits.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'records':len(result),'states':dict(Counter(r['state'] for r in result)),
                      'reasons':dict(Counter(r['reason'] for r in result if r['reason']))}))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for k in ('inputs','dependencies','out'):
        cli.add_argument('--'+k,type=Path,required=True)
    with threadpool_limits(limits=1):
        main(cli.parse_args())
