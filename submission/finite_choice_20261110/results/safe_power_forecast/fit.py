"""Fit the frozen safety and reference policies on the archived choice draws."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from inputs import sources, sets, score_metric, mean_feature


def surface(features, available, weights, target, beta, penalty):
    utility=np.where(available,features@beta,-np.inf)
    normalizer=logsumexp(utility,axis=1)
    prob=np.exp(utility-normalizer[:,None])
    mean=np.einsum('na,nap->np',prob,features)
    error=features-mean[:,None,:]
    value=weights@normalizer-target@beta+penalty*(beta@beta)/2
    grad=weights@mean-target+penalty*beta
    hess=np.einsum('n,na,nap,naq->pq',weights,prob,error,error)+penalty*np.eye(len(beta))
    return value,grad,hess


def fit_model(args,start,tolerance):
    x,available,w,b,ridge=args
    scale=np.maximum(np.max(abs(x),axis=(0,1)),0.05)
    f=x/scale
    target=b/scale
    def update(eta):
        v,g,h=surface(f,available,w,target,eta,0)
        beta=eta/scale
        g+=ridge*beta/scale
        h+=np.diag(ridge/scale**2)
        return v+ridge*(beta@beta)/2,g,h
    opt=minimize(lambda eta:update(eta)[:2],start*scale,jac=True,method='trust-exact',
                 hess=lambda eta:update(eta)[2],options={'gtol':2e-10,'maxiter':300})
    eta=opt.x.copy()
    for _ in range(10):
        _,grad,H=update(eta)
        if np.max(abs(grad*scale))<=1e-9:
            break
        eta-=np.linalg.solve(H,grad)
    _,grad,h=surface(x,available,w,b,eta/scale,ridge)
    max_score=float(np.max(abs(grad)))
    good=bool(np.isfinite(eta).all() and max_score<tolerance and np.linalg.eigvalsh(h)[0]>0)
    return eta/scale,max_score,good


def labelled(panel, selected, ridge):
    x=panel.x[selected]
    b=x[np.arange(len(x)),panel.y[selected]].mean(axis=0)
    return x,panel.available[selected],np.full(len(x),1/len(x)),b,ridge


def corrected(panel, label, aux, power):
    xl,al,wl,b,_=labelled(panel,label,0)
    xu=panel.x[aux]
    b=b+power*(mean_feature(xu,panel.proxy[aux])-mean_feature(xl,panel.proxy[label]))
    available=panel.available[aux]
    wu=np.full(len(xu),1/len(xu))
    if power==1:
        return xu,available,wu,b,0
    return np.concatenate([xl,xu]),np.concatenate([al,available]),np.r_[(1-power)*wl,power*wu],b,0


def forecast(panel,beta):
    if beta is None:
        p=panel.proxy[panel.test]
        with np.errstate(divide='ignore'):
            lp=np.log(p)
    else:
        x,available=panel.x[panel.test],panel.available[panel.test]
        utility=np.where(available,x@beta,-np.inf)
        lp=utility-logsumexp(utility,axis=1,keepdims=True)
        p=np.exp(lp)
    return score_metric(panel,p,lp)


def evaluation_gradient(panel,beta):
    x,available,y=panel.x[panel.test],panel.available[panel.test],panel.y[panel.test]
    _,person=np.unique(panel.ids[panel.test],return_inverse=True)
    count=np.bincount(person)
    weight=1/(len(count)*count[person])
    u=np.where(available,x@beta,-np.inf)
    p=np.exp(u-logsumexp(u,axis=1,keepdims=True))
    return weight@(np.einsum('na,nap->np',p,x)-x[np.arange(len(x)),y])


def save_result(panel,key,method,beta=None,score=None,status='native',power=None):
    exists=status in ('native','labelled_ridge_fallback','frozen_teacher')
    loss,brier=forecast(panel,beta) if exists else (None,None)
    return dict(dataset=panel.name,allocation=key[0],label_people=key[1],draw=key[2],
                method=method,power=power,state=status,loss=loss,brier=brier,
                cost_positive=bool(beta[-1]>0) if exists and beta is not None else None,
                score=score,beta=beta.tolist() if beta is not None else None)


def fits_for_group(panel,archive,index,budget,cap,states,plan):
    label,draws=sets(panel,archive,index,budget)
    eps=plan['tolerance']['proof_cap']
    accuracy=plan['tolerance']['score']
    group=(index,budget,-1)
    ridge,ridge_score,ridge_ok=fit_model(labelled(panel,label,.01),np.zeros(panel.x.shape[-1]),accuracy)
    assert ridge_ok
    records=[save_result(panel,group,'labelled_ridge',ridge,ridge_score)]
    start=ridge
    human=labelled(panel,label,0)
    ordinary=states['ordinary']=='interior'
    if ordinary:
        start,score,ok=fit_model(human,ridge,accuracy)
        assert ok
        records.append(save_result(panel,group,'labelled_MLE',start,score))
        hessian=surface(*human[:-1],start,0)[2]
        val_gradient=evaluation_gradient(panel,start)
        base_loss=records[-1]['loss']
    else:
        records.append(save_result(panel,group,'labelled_MLE',status='support_unavailable'))
    for method,penalty in (('all_human_MLE',0),('all_human_ridge',.01)):
        fit,score,ok=fit_model(labelled(panel,panel.train|label,penalty),ridge,accuracy)
        assert ok
        records.append(save_result(panel,group,method,fit,score))
    records.append(save_result(panel,group,'frozen_teacher',status='frozen_teacher'))
    derivatives=[]
    for j,aux in enumerate(draws):
        key=(index,budget,j)
        conditions={'safe_half':(cap/2,cap>eps),'safe_edge':(cap,cap>eps)}
        for name,power in (('fixed_half',.5),('fixed_unit',1.)):
            state,nl,nu=states['rows'][j,power]
            assert (nl,nu)==(label.sum(),aux.sum())
            conditions[name]=(power,state=='interior')
        for name,(power,eligible) in conditions.items():
            if not eligible:
                fallback=save_result(panel,key,name,ridge,ridge_score,status='labelled_ridge_fallback',power=power)
                fallback['reason']='zero_safe_cap' if name.startswith('safe') else 'noninterior_archive_support'
                records.append(fallback)
                continue
            beta,score,converged=fit_model(corrected(panel,label,aux,power),start,accuracy)
            if converged:
                records.append(save_result(panel,key,name,beta,score,power=power))
            else:
                fallback=save_result(panel,key,name,ridge,ridge_score,status='labelled_ridge_fallback',power=power)
                fallback['reason']='fit_failed_score'
                fallback['failed_score']=score
                records.append(fallback)
        if ordinary and cap>eps:
            sl=mean_feature(panel.x[label],forecast_distribution(panel,start,label)-panel.proxy[label])
            su=mean_feature(panel.x[aux],forecast_distribution(panel,start,aux)-panel.proxy[aux])
            speed=np.linalg.solve(hessian,sl-su)
            slope=float(val_gradient@speed)
            power=.01*cap
            beta,score,ok=fit_model(corrected(panel,label,aux,power),start,accuracy)
            assert ok
            fd=(forecast(panel,beta)[0]-base_loss)/power
            derivatives.append(dict(dataset=panel.name,allocation=index,label_people=budget,draw=j,
                                    slope=slope,small_power=power,secant=fd,score=score,beta=beta.tolist()))
    return records,derivatives


def forecast_distribution(panel,beta,take):
    u=np.where(panel.available[take],panel.x[take]@beta,-np.inf)
    return np.exp(u-logsumexp(u,axis=1,keepdims=True))


def main(args):
    plan,archive,caps,states,panels=sources(args)
    fits=[]
    derivatives=[]
    for panel in panels:
        for j in range(archive['label_allocations']):
            for budget in archive['label_budgets']:
                key=(panel.name,j,budget)
                current,slopes=fits_for_group(panel,archive,j,budget,caps[key],states[key],plan)
                fits.extend(current)
                derivatives.extend(slopes)
    args.out.mkdir()
    for n,data in [('fits',fits),('derivatives',derivatives)]:
        (args.out/f'{n}.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(fits=len(fits), derivatives=len(derivatives),
                     methods=dict(Counter(r['method'] for r in fits)),
                     numeric_failures=sum(r.get('reason')=='fit_failed_score' for r in fits))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('food-data','food-archive','swiss-data','swiss-archive','reference','certificate','out'):
        p.add_argument('--'+key,type=Path,required=True)
    with threadpool_limits(limits=1):
        main(p.parse_args())
