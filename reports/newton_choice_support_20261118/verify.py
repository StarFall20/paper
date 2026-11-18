"""Independently reconstruct the witnesses and check a change of attribute units."""
import argparse
from collections import Counter
from itertools import chain
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from inputs import food, swiss, rows, sha, read


ROOT=Path(__file__).resolve().parent


def alternate_witness(case, threshold, equality, use_proxy=False):
    p=len(case.pilot)
    # Apply a nonsingular diagonal change of units, then work with second moments.
    scales=np.geomspace(0.25,8,p)
    x=case.features*scales
    target=case.moment*scales
    if use_proxy:
        probs=case.auxiliary_proxy
    else:
        eta=case.pilot/scales
        util=np.where(case.available,np.sum(x*eta,axis=2),-np.inf)
        probs=np.exp(util-logsumexp(util,axis=1,keepdims=True))
    means=(x*probs[:,:,None]).sum(axis=1)
    flat=x.reshape(-1,p)
    weighted=probs.ravel()[:,None]*flat
    hessian=(flat.T @ weighted - means.T @ means)/len(x)
    score=means.mean(axis=0)-target
    direction=-np.linalg.solve(hessian,score)
    witness=probs+probs*((x @ direction)-(means@direction)[:,None])
    recoding_step=direction*scales
    nstep=np.linalg.norm(recoding_step)
    unit=direction/nstep if nstep else direction
    ray=np.max(np.where(case.available,x@unit,-np.inf),axis=1).mean()-target@unit
    margin=float(witness[case.available].min())
    moment=(x*witness[:,:,None]).sum(axis=1).mean(axis=0)
    err=max(np.max(abs((moment-target)/scales)),np.max(abs(witness.sum(axis=1)-1)))
    assert err<equality
    result='interior' if margin>threshold else 'outside' if ray<-threshold else 'needs_support_program'
    inverse=np.linalg.inv(hessian)
    center=x-means[:,None,:]
    bound=np.sqrt(np.max(np.sum(center @ inverse * center,axis=2)[case.available])*(score@inverse@score))
    return result,float(ray),margin,nstep,float(bound),float(err)


def reconcile(witness, row, errors):
    state,recession,margin,step,norm,residual=witness
    assert state==row['screen']
    metrics={'normalized_newton_recession':recession,'minimum_probability_at_step':margin,
             'step_norm':step,'norm_sufficient_factor':norm}
    for name,value in metrics.items():
        discrepancy=abs(value-float(row[name]))/max(1,abs(value))
        assert discrepancy<5e-8,(row['dataset'],row['specification'],row['allocation'],name,discrepancy)
        errors[name]=max(errors[name],discrepancy)
    errors['maximum_probability_or_moment_balance']=max(errors['maximum_probability_or_moment_balance'],residual)
    return state


def verify(args):
    saved=read(args.results/'summary.json')
    plan=read(ROOT/'plan.json')
    archive=rows(args.results/'records.csv')
    sensitivity=read(args.sensitivity/'summary.json')
    proxy=rows(args.sensitivity/'records.csv')
    assert sha(args.sensitivity/'records.csv')==sensitivity['records_sha256']
    assert sha(args.results/'summary.json')==sensitivity['primary_sha256']
    assert sha(ROOT/'frozen_proxy_sensitivity_plan.json')==sensitivity['plan_sha256']
    assert sha(args.results/'records.csv')==saved['records_sha256']
    assert sha(ROOT/'plan.json')==saved['plan_sha256']
    assert len(archive)==2576
    counters=Counter()
    errors=Counter()
    proxy_counters=Counter()
    proxy_errors=Counter()
    union=Counter()
    tolerances=(plan['witness_tolerance'],plan['equality_tolerance'])
    for case,old,other in zip(chain(food(args.food_data,args.food_archive),swiss(args.swiss_data,args.swiss_archive)),archive,proxy,strict=True):
        for row in (old,other):
            assert [case.dataset,case.specification,case.allocation,case.labelled_people,case.exact_state] == [
                row['dataset'],row['specification'],int(row['allocation']),int(row['labelled_people']),row['reference']]
        state=reconcile(alternate_witness(case,*tolerances),old,errors)
        proxy_state=reconcile(alternate_witness(case,*tolerances,use_proxy=True),other,proxy_errors)
        assert state in (case.exact_state,'needs_support_program')
        assert proxy_state in (case.exact_state,'needs_support_program')
        assert other['either_decides']==str(case.exact_state in (state,proxy_state))
        assert other['pilot_decides']==str(state==case.exact_state)
        assert other['proxy_decides']==str(proxy_state==case.exact_state)
        counters[(case.exact_state,state)]+=1
        proxy_counters[(case.exact_state,proxy_state)]+=1
        union[case.exact_state]+=case.exact_state in (state,proxy_state)
    assert sum(counters.values())==2576
    assert counters['interior','interior']==saved['totals']['combined']['interior']['decisive']
    assert counters['outside','outside']==saved['totals']['combined']['outside']['decisive']
    assert not counters['outside','interior'] and not counters['interior','outside']
    for truth in ('interior','outside'):
        assert proxy_counters[truth,truth]==sensitivity['totals']['combined'][truth]['proxy_decides']
        assert union[truth]==sensitivity['totals']['combined'][truth]['either_decides']
    check=dict(state='passed',verified_allocations=sum(counters.values()),
               archived_source_receipt_sha256=sha(args.results/'summary.json'),
               final_counts={truth:{detected:qty for (label,detected),qty in counters.items() if label==truth}
                             for truth in ('interior','outside')},
               max_relative_reconstruction_errors=dict(errors),
               proxy_source_receipt_sha256=sha(args.sensitivity/'summary.json'),
               proxy_counts={truth:{detected:qty for (label,detected),qty in proxy_counters.items() if label==truth}
                             for truth in ('interior','outside')},
               union_decisive_counts=dict(union),
               proxy_max_relative_reconstruction_errors=dict(proxy_errors),
               recoding='Rescale each coefficient column by fixed factors from 0.25 to 8. Reconstruct covariance using second moments and solve with an independent formulation. Record recession in original coefficient units.',
               verifier_sha256=sha(Path(__file__)))
    args.out.write_text(json.dumps(check,indent=2,allow_nan=False)+'\n')
    print(json.dumps(check))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('food-data','food-archive','swiss-data','swiss-archive','results','sensitivity','out'):
        parser.add_argument('--'+key,type=Path,required=True)
    with threadpool_limits(limits=1):
        verify(parser.parse_args())
