"""Certify finite estimation via a labelled-only bound on the correction power."""
import argparse
import csv
from itertools import chain
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csc_matrix, hstack, vstack
from threadpoolctl import threadpool_limits

from audit import digest, lookup, module_for, read

ROOT=Path(__file__).resolve().parent
OPTIONS={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}


def labelled_depth(x, available, y, g, tolerance):
    m,_,p=x.shape
    menu,alt=np.nonzero(available)
    n=len(menu)
    features=x[menu,alt]
    weights=np.full(m,1/m)
    anchor=g[menu,alt]/m
    assert np.all(anchor>0)
    B=vstack((coo_matrix((np.ones(n),(menu,np.arange(n))),shape=(m,n)),
              csc_matrix(features.T)),format='csc')
    b=x[np.arange(m),y].mean(axis=0)
    target=np.r_[weights,b]
    g_vector=features.T@anchor
    center=np.r_[weights,g_vector]
    assert max(abs(B@anchor-center))<tolerance

    # q represents the residual (1-epsilon) probability mass.
    mixed=hstack((B,csc_matrix(center[:,None])),format='csc')
    fit=linprog(np.r_[np.zeros(n),-1.0],A_eq=mixed,b_eq=target,
                bounds=[(0,None)]*n+[(0,1)],options=OPTIONS,method='highs-ipm')
    assert fit.success,fit.message
    cap=float(fit.x[-1])
    assert np.max(abs(mixed@fit.x-target))<tolerance
    if max(abs(b-g_vector))<tolerance:
        assert abs(cap-1)<tolerance
    else:
        ray=np.r_[np.zeros(m),b-g_vector]
        radial=hstack((B,csc_matrix(-ray[:,None])),format='csc')
        other=linprog(np.r_[np.zeros(n),-1.0],A_eq=radial,b_eq=center,
                      bounds=[(0,None)]*n+[(1,None)],options=OPTIONS,method='highs-ds')
        assert other.success,other.message
        assert abs(cap-(1-1/other.x[-1]))<tolerance
        assert np.max(abs(radial@other.x-center))<tolerance
    return cap,b,g_vector


def check_aux(x, available, g, tolerance):
    menu,alt=np.nonzero(available)
    origin=available.argmax(axis=1)
    full_rank=np.linalg.matrix_rank(x[menu,alt]-x[menu,origin[menu]])==x.shape[-1]
    assert full_rank
    assert np.all(g[menu,alt]>0)
    assert max(abs(g.sum(axis=1)-1))<tolerance and np.all(g[~available]==0)


def main(args):
    plan=read(ROOT/'safe_power_plan.json')
    flow=read(ROOT/'plan.json')
    assert all(digest(args.reference/k)==v for k,v in flow['input_files'].items())
    source=module_for(args.reference/'run.py')
    archive=read(args.reference/'plan.json')
    file_checks=source.audited_files(args,archive)
    records=lookup(args.reference/'results/allocations.csv')
    tolerance=plan['tolerance']
    rows=[]
    for dataset,trial,order,x,g,y,available,ids in chain(
      source.food(args,file_checks,archive['label_allocations']),source.swiss(args,file_checks,archive['label_allocations'])):
        for budget in archive['label_budgets']:
            take=np.isin(ids,order[:budget])
            cap,moment,center=labelled_depth(x[take],available[take],y[take],g[take],tolerance)
            # Numerical deployments use a fixed margin inside the exact safe bound.
            deployed=cap/2 if cap>tolerance else 0
            rng=np.random.default_rng(archive['auxiliary_generator_seed']+10007*trial+101*budget)
            away={i:0 for i in archive['powers']}
            proven_positive=0
            for draw in range(archive['auxiliary_draws']):
                unseen=np.isin(ids,rng.choice(order[budget:],size=archive['auxiliary_people'],replace=False))
                xu,au,gu=x[unseen],available[unseen],g[unseen]
                check_aux(xu,au,gu,tolerance)
                aux_center=np.einsum('ij,ijp->p',gu,xu)/len(xu)
                if deployed:
                    # Verify strict positive witness mixing the labelled moment with a proxy anchor.
                    x_l=x[take]
                    av_l=available[take]
                    m=len(x_l)
                    j,k=np.nonzero(av_l)
                    features=x_l[j,k]
                    positive=deployed*g[take][j,k]/m
                    block=vstack((coo_matrix((np.ones(len(j)),(j,np.arange(len(j)))),shape=(m,len(j))),
                                  csc_matrix(features.T)),format='csc')
                    rhs=np.r_[np.full(m,1/m),moment]
                    feasibility=linprog(np.zeros(len(j)),A_eq=block,b_eq=rhs,bounds=list(zip(positive,[None]*len(j))),
                                        options=OPTIONS,method='highs-ds')
                    assert feasibility.success
                    remaining=feasibility.x-positive
                    assert min(remaining)>-tolerance
                    mass=np.bincount(j,weights=remaining,minlength=m)
                    assert max(abs(mass-(1-deployed)/m))<tolerance
                    witness=features.T@remaining+deployed*aux_center
                    target=moment+deployed*(aux_center-center)
                    assert max(abs(witness-target))<tolerance
                    proven_positive+=1
                for power in archive['powers']:
                    state,nl,nu=records[dataset,trial,budget,draw,float(power)]
                    assert (nl,nu)==(take.sum(),unseen.sum())
                    if cap>power+tolerance:
                        assert state=='interior'
                    away[power]+=state=='outside'
            rows.append(dict(dataset=dataset,allocation=trial,label_people=budget,safe_cap=cap,deployed_power=deployed,
                             proven_positive_aux_designs=proven_positive,
                             observed_unit_external=away[1.0],observed_half_external=away[0.5]))
    args.out.mkdir()
    with (args.out/'records.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sides={}
    for dataset in ('RADIANT','Swissmetro','combined'):
        sample=[r for r in rows if dataset in (r['dataset'],'combined')]
        caps=[r['safe_cap'] for r in sample]
        sides[dataset]=dict(label_groups=len(sample),positive_cap=sum(c>tolerance for c in caps),
              at_least_half=sum(c>=0.5-tolerance for c in caps),
              median_all=float(np.median(caps)),
              median_positive=float(np.median([c for c in caps if c>tolerance])),
              positive_witnesses=sum(r['proven_positive_aux_designs'] for r in sample),
              archived_unit_external_with_positive_safe_power=sum(r['observed_unit_external'] for r in sample if r['deployed_power']>0),
              archived_half_external_with_positive_safe_power=sum(r['observed_half_external'] for r in sample if r['deployed_power']>0))
    assert len(rows)==96
    result=dict(status='passed',groups=len(rows),summary=sides,
                records_sha256=digest(args.out/'records.csv'),plan_sha256=digest(ROOT/'safe_power_plan.json'),
                script_sha256=digest(Path(__file__)))
    (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(sides))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('food-data','food-archive','swiss-data','swiss-archive','reference','out'):
        p.add_argument('--'+key,type=Path,required=True)
    with threadpool_limits(limits=1):
        main(p.parse_args())
