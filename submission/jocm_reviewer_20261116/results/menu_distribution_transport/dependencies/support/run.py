"""Compare support while holding labels, predictor, power and auxiliary counts fixed."""
import argparse
from collections import Counter
from hashlib import sha256
import importlib.util
from itertools import chain
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csc_matrix, hstack, vstack
from scipy.special import softmax
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
TOL = 1e-8


def digest(p):
    return sha256(p.read_bytes()).hexdigest()


def load(module_path, name):
    spec=importlib.util.spec_from_file_location(name, module_path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(path.read_text())


def audited_files(args, plan):
    files = dict(food_model=args.food_archive/'analysis/existing_panel_model.py',
                 food_plan=args.food_archive/'inputs/model_plan.json',
                 food_prediction=args.food_archive/'inputs/preflight.json',
                 swiss_model=args.swiss_archive/'audit.py',
                 swiss_plan=args.swiss_archive/'plan.json',
                 swiss_prediction=args.swiss_archive/'results/summary.json')
    assert {name:digest(p) for name,p in files.items()}==plan['source_files_sha256']
    return files


def food(args, files, trials):
    source=load(files['food_model'],'fixed_label_food')
    ref=read(files['food_plan'])
    data=source.load(args.food_data,ref)
    pool=data['pool']
    x,g,y=data['f'],softmax(data['teacher']@np.asarray(read(files['food_prediction'])['teacher_coefficients']),axis=2),data['y']
    rng=np.random.default_rng(ref['allocation_seed'])
    sizes=np.full(len(y),y.shape[1])
    order=np.asarray([rng.permutation(pool) for _ in range(trials)])
    for trial, respondent_order in enumerate(order):
        yield 'RADIANT',trial,respondent_order,x.reshape(-1,*x.shape[-2:]),g.reshape(-1,g.shape[-1]),y.ravel(),np.ones((y.size,g.shape[-1]),bool),np.repeat(np.arange(len(y)),sizes)


def swiss(args, files, trials):
    source=load(files['swiss_model'],'fixed_label_swiss')
    ref=read(files['swiss_plan'])
    assert digest(args.swiss_data)==ref['data_sha256']
    ids,y,available,features=source.load_data(args.swiss_data)
    shuffled=np.random.default_rng(ref['split_seed']).permutation(np.unique(ids))
    pool=shuffled[ref['teacher_count']+ref['evaluation_count']:]
    mask=np.isin(ids,pool)
    ids,y,available=ids[mask],y[mask],available[mask]
    x=source.center_features(features['canonical_4'][mask],available)
    coefficient=np.asarray(read(files['swiss_prediction'])['input_summary']['teacher_coefs'])
    g=source.probability(x,available,coefficient)
    for trial in range(trials):
        order=np.random.default_rng(ref['allocation_seed']+trial).permutation(pool)
        yield 'Swissmetro',trial,order,x,g,y,available,ids


def power_inputs(x, available, y, g, take, unseen, power):
    labelled_x = x[take]
    human = labelled_x[np.arange(take.sum()),y[take]].mean(axis=0)
    labelled_proxy = np.einsum('ij,ijp->p',g[take],labelled_x)/take.sum()
    if power==0:
        return labelled_x,available[take],np.full(take.sum(),1/take.sum()),human
    aux_x = x[unseen]
    aux_proxy = np.einsum('ij,ijp->p',g[unseen],aux_x)/unseen.sum()
    target = human+power*(aux_proxy-labelled_proxy)
    if power==1:
        return aux_x,available[unseen],np.full(unseen.sum(),1/unseen.sum()),target
    return (np.concatenate((labelled_x,aux_x)),np.concatenate((available[take],available[unseen])),
            np.r_[np.full(len(labelled_x),(1-power)/len(labelled_x)),np.full(len(aux_x),power/len(aux_x))],target)


def support(x, available, weights, target):
    menu,alternative=np.nonzero(available)
    n,_,p=x.shape
    row_count=available.sum(axis=1)
    centered=x[menu,alternative]-(np.where(available[:,:,None],x,0).sum(axis=1)/row_count[:,None])[menu]
    if np.linalg.matrix_rank(centered)<p:
        return 'rank_deficient',None,None
    features=x[menu,alternative]
    nc=len(menu)
    totals=coo_matrix((np.ones(nc),(menu,np.arange(nc))), shape=(n,nc))
    primary=vstack((totals, csc_matrix(features.T)),format='csc')
    moment_share=weights[menu]/row_count[menu]
    share_coeff=np.r_[weights,features.T@moment_share]
    mixture=hstack((primary,csc_matrix(share_coeff[:,None])),format='csc')
    sol=linprog(np.r_[np.zeros(nc),-1.],A_eq=mixture,b_eq=np.r_[weights,target],
                bounds=[(0,None)]*nc+[(0,1)],method='highs-ipm')
    limits=coo_matrix((np.tile([-1.,1.],nc),
                      (np.repeat(np.arange(nc),2),np.column_stack((np.arange(nc),np.full(nc,nc))).ravel())),
                      shape=(nc,nc+1)).tocsc()
    alternate=linprog(np.r_[np.zeros(nc),-1.],A_eq=hstack((primary,csc_matrix((n+p,1)))),
                      b_eq=np.r_[weights,target],A_ub=limits,b_ub=np.zeros(nc),
                      bounds=[(0,None)]*(nc+1),method='highs-ds')
    if sol.status==2 and alternate.status==2:
        directions=coo_matrix((-np.ones(nc),(np.arange(nc),menu)),shape=(nc,n)).tocsc()
        rule=hstack((csc_matrix(features),directions),format='csc')
        ray=linprog(np.r_[-target,weights],A_ub=rule,b_ub=np.zeros(nc),
                    bounds=[(-1,1)]*p+[(None,None)]*n,method='highs-ds')
        assert ray.success,ray.message
        beta=ray.x[:p]
        exact=weights@np.max(np.where(available,x@beta,-np.inf),axis=1)-target@beta
        assert exact<-TOL and abs(exact-ray.fun)<TOL
        return 'outside',None,float(exact)
    if sol.success and alternate.success:
        q=sol.x[:-1]+sol.x[-1]*moment_share
        assert np.max(abs(primary@q-np.r_[weights,target]))<TOL
        assert np.max(abs(primary@alternate.x[:-1]-np.r_[weights,target]))<TOL
        same_sign = ((sol.x[-1]>TOL) == (alternate.x[-1]>TOL/n))
        assert same_sign,(sol.x[-1],alternate.x[-1])
        return ('interior' if sol.x[-1]>TOL else 'boundary_or_unresolved'),float(sol.x[-1]),None
    raise RuntimeError((sol.status,sol.message,alternate.status,alternate.message))


def results_for_group(data, plan):
    dataset,trial,order,x,g,y,available,ids=data
    rows=[]
    for budget in plan['label_budgets']:
        take=np.isin(ids,order[:budget])
        ordinary,*_=support(*power_inputs(x,available,y,g,take,None,0))
        rng=np.random.default_rng(plan['auxiliary_generator_seed']+10007*trial+101*budget)
        for j in range(plan['auxiliary_draws']):
            selected=rng.choice(order[budget:],size=plan['auxiliary_people'],replace=False)
            unseen=np.isin(ids,selected)
            assert not (take&unseen).any() and len(np.unique(ids[unseen]))==plan['auxiliary_people']
            for power in plan['powers']:
                status,margin,ray=support(*power_inputs(x,available,y,g,take,unseen,power))
                rows.append(dict(dataset=dataset,allocation=trial,label_people=budget,aux_people=len(selected),
                                 draw=j,power=power,status=status,labelled_only=ordinary,label_tasks=int(take.sum()),
                                 aux_tasks=int(unseen.sum()),relative_interior_fraction=margin,recession=ray))
    return rows


def main(args):
    plan=read(ROOT/'plan.json')
    files=audited_files(args,plan)
    data=list(chain(food(args,files,plan['label_allocations']),swiss(args,files,plan['label_allocations'])))
    table=[]
    for group in data:
        table.extend(results_for_group(group,plan))
    args.out.mkdir()
    with (args.out/'allocations.csv').open('w',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    summaries=[]
    for dataset in ('RADIANT','Swissmetro'):
        for budget in plan['label_budgets']:
            for power in plan['powers']:
                sub=[row for row in table if (row['dataset'],row['label_people'],row['power'])==(dataset,budget,power)]
                counts=Counter(row['status'] for row in sub)
                flip=0
                ordinary_flip=0
                for trial in range(plan['label_allocations']):
                    rows=[row for row in sub if row['allocation']==trial]
                    assert len(rows)==plan['auxiliary_draws']
                    states={row['status'] for row in rows}
                    if {'interior','outside'}<=states:
                        flip+=1
                        ordinary_flip+=rows[0]['labelled_only']=='interior'
                summaries.append(dict(dataset=dataset,label_people=budget,power=power,ordinary_interior_groups=sum(
                    row['labelled_only']=='interior' for row in sub if row['draw']==0),
                    fixed_label_sets=plan['label_allocations'],aux_draws_per_set=plan['auxiliary_draws'],
                    observable_statuses=dict(counts),sets_with_interior_and_exterior=flip,
                    sets_with_interior_and_exterior_where_ordinary_finite=ordinary_flip))
    expected=2*plan['label_allocations']*len(plan['label_budgets'])*plan['auxiliary_draws']*len(plan['powers'])
    assert len(table)==expected
    result=dict(state='passed',corrected_allocation_records=len(table),
                verified_exterior_rays=sum(row['recession'] is not None for row in table),
                summaries=summaries,allocations_sha256=digest(args.out/'allocations.csv'),
                plan_sha256=digest(ROOT/'plan.json'),script_sha256=digest(Path(__file__)))
    (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('food-data','food-archive','swiss-data','swiss-archive','out'):
        p.add_argument('--'+arg,type=Path,required=True)
    with threadpool_limits(limits=1):
        main(p.parse_args())
