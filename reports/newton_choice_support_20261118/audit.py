"""Evaluate a single-solve support screen on fixed archived MNL objectives."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from itertools import chain
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from inputs import food, swiss, sha, checked_manifest


ROOT=Path(__file__).resolve().parent


def screen(features,available,target,pilot,margin,equality):
    utilities=np.where(available,features@pilot,-np.inf)
    return screen_probabilities(features,available,target,softmax(utilities,axis=1),margin,equality)


def screen_probabilities(features,available,target,probabilities,margin,equality):
    menu_mean=np.einsum('nj,njp->np',probabilities,features)
    grad=menu_mean.mean(axis=0)-target
    dev=features-menu_mean[:,None,:]
    n=len(features)
    hessian=np.einsum('nj,nja,njb->ab',probabilities,dev,dev)/n
    eig=np.linalg.eigvalsh(hessian)
    assert eig[0]>1e-10
    chol=np.linalg.cholesky(hessian)
    standardized=solve_triangular(chol,grad,lower=True)
    d=-solve_triangular(chol.T,standardized)
    decrement=np.linalg.norm(standardized)
    transformed=solve_triangular(chol,dev[available].T,lower=True)
    norm_product=np.linalg.norm(transformed,axis=0).max()*decrement
    probabilities_at_step=probabilities*(1+dev@d)
    row_balance=float(np.max(abs(probabilities_at_step.sum(axis=1)-1)))
    attained=bool(probabilities_at_step[available].min()>margin)
    moment=np.einsum('nj,njp->p',probabilities_at_step,features)/n
    moment_error=float(np.max(abs(moment-target)))
    assert row_balance<equality and moment_error<equality
    if norm_product<1-1e-6:
        assert probabilities_at_step[available].min()>0
    norm=np.linalg.norm(d)
    if norm>0:
        scaled=d/norm
        directional=float(np.max(np.where(available,features@scaled,-np.inf),axis=1).mean()-target@scaled)
    else:
        directional=0.
    outside=directional < -margin
    assert not (attained and outside)
    state='interior' if attained else 'outside' if outside else 'needs_support_program'
    return dict(screen=state,minimum_hessian_eigenvalue=float(eig[0]),hessian_condition=float(eig[-1]/eig[0]),
                newton_decrement=float(decrement),norm_sufficient_factor=float(norm_product),step_norm=float(norm),
                minimum_probability_at_step=float(probabilities_at_step[available].min()),
                maximum_probability_row_error=row_balance,maximum_moment_error=moment_error,
                normalized_newton_recession=directional)


def summary_table(records):
    cells=defaultdict(list)
    for r in records:
        cells[(r['dataset'],r['specification'],r['labelled_people'])].append(r)
    out=[]
    for (dataset,spec,budget),rows in cells.items():
        out.append(dict(dataset=dataset,specification=spec,labelled_people=budget,designs=len(rows),
                         counts_reference=dict(Counter(r['reference'] for r in rows)),
                         screen_by_reference={truth:dict(Counter(r['screen'] for r in rows if r['reference']==truth))
                                              for truth in ('interior','outside')},
                         norm_passes=int(sum(r['norm_sufficient_factor']<1 for r in rows))))
    return out


def main(args):
    plan_path=ROOT/'plan.json'
    plan=json.loads(plan_path.read_text())
    checked={
        'food':checked_manifest(args.food_archive,['analysis/existing_panel_model.py','inputs/model_plan.json',
              'inputs/preflight.json','inputs/baseline_records.csv','inputs/labelled_ridge_coefficients.json']),
        'swissmetro':checked_manifest(args.swiss_archive,['audit.py','plan.json','results/allocations.csv',
                      'results/coefficients.json','results/summary.json']),
    }
    records=[]
    for case in chain(food(args.food_data,args.food_archive),swiss(args.swiss_data,args.swiss_archive)):
        row=screen(case.features,case.available,case.moment,case.pilot,plan['witness_tolerance'],plan['equality_tolerance'])
        assert row['screen'] in [case.exact_state,'needs_support_program'],case
        records.append(dict(dataset=case.dataset,specification=case.specification,allocation=case.allocation,
                            labelled_people=case.labelled_people,reference=case.exact_state,**row))
    assert len(records)==2576 and sum(r['reference']=='outside' for r in records)==344
    args.out.mkdir(parents=True)
    with (args.out/'records.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    totals={}
    for dataset in ('RADIANT','Swissmetro','combined'):
        byref={}
        sub=[r for r in records if dataset=='combined' or r['dataset']==dataset]
        for truth in ('interior','outside'):
            cell=[r for r in sub if r['reference']==truth]
            byref[truth]=dict(designs=len(cell),decisive=sum(r['screen']==truth for r in cell),
                              norm_passes=sum(r['norm_sufficient_factor']<1 for r in cell),
                              median_Hessian_condition=float(np.median([r['hessian_condition'] for r in cell])),
                              finite_Newton_direction=sum(np.isfinite(r['step_norm']) for r in cell))
            byref[truth]['finite_Newton_direction']=int(byref[truth]['finite_Newton_direction'])
        totals[dataset]=byref
    meta=dict(state='passed',record_count=len(records),incorrect_decisions=0,
              totals=totals,summary=summary_table(records),
              largest_moment_balance_error=max(r['maximum_moment_error'] for r in records),
              largest_menu_balance_error=max(r['maximum_probability_row_error'] for r in records),
              input_sha256={'food':sha(args.food_data),'swissmetro':sha(args.swiss_data)},
              input_archive_sha256=checked, plan_sha256=sha(plan_path),records_sha256=sha(args.out/'records.csv'))
    (args.out/'summary.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
    print(json.dumps(totals))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('food-data','food-archive','swiss-data','swiss-archive','out'):
        parser.add_argument('--'+flag,type=Path,required=True)
    with threadpool_limits(limits=1):
        main(parser.parse_args())
