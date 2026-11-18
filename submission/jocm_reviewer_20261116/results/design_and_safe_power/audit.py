"""Evaluate response-independent recession obstructions and observed-case directions."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
import importlib.util
from itertools import chain
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def module_for(path):
    spec=importlib.util.spec_from_file_location('audit_fixed_label',path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def directions(p):
    basis=np.eye(p)
    base=-basis[0]-basis[1]
    vectors=[('asc_first',basis[0]),('asc_second',basis[1]),('asc_reference',base)]
    vectors.extend((f'attribute_{j+1}',basis[j]) for j in range(2,p))
    return [(name+suffix,sgn*beta,name.startswith('asc_')) for name,beta in vectors
            for suffix,sgn in (('+',1),('-',-1))]


def loss_slope(utilities, maximum, predictions, label, power):
    """Direct expression from the three losses, evaluated along a fixed ray."""
    if label is None:
        return power*np.mean(maximum-np.sum(predictions*utilities,axis=1))
    sample=np.arange(len(label))
    human=np.mean(maximum-utilities[sample,label])
    removal=power*np.mean(maximum-np.sum(predictions*utilities,axis=1))
    return human-removal


def audit_pair(x,available,g,take,unseen,y,plan,previous):
    tolerance=plan['tolerance']
    labels=y[take]
    gl,gu=g[take],g[unseen]
    x_l,x_u=x[take],x[unseen]
    av_l,av_u=available[take],available[unseen]
    table=[]
    for name,beta,asc in directions(x.shape[-1]):
        ul,uu=x_l@beta,x_u@beta
        al=np.where(av_l,ul,-np.inf)
        au=np.where(av_u,uu,-np.inf)
        h_l,h_u=al.max(axis=1),au.max(axis=1)
        width_l, width_u = (np.mean(h-np.where(mask,u,np.inf).min(axis=1))
                            for h,mask,u in ((h_l,av_l,ul),(h_u,av_u,uu)))
        gap=np.mean(h_u-np.sum(gu*uu,axis=1))-np.mean(h_l-np.sum(gl*ul,axis=1))
        # Independently recompute the difference in long precision as chosen-minus-proxy contrasts.
        first=al.argmax(axis=1)
        last=al.shape[1]-1-al[:,::-1].argmax(axis=1)
        gx=np.asarray(gu,dtype=np.longdouble)
        ulh=np.asarray(ul,dtype=np.longdouble)
        uuh=np.asarray(uu,dtype=np.longdouble)
        upper_alternative=au.argmax(axis=1)
        upper= np.sum(gx*(uuh[np.arange(len(uu)),upper_alternative,None]-uuh),axis=1).mean()
        for chosen in (first,last):
            lower=np.sum(np.asarray(gl,dtype=np.longdouble)*(ulh[np.arange(len(ul)),chosen,None]-ulh),axis=1).mean()
            assert abs(float(upper-lower)-gap)<tolerance
            for power in plan['powers_to_recheck']:
                direct=loss_slope(uu,h_u,gu,None,power)+loss_slope(ul,h_l,gl,chosen,power)
                assert abs(direct-power*gap)<tolerance*power
                if gap<-tolerance:
                    assert direct<-tolerance*power
        # The gap above depends solely on the design. Observed choices enter this comparison.
        regret=float(np.mean(h_l-ul[np.arange(len(labels)),labels]))
        assert regret>=0
        observed={}
        for power, reference in previous.items():
            original=loss_slope(uu,h_u,gu,None,power)+loss_slope(ul,h_l,gl,labels,power)
            assert abs(original-(regret+power*gap))<tolerance
            assert original>=-tolerance or reference=='outside'
            observed[power]=float(original)
        table.append(dict(direction=name,asc=asc,gap=float(gap),equal_width=abs(width_l-width_u)<tolerance,
                          regret=regret,observed=observed))
    worst=min(table,key=lambda it:it['gap'])
    worst_asc=min([it for it in table if it['asc']],key=lambda it:it['gap'])
    output=dict(any_direction=worst['direction'],any_gap=worst['gap'],any_regret=worst['regret'],
                asc_direction=worst_asc['direction'],asc_gap=worst_asc['gap'],
                asc_equal_width=any(it['asc'] and it['equal_width'] for it in table),
                any_state='uniform_failure_proved' if worst['gap']<-tolerance else 'unresolved',
                asc_state='uniform_failure_proved' if worst_asc['gap']<-tolerance else 'unresolved')
    for power, ref in previous.items():
        suffix='unit' if power==1 else 'half'
        observed=min(table,key=lambda it:it['observed'][power])
        observed_asc=min((it for it in table if it['asc']),key=lambda it:it['observed'][power])
        output[f'{suffix}_exact_support']=ref
        output[f'{suffix}_observed_direction']=observed['direction']
        output[f'{suffix}_observed_recession']=observed['observed'][power]
        output[f'{suffix}_observed_ASC_recession']=observed_asc['observed'][power]
    return output


def lookup(reference):
    with reference.open(newline='') as stream:
        entries=list(csv.DictReader(stream))
    keys={(r['dataset'],int(r['allocation']),int(r['label_people']),int(r['draw']),float(r['power'])):
          (r['status'],int(r['label_tasks']),int(r['aux_tasks'])) for r in entries}
    assert len(keys)==len(entries)==1536
    return keys


def main(args):
    plan=read(ROOT/'plan.json')
    assert all(digest(args.reference/k)==v for k,v in plan['input_files'].items())
    source=module_for(args.reference/'run.py')
    old_plan=read(args.reference/'plan.json')
    files=source.audited_files(args,old_plan)
    results=lookup(args.reference/'results/allocations.csv')
    rows=[]
    for dataset,trial,order,x,g,y,available,ids in chain(
            source.food(args,files,old_plan['label_allocations']),source.swiss(args,files,old_plan['label_allocations'])):
        for budget in old_plan['label_budgets']:
            take=np.isin(ids,order[:budget])
            rng=np.random.default_rng(old_plan['auxiliary_generator_seed']+10007*trial+101*budget)
            for draw in range(old_plan['auxiliary_draws']):
                unseen=np.isin(ids,rng.choice(order[budget:],size=old_plan['auxiliary_people'],replace=False))
                previous={}
                for power in old_plan['powers']:
                    state,nl,nu=results[dataset,trial,budget,draw,float(power)]
                    assert nl==take.sum() and nu==unseen.sum() and not (take&unseen).any()
                    previous[float(power)]=state
                rows.append(dict(dataset=dataset,allocation=trial,label_people=budget,draw=draw,
                                 aux_tasks=int(unseen.sum()),**audit_pair(x,available,g,take,unseen,y,plan,previous)))
    args.out.mkdir()
    with (args.out/'records.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries={}
    for name in ('RADIANT','Swissmetro','combined'):
        group=[r for r in rows if name in (r['dataset'],'combined')]
        sums=Counter()
        for row in group:
            sums['ASC_witnesses']+=row['asc_state']=='uniform_failure_proved'
            sums['any_witnesses']+=row['any_state']=='uniform_failure_proved'
            sums['same_ASC_width']+=row['asc_equal_width']
            for suffix in ('unit','half'):
                sums[f'{suffix}_external']+=row[f'{suffix}_exact_support']=='outside'
                sums[f'{suffix}_actual_diagnosed']+=row[f'{suffix}_observed_recession']<-plan['tolerance']
                sums[f'{suffix}_actual_ASC_diagnosed']+=row[f'{suffix}_observed_ASC_recession']<-plan['tolerance']
        summaries[name]=dict(sums)|dict(pairs=len(group),
          median_ASC_gap=float(np.median([r['asc_gap'] for r in group])),
          median_any_gap=float(np.median([r['any_gap'] for r in group])))
    receipt=dict(status='passed',pairs=len(rows),summaries=summaries,
                 records_sha256=digest(args.out/'records.csv'),plan_sha256=digest(ROOT/'plan.json'),
                 script_sha256=digest(Path(__file__)))
    assert len(rows)==768
    (args.out/'summary.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(summaries))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('food-data','food-archive','swiss-data','swiss-archive','reference','out'):
        parser.add_argument('--'+key,type=Path,required=True)
    with threadpool_limits(limits=1):
        main(parser.parse_args())
