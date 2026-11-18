"""Repeat the fixed-design support check at the archived proxy probabilities."""
import argparse
from collections import Counter
import csv
from itertools import chain
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from audit import screen_probabilities
from inputs import food, swiss, read, rows, sha

ROOT=Path(__file__).resolve().parent


def run(args):
    primary=read(args.primary/'summary.json')
    margin=read(ROOT/'plan.json')['witness_tolerance']
    equality=read(ROOT/'plan.json')['equality_tolerance']
    table=rows(args.primary/'records.csv')
    assert sha(args.primary/'records.csv')==primary['records_sha256']
    assert primary['input_sha256']=={'food':sha(args.food_data),'swissmetro':sha(args.swiss_data)}
    results=[]
    for case, archived in zip(chain(food(args.food_data,args.food_archive),swiss(args.swiss_data,args.swiss_archive)),table,strict=True):
        assert [case.dataset,case.specification,case.allocation,case.labelled_people,case.exact_state] == [
            archived['dataset'],archived['specification'],int(archived['allocation']),int(archived['labelled_people']),archived['reference']]
        q=screen_probabilities(case.features,case.available,case.moment,case.auxiliary_proxy,margin,equality)
        determined=q['screen']!='needs_support_program'
        old=archived['screen']!='needs_support_program'
        assert q['screen'] in (case.exact_state,'needs_support_program')
        results.append(dict(dataset=case.dataset,specification=case.specification,allocation=case.allocation,
                            labelled_people=case.labelled_people,reference=case.exact_state,proxy_decides=determined,
                            pilot_decides=old,either_decides=determined or old,**q))
    args.out.mkdir(parents=True)
    with (args.out/'records.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    totals={}
    for dataset in ('RADIANT','Swissmetro','combined'):
        counts=Counter()
        for r in results:
            if dataset=='combined' or r['dataset']==dataset:
                counts[(r['reference'],'denominator')]+=1
                for label in ('proxy_decides','pilot_decides','either_decides'):
                    counts[(r['reference'],label)]+=r[label]
        totals[dataset]={key:{metric:counts[key,metric] for metric in ('denominator','proxy_decides','pilot_decides','either_decides')}
                         for key in ('interior','outside')}
    receipt=dict(state='passed', exploratory=True, totals=totals, record_count=len(results),
                 primary_sha256=sha(args.primary/'summary.json'), records_sha256=sha(args.out/'records.csv'),
                 plan_sha256=sha(ROOT/'frozen_proxy_sensitivity_plan.json'))
    (args.out/'summary.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(totals))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('food-data','food-archive','swiss-data','swiss-archive','primary','out'):
        parser.add_argument('--'+field,type=Path,required=True)
    with threadpool_limits(limits=1):
        run(parser.parse_args())
