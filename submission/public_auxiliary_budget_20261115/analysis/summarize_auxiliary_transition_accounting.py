"""Reconcile all endpoint states, retaining zero-power nonattainment explicitly."""
from pathlib import Path
from collections import defaultdict, Counter
import argparse, csv, gzip, hashlib, json

p=argparse.ArgumentParser()
p.add_argument('--base',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
records=a.base/'results/records.csv.gz'
if records.exists():
    stream=gzip.open(records,'rt',newline='')
else:
    records=a.base/'results/records.csv'
    stream=records.open(newline='')
groups=defaultdict(dict)
with stream:
    rows=list(csv.DictReader(stream))
for r in rows:
    key=(int(r['label_rows']),r['prediction'],r['method'])
    pos=(int(r['trial']),int(r['auxiliary_rows']))
    assert pos not in groups[key]
    groups[key][pos]=r
assert len(rows)==36000
out=[]
retuned_union=set()
for (m,pred,method),g in sorted(groups.items()):
    ns=sorted({n for _,n in g});trials=sorted({t for t,_ in g})
    assert len(ns)==6 and trials==list(range(500))
    first,last=ns[0],ns[-1]
    flow=Counter();lost_exterior=[];lost_zero=[];gained=[]
    for t in trials:
        x,y=g[t,first],g[t,last]
        flow[x['status'],y['status']]+=1
        if x['status']=='interior' and y['status']!='interior':
            if y['status']=='outside_mean_support':lost_exterior.append(t)
            else:
                assert y['status']=='boundary_or_near_boundary'
                assert float(y['power'])==0
                assert y['ordinary_status']=='nonattained_separated'
                lost_zero.append(t)
        if x['status']!='interior' and y['status']=='interior':gained.append(t)
    f0=sum(g[t,first]['status']=='interior' for t in trials)
    f1=sum(g[t,last]['status']=='interior' for t in trials)
    assert f1-f0==len(gained)-len(lost_exterior)-len(lost_zero)
    if method=='source_tuned_each_N':
        retuned_union.update((m,t) for t in lost_exterior+lost_zero)
    out.append(dict(label_rows=m,prediction=pred,method=method,allocations=500,
        first_auxiliary_rows=first,last_auxiliary_rows=last,
        finite_first=f0,finite_last=f1,
        finite_to_exterior_trials=lost_exterior,finite_to_zero_power_trials=lost_zero,
        total_finite_loss=len(lost_exterior)+len(lost_zero),
        nonfinite_to_finite_trials=gained,
        all_endpoint_transitions=[dict(from_status=x,to_status=y,count=n) for (x,y),n in sorted(flow.items())]))
report=dict(status='complete',scope='First-to-last state accounting of the existing 36000 records; no refitting.',
    primary_record_sha256=hashlib.sha256(records.read_bytes()).hexdigest(),
    existing_record_count=len(rows),groups=out,
    retuned_distinct_loss_allocations=[dict(label_rows=m,trial=t) for m,t in sorted(retuned_union)],
    note='Finite-to-exterior counts alone omit losses caused by selecting exactly zero power on a separated human design.',
    new_human_or_AI_answers=0)
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(records=len(rows),groups=len(out),retuned_distinct_loss_allocations=len(retuned_union))))
