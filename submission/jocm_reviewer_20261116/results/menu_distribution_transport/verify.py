"""Recalculate intervention scores and losses from uncompressed choice menus."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import groupby
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
KEY = ('context', 'rep', 'coverage', 'label_people', 'label_fold', 'eval_fold')


def normalized(x):
    output = []
    for a in x:
        anchor = min(map(tuple,a))
        centered = a-np.asarray(anchor)
        output.append((centered,tuple(sorted(map(tuple,centered)))))
    return output


def lookup(menus, forecast):
    raw = defaultdict(list)
    for (a,key),probs in zip(menus,forecast,strict=True):
        raw[key].append(probs@a)
    return {k:np.mean(v,axis=0) for k,v in raw.items()}


def softmax(a,beta):
    u=a@beta
    u-=np.max(u,axis=1,keepdims=True)
    expu=np.exp(u)
    return u-np.log(expu.sum(axis=1,keepdims=True))


def mean_moment(x, weights):
    return (weights[:,:,None]*x).sum(axis=1).mean(axis=0)


def support(backend, x, target):
    return backend.support.support(x,np.ones(x.shape[:2],bool),np.full(len(x),1/len(x)),target)[0]


def validate(case, rec, old, backend, plan):
    xl,y,gl,_=case.label
    xu,gu,_=case.aux
    holdout,outcome,_,people=case.eval
    l,u=normalized(xl),normalized(xu)
    lx,ux=[np.stack([a for a,_ in sample]) for sample in (l,u)]
    lm,um=lookup(l,gl),lookup(u,gu)
    common=lm.keys()&um.keys()
    assert rec['common_menus']==len(common) and (rec['labelled_menus'],rec['auxiliary_menus'])==(len(lm),len(um))
    for field, sample in [('labelled_coverage',l),('auxiliary_coverage',u)]:
        assert abs(rec[field]-np.mean([k in common for _,k in sample]))<1e-12
    contrast=np.mean([um[k]-lm[k] if k in common else np.zeros(xl.shape[-1]) for _,k in l],axis=0)
    pooled={k:(lm[k]+um[k])/2 if k in common else lm.get(k,um.get(k)) for k in lm.keys()|um.keys()}
    balance=np.mean([pooled[k] for _,k in u],axis=0)-np.mean([pooled[k] for _,k in l],axis=0)
    truth=mean_moment(lx,y)
    reference=old['fixed_unit']
    assert rec['unit']['loss']==reference['loss']
    assert rec['ridge']==old['labelled_ridge']['loss'] and rec['matched_prompt_MLE']==old['prompt_matched_MLE']['loss']
    assert rec['ordinary']==old['labelled_MLE']['loss']
    unknown=mean_moment(ux,gu)-mean_moment(lx,gl)
    original=support(backend,ux,truth+unknown)
    assert original==rec['unit']['state']
    for name,x,target in [('poststratified',lx,truth+contrast),('composition_only',ux,truth+balance)]:
        r=rec[name]
        assert r['state']==support(backend,x,target)
        if r['state']=='interior':
            beta=np.asarray(r['beta']); adjusted=np.exp(softmax(x,beta))
            grad=mean_moment(x,adjusted)-target
            assert np.max(abs(grad))<plan['score_tol'] and abs(np.max(abs(grad))-r['score'])<1e-12
            task=-softmax(holdout,beta)[outcome.astype(bool)]
            loss=np.mean([task[people==pid].mean() for pid in np.unique(people)])
            assert abs(loss-r['loss'])<1e-11
        else:
            assert r['beta'] is None and r['loss']==rec['ridge']
    if original=='outside':
        r=rec['recession']; d=np.asarray(r['direction'])
        hs=lambda x: np.max(x@d,axis=1).mean()
        attributes=dict(observed=hs(ux)-(truth+unknown)@d, human=hs(lx)-truth@d,
                        composition=hs(ux)-hs(lx)-balance@d, within=-(unknown-balance)@d,
                        transported=hs(lx)-(truth+contrast)@d)
        assert attributes['observed'] < -plan['recession_tol']
        assert all(abs(r[key]-v)<1e-10 for key,v in attributes.items())
    else:
        assert 'recession' not in rec


def summarize(rows):
    methods=('unit','poststratified','composition_only')
    n=len(rows)
    repro=dict(cases=n)
    for key in ('labelled_coverage','auxiliary_coverage'):
        counts=np.asarray([r[key] for r in rows]); repro[key]=[float(a) for a in (counts.min(),np.median(counts),counts.max())]
    repro['states']={k:dict(Counter(r[k]['state'] for r in rows)) for k in methods}
    repro['transitions']={k:dict(Counter(r['unit']['state']+' -> '+r[k]['state'] for r in rows)) for k in methods[1:]}
    bounds=[r['recession'] for r in rows if 'recession' in r]; tol=1e-8
    repro['negative_components']={k:sum(r[k]<-tol for r in bounds) for k in ('observed','human','composition','within','transported')}
    repro['mean_gap_components']={k:float(np.mean([r[k] for r in bounds])) for k in ('observed','human','composition','within','transported')}
    repro['paired_log_loss']={k:{} for k in methods}
    for k in methods:
        for b in ('unit','ridge','matched_prompt_MLE'):
            values=np.array([r[k]['loss']-(r[b]['loss'] if b=='unit' else r[b]) for r in rows])
            repro['paired_log_loss'][k][b]=dict(mean=float(values.mean()),wins=int((values<-1e-10).sum()),ties=int((abs(values)<=1e-10).sum()))
    return repro


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('upstream','dependencies','inputs','reference','results','out'):
        parser.add_argument('--'+k,required=True,type=Path)
    args=parser.parse_args()
    sys.path.insert(0,str(args.upstream.resolve()))
    from source_data import cases, dependencies, read, sha
    plan=read(ROOT/'plan.json'); assert sha(args.reference)==plan['reference_fits_sha256']
    old_plan,backend=dependencies(args.dependencies)
    reference={k:{r['method']:r for r in v} for k,v in groupby(read(args.reference),key=lambda r:tuple(r[n] for n in KEY))}
    raw=read(args.results)
    records={tuple(r[k] for k in KEY):r for r in raw}
    assert len(records)==len(raw)==len(reference)==480 and set(reference)==set(records)
    seen=set()
    with threadpool_limits(limits=1):
        for case in cases(args,old_plan,backend):
            validate(case,records[case.key],reference[case.key],backend,plan)
            seen.add(case.key)
    assert seen==set(records)
    out=dict(status='passed',cases=len(seen),results_sha256=sha256(args.results.read_bytes()).hexdigest(),
             budgets={str(m):summarize([r for r in raw if r['label_people']==m]) for m in (15,5)})
    args.out.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print('Independently verified 480 original-menu designs and both intervention fits')


if __name__=='__main__':
    main()
