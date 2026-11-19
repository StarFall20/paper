"""Read published fixed predictions and reconstruct the deposited prompt design."""
from contextlib import contextmanager
from hashlib import sha256
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
COVARIATES=['age','gender','education','income_range','hh_under18','hh_18to65','hh_over65']
QUOTAS={'alta':15,'media':10,'baixa':5}


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


@contextmanager
def dependency_dir(path):
    sys.path.insert(0,str(path))
    try:
        yield
    finally:
        assert sys.path.pop(0)==str(path)


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    with dependency_dir(path.parent):
        spec.loader.exec_module(result)
    return result


def dependencies(folder):
    plan=read(ROOT/'plan.json')
    assert all(sha(folder/n)==value for n,value in plan['code_sha256'].items())
    modules={
       'external':module(folder/'external/audit_released_workflow.py','released_alves'),
       'support':module(folder/'support/run.py','previous_full_support'),
       'certificate':module(folder/'certificate/safe_power.py','previous_label_cap'),
       'forecast':module(folder/'forecast/fit.py','previous_optimizer'),
    }
    return plan,SimpleNamespace(**modules)


def prompt_pool(core,context,tasks,rep):
    people=pd.read_excel(io.BytesIO(core.read(context+'.xlsx')),'survey')
    people.columns=[{'labl':'label'}.get(str(c).strip(),str(c).strip()) for c in people.columns]
    for column in COVARIATES[-3:]:
        if column in people:
            people[column]=people[column].clip(upper=3)
    covariates=[name for name in COVARIATES if name in people]
    people=people[['unique_id']+covariates].drop_duplicates('unique_id').set_index('unique_id').fillna(-99)
    people=people.loc[sorted({pid for pid,_ in tasks}&set(people.index))]
    rng=np.random.default_rng(20261003+1000*rep+sum(map(ord,context)))
    kind=people.age.isin([1,2])
    target=rng.permutation(people.index[kind].to_numpy())
    other=rng.permutation(people.index[~kind].to_numpy())
    donor_total=len(people)-70-70-40
    donors={}
    for coverage,quota in QUOTAS.items():
        assert len(target[50:])>=quota and len(other[90:])>=donor_total-quota
        donors[coverage]=np.r_[target[50:50+quota],other[90:90+donor_total-quota]]
        assert len(donors[coverage])==donor_total
        assert not set(donors[coverage]) & set(target[:50])
    return people,target[:35],target[35:50],donors


def example_keys(people, donor, targets, tasks, rep):
    known={}
    for key in sorted(tasks):
        if key[0] in donor:
            known.setdefault(key[0],[]).append(key)
    profiles=people.loc[donor].to_numpy()
    exact={}
    for person in targets:
        profile=people.loc[person].to_numpy()
        # Tie breaking is reset to the deposited seed for each target.
        scores=np.sum(profiles!=profile,axis=1).astype(float)
        scores+=np.random.default_rng(20261003+rep).random(len(profiles))*1e-3
        exact[person]=tuple(key for index in np.argsort(scores)[:4] for key in known[donor[index]][:2])
        assert len(exact[person])==8
    return exact


def context_offset(context,rep):
    return 1000*rep+sum(map(ord,context))


def cases(args,plan,backend):
    origin=backend.external
    path=args.inputs
    for name,want in origin.HASHES.items():
        assert sha(path/name)==want
    with ZipFile(path/origin.CORE) as core,ZipFile(path/origin.OUTPUT) as outputs:
        pipeline=next(n for n in core.namelist() if n.endswith('estudo2_pipeline.py'))
        assert sha256(core.read(pipeline)).hexdigest()==plan['sources']['required_original_pipeline_sha256']
        def payload(name):
            return outputs.read(next(n for n in outputs.namelist() if n.endswith('/'+name)))
        fixed=payload('previsoes_llm.csv')
        assert sha256(fixed).hexdigest()==json.loads(payload('congelamento.json'))['hash_previsoes']
        previous=pd.read_csv(io.BytesIO(fixed))
        assert len(previous)==7200
        for context,collections in previous.groupby('contexto',sort=True):
            people_age,tasks,names,_=origin.read_table(core,context)
            for rep,split in collections.groupby('rep',sort=True):
                rep=int(rep)
                people,h,v,donors=prompt_pool(core,context,tasks,rep)
                h0,v0=origin.split_target(people_age,context,rep)
                assert set(h)==h0 and set(v)==v0
                evaluate_folds=np.random.default_rng(plan['evaluation']['seed']+context_offset(context,rep)).permutation(h).reshape(5,7)
                small_labels=np.random.default_rng(plan['labels']['seed']+context_offset(context,rep)).permutation(v).reshape(3,5)
                groups=[(-1,v)]+list(enumerate(small_labels))
                for coverage,part in split.groupby('cond',sort=True):
                    assert set(part.pid)==set(h)|set(v) and len(part)==300
                    data=part.set_index(['pid','card'])
                    x,y,g,pid,missing=origin.subset(tasks,data,set(h)|set(v))
                    assert not missing and np.max(abs(g.sum(axis=1)-1))<1e-12
                    assert np.all(g>0)
                    g=g/g.sum(axis=1,keepdims=True)
                    all_examples=example_keys(people,donors[coverage],np.r_[h,v],tasks,rep)
                    for label_index,labels in groups:
                        for test_index,test_people in enumerate(evaluate_folds):
                            train_u=set(h)-set(test_people)
                            assert len(train_u)==28
                            l,u,e=(np.isin(pid,list(people_here)) for people_here in (labels,train_u,test_people))
                            assert l.sum()==6*len(labels) and u.sum()==168 and e.sum()==42
                            assert not ((l&u)|(l&e)|(u&e)).any()
                            examples=sorted({key for person in np.r_[labels,list(train_u)] for key in all_examples[person]})
                            assert not {person for person,_ in examples} & (set(h)|set(v))
                            ex_x=np.stack([tasks[key][0] for key in examples])
                            ex_y=np.eye(g.shape[1])[[tasks[key][1] for key in examples]]
                            yield SimpleNamespace(
                                key=(context,rep,coverage,len(labels),label_index,test_index), names=names,
                                label=(x[l],y[l],g[l],pid[l]), aux=(x[u],g[u],pid[u]),
                                eval=(x[e],y[e],g[e],pid[e]), prompt=(ex_x,ex_y),
                                prompt_people=len({key[0] for key in examples}),prompt_menus=len(examples))
