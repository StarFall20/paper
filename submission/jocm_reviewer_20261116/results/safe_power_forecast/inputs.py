"""Rebuild the existing labelled, auxiliary and evaluation panels."""
import csv
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

ROOT = Path(__file__).resolve().parent


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def table(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class Panel:
    name: str
    x: np.ndarray
    available: np.ndarray
    y: np.ndarray
    proxy: np.ndarray
    ids: np.ndarray
    train: np.ndarray
    test: np.ndarray
    allocation_orders: list


def food(args,files,archive):
    source=load(files['food_model'],'forecast_food')
    specification=read(files['food_plan'])
    subject=source.load(args.food_data,specification)
    training,evaluation,pool=(subject[k] for k in ('train','test','pool'))
    assert tuple(map(len,(training,evaluation,pool)))==(100,60,138)
    x=subject['f'].reshape(-1,3,11)
    teacher=subject['teacher'].reshape(-1,3,33)
    beta=np.asarray(read(files['food_prediction'])['teacher_coefficients'])
    y=subject['y'].ravel()
    ids=np.repeat(np.arange(len(subject['y'])),6)
    rng=np.random.default_rng(specification['allocation_seed'])
    orders=[rng.permutation(pool) for _ in range(archive['label_allocations'])]
    return Panel('RADIANT',x,np.ones(x.shape[:2],bool),y,softmax(teacher@beta,axis=1),
                 ids,np.isin(ids,training),np.isin(ids,evaluation),orders)


def swiss(args,files,archive):
    source=load(files['swiss_model'],'forecast_swiss')
    specification=read(files['swiss_plan'])
    assert digest(args.swiss_data)==specification['data_sha256']
    ids,y,available,features=source.load_data(args.swiss_data)
    x=source.center_features(features['canonical_4'],available)
    beta=np.asarray(read(files['swiss_prediction'])['input_summary']['teacher_coefs'])
    order=np.random.default_rng(specification['split_seed']).permutation(np.unique(ids))
    train,test,pool=np.split(order,[600,840])
    assert tuple(map(len,(train,test,pool)))==(600,240,351)
    orders=[np.random.default_rng(specification['allocation_seed']+j).permutation(pool)
            for j in range(archive['label_allocations'])]
    return Panel('Swissmetro',x,available,y,source.probability(x,available,beta),ids,
                 np.isin(ids,train),np.isin(ids,test),orders)


def sources(args):
    plan=read(ROOT/'plan.json')
    for key,value in plan['pinned_inputs'].items():
        folder,name=key.split('/',1)
        path=getattr(args,folder)/name
        assert digest(path)==value,path
    shared=load(args.reference/'run.py','forecast_prior_design')
    archive=read(args.reference/'plan.json')
    files=shared.audited_files(args,archive)
    cap={(r['dataset'],int(r['allocation']),int(r['label_people'])):float(r['safe_cap'])
         for r in table(args.certificate/'safe_results/records.csv')}
    states={}
    for row in table(args.reference/'results/allocations.csv'):
        key=(row['dataset'],int(row['allocation']),int(row['label_people']))
        group=states.setdefault(key,dict(ordinary=row['labelled_only'],rows={}))
        assert group['ordinary']==row['labelled_only']
        draw,power=int(row['draw']),float(row['power'])
        group['rows'][(draw,power)] = (row['status'],int(row['label_tasks']),int(row['aux_tasks']))
    assert len(cap)==len(states)==96 and set(states)==set(cap)
    return plan,archive,cap,states,[food(args,files,archive),swiss(args,files,archive)]


def sets(panel,archive,index,budget):
    order=panel.allocation_orders[index]
    label=np.isin(panel.ids,order[:budget])
    assert not (label & (panel.train | panel.test)).any()
    rng=np.random.default_rng(archive['auxiliary_generator_seed']+10007*index+101*budget)
    draws=[np.isin(panel.ids,rng.choice(order[budget:],size=archive['auxiliary_people'],replace=False))
           for j in range(archive['auxiliary_draws'])]
    assert not any(np.any(u&(panel.train | panel.test | label)) for u in draws)
    return label,draws


def forecast_prob(x, available, beta):
    utilities=np.where(available,x@beta,-np.inf)
    return softmax(utilities,axis=1)


def mean_feature(x,p):
    return np.einsum('na,nap->p',p,x)/len(x)


def score_metric(panel,prob,log_prob):
    y=panel.y[panel.test]
    _,person=np.unique(panel.ids[panel.test],return_inverse=True)
    count=np.bincount(person)
    errors=-log_prob[np.arange(len(y)),y]
    brier=np.sum((prob-np.eye(prob.shape[1])[y])**2,axis=1)
    average=lambda v:float(np.mean(np.bincount(person,weights=v)/count))
    return average(errors),average(brier)
