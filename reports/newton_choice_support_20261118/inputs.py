"""Recover declared datasets, archived pilot coefficients and support outcomes."""
from dataclasses import dataclass
from hashlib import sha256
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax


@dataclass
class Case:
    dataset: str
    specification: str
    allocation: int
    labelled_people: int
    exact_state: str
    features: np.ndarray
    available: np.ndarray
    moment: np.ndarray
    pilot: np.ndarray
    auxiliary_proxy: np.ndarray


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def checked_manifest(directory, names):
    manifest=dict(line.split('  ',1)[::-1] for line in (directory/'MANIFEST.sha256').read_text().splitlines())
    for n in names:
        assert sha(directory/n)==manifest[n],n
    return {n:manifest[n] for n in names}


def load_module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def proxy_moment(design, probs):
    return np.einsum('nj,njp->p',probs,design)/len(design)


def unit_target(x_l,x_u,y_l,g_l,g_u):
    return x_l[np.arange(len(y_l)),y_l].mean(axis=0)-proxy_moment(x_l,g_l)+proxy_moment(x_u,g_u)


def food(data_path, evidence):
    model=load_module(evidence/'analysis/existing_panel_model.py','food_existing_model')
    plan=read(evidence/'inputs/model_plan.json')
    data=model.load(data_path,plan)
    teacher=np.asarray(read(evidence/'inputs/preflight.json')['teacher_coefficients'])
    proxy=softmax(data['teacher'] @ teacher,axis=2)
    original=rows(evidence/'inputs/baseline_records.csv')
    support={(int(r['allocation']),int(r['labelled_people'])):r['support']
             for r in original if r['method']=='corrected_unit'}
    pilots={z['record']:np.asarray(z['beta']) for z in read(evidence/'inputs/labelled_ridge_coefficients.json')}
    pilot_ids={(int(r['allocation']),int(r['labelled_people'])):int(r['record'])
               for r in original if r['method']=='labelled_ridge'}
    assert len(support)==len(pilots)==len(pilot_ids)==2000
    rng=np.random.default_rng(plan['allocation_seed'])
    for trial in range(500):
        order=rng.permutation(data['pool'])
        for budget in plan['labelled_respondent_budgets']:
            labelled,aux=order[:budget],order[budget:]
            f_l=data['f'][labelled].reshape(-1,3,11)
            f_u=data['f'][aux].reshape(-1,3,11)
            g_l,g_u=proxy[labelled].reshape(-1,3),proxy[aux].reshape(-1,3)
            moment=unit_target(f_l,f_u,data['y'][labelled].ravel(),g_l,g_u)
            key=(trial,budget)
            yield Case('RADIANT','food_11',trial,budget,support[key],f_u,
                       np.ones(f_u.shape[:2],bool),moment,pilots[pilot_ids[key]],g_u)


def swiss(data_path,evidence):
    model=load_module(evidence/'audit.py','swiss_existing_model')
    plan=read(evidence/'plan.json')
    assert sha(data_path)==plan['data_sha256']
    ids,y,available,features=model.load_data(data_path)
    shuffled=np.random.default_rng(plan['split_seed']).permutation(np.unique(ids))
    teacher_end=plan['teacher_count']+plan['evaluation_count']
    pool=shuffled[teacher_end:]
    mask_pool=np.isin(ids,pool)
    assert len(pool)==351
    ids,y,available=ids[mask_pool],y[mask_pool],available[mask_pool]
    centers={name:model.center_features(value[mask_pool],available) for name,value in features.items()}
    frozen=np.asarray(read(evidence/'results/summary.json')['input_summary']['teacher_coefs'])
    predicted=model.probability(centers['canonical_4'],available,frozen)
    pilots={z['record']:np.asarray(z['beta']) for z in read(evidence/'results/coefficients.json') if z['method']=='labelled_ridge'}
    orders=[np.random.default_rng(plan['allocation_seed']+i).permutation(pool) for i in range(plan['allocations'])]
    stored=rows(evidence/'results/allocations.csv')
    assert len(stored)==len(pilots)==576
    for r in stored:
        name=r['specification']
        trial,budget,rec=(int(r[n]) for n in ('allocation','labelled_people','record'))
        take=np.isin(ids,orders[trial][:budget])
        assert take.sum()==int(r['labelled_tasks'])
        f=centers[name]
        cap=float(r['cap'])
        if r['endpoint']=='unit_interior':
            assert r['unit_support']=='interior' and cap>1-1e-8
            state='interior'
        else:
            assert cap<1-1e-8
            state='outside'
        moment=unit_target(f[take],f[~take],y[take],predicted[take],predicted[~take])
        yield Case('Swissmetro',name,trial,budget,state,f[~take],available[~take],moment,pilots[rec],predicted[~take])
