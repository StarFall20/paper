"""Verify recorded exterior moments using a separate separating-ray LP.

This is a classical convex-support calculation. It does not rerun or select
the matched point-estimation experiment, and it does not claim new theory.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from public_conjoint_optimizer_audit import load_data, Logistic, save
from public_conjoint_matched_audit import nuisance


def separating_direction(x, target):
    n, d = x.shape
    moment = x.T @ target / n
    # d = u-v, ||d||_1 <= sum(u+v) <= 1. Epigraph t_i >= x_i d, 0.
    aub = sparse.vstack([
        sparse.hstack([sparse.csr_matrix(x), sparse.csr_matrix(-x), -sparse.eye(n)]),
        sparse.csr_matrix(np.r_[np.ones(2*d), np.zeros(n)][None, :])
    ], format='csr')
    fit = linprog(np.r_[-moment, moment, np.ones(n)/n], A_ub=aub,
                  b_ub=np.r_[np.zeros(n), 1.], bounds=(0., None), method='highs')
    assert fit.success, fit.message
    direction = fit.x[:d] - fit.x[d:2*d]
    gap = float(moment @ direction - np.maximum(x @ direction, 0).mean())
    assert np.abs(direction).sum() <= 1+1e-8
    assert gap > 1e-8, (gap, fit.fun)
    loss = Logistic(x, target)
    # softplus(u) differs from positive_part(u) by at most log(2).
    rays = {str(k): loss.value(k*direction)[0] for k in [0., 100., 10000., 1000000.]}
    for k in [10000., 1000000.]:
        assert -1e-9 <= rays[str(k)] + k*gap <= np.log(2)+1e-8
    return {'separation_gap': gap, 'direction_l1': float(abs(direction).sum()),
            'direction': direction, 'ray_mean_objectives': rays,
            'recession_slope': -gap}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a=p.parse_args()
    cases=json.loads((a.out/'unresolved_and_nonfinite_cases.json').read_text())
    assert len(cases)==7 and all(c['status']=='outside_mean_support' for c in cases)
    x,y,z,_,_=load_data(a.input_dir);x=x[:1200];y=y[:1200]
    rng=np.random.RandomState(20261109); results=[]
    for trial in range(max(c['trial'] for c in cases)+1):
        order=(rng.permutation(240)[:,None]*5+np.arange(5)[None]).reshape(-1)
        for case in [c for c in cases if c['trial']==trial]:
            m=case['label_rows'];xx=x[order];yy=y[order];zz=z[order]
            g,fallback=nuisance(xx,yy,zz,m,case['method']=='corrected_AI')
            target=g+np.r_[np.full(m,1200/m),np.zeros(1200-m)]*(yy-g)
            results.append({**case, **separating_direction(xx,target),
                            'nuisance_fallbacks':fallback})
    save(a.out/'nonexistence_ray_checks.json', {'cases':results,
         'all_exterior_cases_independently_separated':len(results)==len(cases),
         'criterion_scope':'classical finite binary logit convex support',
         'new_theory_claim':False})
    print(json.dumps({'cases':len(results), 'minimum_separation_gap':min(r['separation_gap'] for r in results)}))


if __name__=='__main__': main()
