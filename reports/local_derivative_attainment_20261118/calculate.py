"""Construct rational matched choice designs and compute approximation witnesses."""
import argparse
from fractions import Fraction as F
from math import prod
from pathlib import Path
import hashlib
import json

import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def pair(magnitudes):
    squares = [x*x for x in magnitudes]
    signed = [1/prod(t-u for j,u in enumerate(squares) if j!=i) for i,t in enumerate(squares)]
    positives = sum(w for w in signed if w>0)
    negative = -sum(w for w in signed if w<0)
    assert positives==negative and positives>0
    a = [(t,w/positives) for t,w in zip(magnitudes,signed) if w>0]
    b = [(t,-w/negative) for t,w in zip(magnitudes,signed) if w<0]
    mean_a,mean_b = (sum(x*w for x,w in d) for d in (a,b))
    assert mean_a!=mean_b
    return (a,b) if mean_a>mean_b else (b,a)


def design(k, magnitudes):
    labelled,auxiliary = pair(magnitudes)
    m_l,m_u = (sum(x*w for x,w in group) for group in (labelled,auxiliary))
    gap = m_l-m_u
    cutoff = F(1,2)
    tilt = (m_l-cutoff*gap)/m_l
    assert 0<tilt<1
    return dict(K=k,matched_total_degree=2*k+1,
                nodes=[str(x) for x in magnitudes],
                labelled_magnitude_mass=[[str(x),str(w)] for x,w in labelled],
                auxiliary_magnitude_mass=[[str(x),str(w)] for x,w in auxiliary],
                absolute_moment_gap=str(gap),labelled_outcome_tilt=str(tilt),
                slope_moment=str(tilt*m_l/2),critical_power=str(cutoff),
                prediction_shrink_product='1/16',
                exact_slope_recession={str(p):str((cutoff-p)*gap/2) for p in (F(1,4),cutoff,F(3,4))})


def grid_witness(k, grid_n):
    x=np.linspace(0,1,grid_n+1)
    basis=chebvander(x,2*k)[:,::2]
    condition=np.vstack((np.column_stack((basis,-np.ones(len(x)))),
                         np.column_stack((-basis,-np.ones(len(x))))))
    result=linprog(np.r_[np.zeros(k+1),1.], A_ub=condition,b_ub=np.r_[x,-x],
                   bounds=[(None,None)]*(k+1)+[(0,None)],method='highs',
                   options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    assert result.success,result.message
    dual=result.ineqlin.marginals
    support=sorted(set(map(int,np.flatnonzero(abs(dual)>1e-8) % (grid_n+1))))
    assert len(support)==k+2,(k,support)
    case=design(k,[F(index,grid_n) for index in support])
    residual=abs(float(F(case['absolute_moment_gap']))/2-result.fun)
    assert residual<1e-8
    case.update(grid_n=grid_n, support_indices=support,
                approx_even_chebyshev_coefficients=[format(z,'.17g') for z in result.x[:-1]])
    return case


def main(out):
    plan=ROOT/'plan.json'
    spec=json.loads(plan.read_text())
    exact=[design(k,[F(i+1,k+2) for i in range(k+2)]) for k in spec['exact_orders_K']]
    near=[grid_witness(k,spec['uniform_grid_intervals']) for k in spec['approximation_orders_K']]
    out.mkdir(parents=True)
    write(out/'designs.json',dict(status='generated',plan_sha256=sha(plan),script_sha256=sha(Path(__file__)),
                                  exact_constructions=exact,near_extremal_grid_constructions=near))
    print(json.dumps(dict(exact_cases=len(exact),approximation_cases=len(near))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    with threadpool_limits(limits=1):
        main(parser.parse_args().out)
