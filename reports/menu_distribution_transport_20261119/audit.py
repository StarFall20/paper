"""Compare composition and conditional prediction drift in released choice fits."""
import argparse
from collections import defaultdict
from itertools import groupby
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csc_matrix, hstack
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
KEY = ('context', 'rep', 'coverage', 'label_people', 'label_fold', 'eval_fold')


def menu_groups(x, g):
    groups = defaultdict(list)
    for menu, prob in zip(x, g, strict=True):
        order = np.lexsort(menu.T[::-1])
        sorted_menu = menu[order]
        aligned = sorted_menu-sorted_menu[0]
        groups[tuple(aligned.ravel())].append(prob[order])
    return groups


def compare_groups(xl, gl, xu, gu):
    left, right = menu_groups(xl,gl), menu_groups(xu,gu)
    rows = []
    for code in sorted(left.keys() | right.keys()):
        l, u = left.get(code), right.get(code)
        nl, nu = len(l or ()), len(u or ())
        x = np.array(code).reshape(xl.shape[1:])
        lmean = np.mean(l if l else u,axis=0)
        umean = np.mean(u if u else l,axis=0)
        rows.append((x, nl, nu, x.T@lmean, x.T@umean))
    x,nl,nu,ml,mu = (np.asarray(block) for block in zip(*rows,strict=True))
    wl,wu = nl/len(xl),nu/len(xu)
    common = (nl>0)&(nu>0)
    assert np.all((ml==mu).all(axis=1) | common)
    return x,wl,wu,ml,mu,common


def fit_inputs(x, weights, target):
    keep = weights>0
    assert abs(weights.sum()-1)<1e-12
    return x[keep],np.ones(x[keep].shape[:2],bool),weights[keep],target,0


def separation_direction(fit, tolerance):
    x,_,w,b,_ = fit
    n,k,p = x.shape
    physical = csc_matrix(x.reshape(-1,p))
    ceiling = coo_matrix((-np.ones(n*k),(np.arange(n*k),np.arange(n).repeat(k))),shape=(n*k,n)).tocsc()
    sol = linprog(np.r_[-b,w],A_ub=hstack((physical,ceiling)),b_ub=np.zeros(n*k),
                  bounds=[(-1,1)]*p+[(None,None)]*n,method='highs-ds')
    assert sol.success and sol.fun < -tolerance
    d=sol.x[:p]
    gap=w@np.max(x@d,axis=1)-b@d
    assert abs(gap-sol.fun)<tolerance and gap < -tolerance
    return d,gap


def fit_method(fit, backend, base, case, beta0, fallback, tolerance):
    state = base.support(backend,fit)
    record = dict(state=state,beta=None,loss=fallback)
    if state=='interior':
        beta,error,ok=backend.forecast.fit_model(fit,beta0,tolerance)
        assert ok, (case.key,error)
        record.update(beta=beta.tolist(),score=error,loss=base.metric(case,beta)[0])
    return record


def record(case, reference, base, backend, plan):
    xl,y,gl,_=case.label
    xu,gu,_=case.aux
    xr,wl,wu,ml,mu,common=compare_groups(xl,gl,xu,gu)
    lcenter=np.asarray([a[np.lexsort(a.T[::-1])[0]] for a in xl])
    ucenter=np.asarray([a[np.lexsort(a.T[::-1])[0]] for a in xu])
    li,ui=xl-lcenter[:,None],xu-ucenter[:,None]
    b=base.moments(li,y)
    delta=wu@mu-wl@ml
    common_g=(ml+mu)/2
    mix=(wu-wl)@common_g
    drift=((wu+wl)/2)@(mu-ml)
    assert np.max(abs(delta-mix-drift))<1e-12
    assert np.max(abs(delta-(base.moments(ui,gu)-base.moments(li,gl))))<1e-12
    original=fit_inputs(xr,wu,b+delta)
    strata=fit_inputs(xr,wl,b+wl@(mu-ml))
    composition=fit_inputs(xr,wu,b+mix)
    tl=plan['recession_tol']
    u_status=base.support(backend,original)
    unit=reference['fixed_unit']
    assert u_status==('interior' if unit['state']=='native' else unit['reason']),case.key
    ridge=reference['labelled_ridge']
    start=np.array(ridge['beta'])
    output=dict(zip(KEY,case.key,strict=True)) | dict(
        labelled_coverage=float(wl@common),auxiliary_coverage=float(wu@common),
        common_menus=int(common.sum()),labelled_menus=int((wl>0).sum()),auxiliary_menus=int((wu>0).sum()),
        unit=dict(state=u_status,loss=unit['loss']),ridge=ridge['loss'],
        ordinary=reference['labelled_MLE']['loss'],matched_prompt_MLE=reference['prompt_matched_MLE']['loss'])
    output['poststratified']=fit_method(strata,backend,base,case,start,ridge['loss'],plan['score_tol'])
    output['composition_only']=fit_method(composition,backend,base,case,start,ridge['loss'],plan['score_tol'])
    if not common.any():
        assert output['poststratified']['state']==('interior' if output['ordinary'] is not None else reference['labelled_MLE']['reason'])
        ordinary_loss=ridge['loss'] if output['ordinary'] is None else output['ordinary']
        assert abs(output['poststratified']['loss']-ordinary_loss)<1e-7
        assert output['composition_only']['state']==u_status
        assert abs(output['composition_only']['loss']-unit['loss'])<1e-7
    if u_status=='outside':
        d,gap=separation_direction(original,tl)
        h=np.max(xr@d,axis=1)
        human=wl@h-b@d
        composition_gap=(wu-wl)@h-mix@d
        drift_gap=-drift@d
        assert human>=-tl and abs(gap-human-composition_gap-drift_gap)<tl
        rawgap=np.max(ui@d,axis=1).mean()-d@(b+base.moments(ui,gu)-base.moments(li,gl))
        transport_gap=human-(wl@(mu-ml))@d
        assert abs(rawgap-gap)<tl
        output['recession']=dict(direction=d.tolist(),observed=float(gap),human=float(human),
                                 composition=float(composition_gap),within=float(drift_gap),
                                 transported=float(transport_gap))
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('upstream','dependencies','inputs','reference','out'):
        parser.add_argument('--'+k,required=True,type=Path)
    args=parser.parse_args()
    sys.path.insert(0,str(args.upstream.resolve()))
    from source_data import cases, dependencies, read, sha
    import run as base
    plan=read(ROOT/'plan.json')
    assert sha(args.reference)==plan['reference_fits_sha256']
    old_plan,backend=dependencies(args.dependencies)
    original={k:{r['method']:r for r in v} for k,v in groupby(read(args.reference),key=lambda r:tuple(r[n] for n in KEY))}
    with threadpool_limits(limits=1):
        result=[record(c,original[c.key],base,backend,plan) for c in cases(args,old_plan,backend)]
    assert len(result)==480 and len({tuple(r[k] for k in KEY) for r in result})==len(original)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Verified original and replayed two menu interventions on all 480 cases')


if __name__=='__main__':
    main()
