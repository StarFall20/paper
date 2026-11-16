"""Provisional finite-ratio boundary and independently formulated LP checks."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq,linprog,minimize_scalar
from scipy.special import ndtr
from scipy.stats import norm
from public_conjoint_optimizer_audit import save
from power_tuned_choice_audit import objective,support


def tail(a):return np.maximum(0.,(1+a*a)*ndtr(-a)-a*norm.pdf(a))


def critical_ratio(tau,epsilon,lam):
    if lam==0:return .5,0.
    extra=(1-lam)/lam
    def function(v):
        return ((1-epsilon)*(tail(-epsilon*v)+tail((extra+epsilon)*v))
                +epsilon*(tail(-(1-epsilon)*v)+tail((extra+1-epsilon)*v))
                +tau*(tail((1-epsilon)*v/tau)+tail(epsilon*v/tau)))
    def derivative(v):
        fp=lambda a:2*(a*ndtr(-a)-norm.pdf(a))
        return ((1-epsilon)*(-epsilon*fp(-epsilon*v)+(extra+epsilon)*fp((extra+epsilon)*v))
                +epsilon*(-(1-epsilon)*fp(-(1-epsilon)*v)+(extra+1-epsilon)*fp((extra+1-epsilon)*v))
                +(1-epsilon)*fp((1-epsilon)*v/tau)+epsilon*fp(epsilon*v/tau))
    upper=1.
    while derivative(upper)<=0:upper*=2
    optimum=brentq(derivative,0.,upper,xtol=1e-12)
    value=float(1+tau-function(optimum))
    assert 0<value<1+tau
    return value,optimum


def interval_feasibility(x,y,z,m,epsilon,lam):
    N=len(y)-m;tau=N/m
    signs=np.r_[1-2*y[:m],2*z[m:]-1]
    oriented=x*signs[:,None]
    residual=np.where(y[:m]==z[:m],epsilon,1-epsilon)
    lower=np.r_[lam*residual,np.full(N,-lam/tau*(1-epsilon))]
    upper=np.r_[1-lam+lam*residual,np.full(N,lam/tau*epsilon)]
    fit=linprog(np.zeros(len(y)),A_eq=oriented.T,b_eq=np.zeros(x.shape[1]),bounds=list(zip(lower,upper)),method='highs')
    if fit.status==2:return 'outside'
    if not fit.success:return 'unresolved'
    if max(abs(oriented.T@fit.x))>1e-6:return 'unresolved'
    return 'feasible'


def numeric_checks():
    maximum=0.
    for a in [-2.,0.,.3,2.]:maximum=max(maximum,abs(tail(a)-quad(lambda g:(g-a)**2*norm.pdf(g),a,12,epsabs=1e-12)[0]))
    assert maximum<1e-10
    reduction=0.;limit_errors=[]
    for tau in [1.,4.,16.]:
        for e in [.1,.25]:
            crit,v=critical_ratio(tau,e,1.)
            fn=lambda u:tau*e*(1-e)*u*u+tail((1-e)*u)+tail(e*u)
            separate=minimize_scalar(fn,bounds=(0.,20.),method='bounded',options={'xatol':1e-12})
            reduction=max(reduction,abs(crit-tau*(1-separate.fun)))
    assert reduction<1e-8
    for e in [.1,.25]:
        finite,_=critical_ratio(10000.,e,1.);limit=1/(2*np.pi*e*(1-e))
        assert abs(finite-limit)<.002
        limit_errors.append({'epsilon':e,'finite_ratio':10000,'boundary':finite,'population_limit':limit})
    rng=np.random.default_rng(301);matches=0;types={}
    for j in range(100):
        m=12+j%9;N=m*(1+j%3);d=4+j%22;lam=[.25,.7,1.][j%3];e=[.1,.25][j%2]
        x=rng.normal(size=(m+N,d));y=rng.integers(0,2,m+N).astype(float)
        z=np.logical_xor(y,rng.random(m+N)<e).astype(float);g=e+(1-2*e)*z
        original=support(objective(x,y,g,m,lam))['status'];box=interval_feasibility(x,y,z,m,e,lam)
        expected='feasible' if original=='interior' else 'outside' if original=='outside_mean_support' else 'unresolved'
        assert expected==box,(j,original,box)
        matches+=1;types[original]=types.get(original,0)+1
    return {'normal_tail_quadrature_max_error':maximum,'unit_power_reduction_error':reduction,
            'large_auxiliary_limit_checks':limit_errors,'original_interval_LP_matches':matches,
            'identity_status_counts':types,'independent_priority_verified':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    checks=numeric_checks();save(a.out/'finite_ratio_boundary_numeric_checks.json',checks)
    rng=np.random.default_rng(20261113);rows=[];curves=[]
    for m in [40,80]:
        for tau in [1,4]:
            N=tau*m
            for e in [.1,.25]:
                for name,lam in [('unit',1.),('variance_optimal',tau/(1+tau))]:
                    critical,v=critical_ratio(tau,e,lam)
                    dimensions=sorted({max(1,int(round((critical+offset)*m))) for offset in [-.2,-.1,0.,.1,.2]})
                    counts={d:{'feasible':0,'outside':0,'unresolved':0} for d in dimensions}
                    for rep in range(500):
                        x=rng.normal(size=(m+N,max(dimensions)));y=rng.integers(0,2,m+N).astype(float)
                        z=np.logical_xor(y,rng.random(m+N)<e).astype(float)
                        for d in dimensions:counts[d][interval_feasibility(x[:,:d],y,z,m,e,lam)]+=1
                    for d,count in counts.items():rows.append({'labels':m,'auxiliary_rows':N,'tau':tau,'epsilon':e,'power_name':name,'power':lam,
                         'dimensions':d,'dimension_label_ratio':d/m,'critical_ratio':critical,'replications':500,**count,'finite_frequency':count['feasible']/500})
                    curves.append({'labels':m,'tau':tau,'epsilon':e,'power_name':name,'power':lam,'critical_ratio':critical,'scaled_minimizer':v})
                    pd.DataFrame(rows).to_csv(a.out/'finite_ratio_phase_checks.csv',index=False)
                    pd.DataFrame(curves).to_csv(a.out/'finite_ratio_boundary_values.csv',index=False)
                    save(a.out/'finite_ratio_phase_status.json',{'completed_cells':len(rows),'requested_cells':80,'replications_per_cell':500,'independent_novelty_verified':False})
                    print('finite-ratio phase cells complete',len(rows),'of80',flush=True)
    print('All planned finite-ratio phase cells complete.',flush=True)


if __name__=='__main__':main()
