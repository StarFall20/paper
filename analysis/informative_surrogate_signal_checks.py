"""Nonnull Gaussian checks with an oracle for a genuinely informative proxy."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import norm
from public_conjoint_optimizer_audit import save
from rectified_choice_existence_checks import moments,radius


def proxy_risk(gamma,epsilon):
    shift=np.log((1-epsilon)/epsilon)
    def integrand(t):
        p=expit(gamma*t);z1=epsilon+(1-2*epsilon)*p
        g1=expit(gamma*t+shift);g0=expit(gamma*t-shift)
        return norm.pdf(t)*(z1*g1*(1-g1)+(1-z1)*g0*(1-g0))
    return quad(integrand,-12,12,epsabs=1e-12)[0]


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    moments_by_case={};posterior_error=0.
    for gamma in [1.,3.]:
        slope,_,_=moments(gamma)
        for epsilon in [.1,.25,.4]:
            risk=proxy_risk(gamma,epsilon)
            moments_by_case[(gamma,epsilon)]=(slope,risk)
            grid=np.linspace(-6,6,31);p1=expit(gamma*grid);shift=np.log((1-epsilon)/epsilon)
            bayes1=p1*(1-epsilon)/(epsilon+(1-2*epsilon)*p1)
            bayes0=p1*epsilon/(1-epsilon-(1-2*epsilon)*p1)
            posterior_error=max(posterior_error,float(abs(expit(gamma*grid+shift)-bayes1).max()),float(abs(expit(gamma*grid-shift)-bayes0).max()))
    assert posterior_error<1e-8
    rng=np.random.default_rng(20261111);rows=[]
    for m in [20,60,120]:
        dimensions=sorted({max(1,int(round(r*m))) for r in [.1,.25,.5,.75,1.,1.5,2.,3.,4.]})
        counts={(gamma,e,d):0 for gamma,e in moments_by_case for d in dimensions}
        for start in range(0,2000,50):
            size=min(50,2000-start);x=rng.normal(size=(size,m,max(dimensions)));u=rng.random((size,m));flip=rng.random((size,m))
            for (gamma,e),(slope,risk) in moments_by_case.items():
                y=(u<expit(gamma*x[:,:,0])).astype(float);z=np.logical_xor(y,flip<e).astype(float)
                g=expit(gamma*x[:,:,0]+(2*z-1)*np.log((1-e)/e));residual=y-g
                t=.5+residual.mean(1);s=(x*residual[:,:,None]).mean(1);s[:,0]+=slope
                norm2=np.cumsum(s*s,axis=1);bound=radius(t)**2;valid=(t>0)&(t<1)
                for d in dimensions:counts[(gamma,e,d)]+=int(np.sum(valid&(norm2[:,d-1]<bound)))
        for (gamma,e,d),count in counts.items():
            slope,risk=moments_by_case[(gamma,e)]
            rows.append({'labels':m,'dimensions':d,'dimension_label_ratio':d/m,'signal':gamma,'proxy_flip_probability':e,
                         'replications':2000,'finite_count':count,'finite_frequency':count/2000,'oracle_Brier_risk':risk,
                         'critical_ratio':(norm.pdf(0)**2-slope*slope)/risk})
        print('nonnull informative-proxy checks complete for labels',m,flush=True)
    pd.DataFrame(rows).to_csv(a.out/'informative_surrogate_signal_frequency.csv',index=False)
    save(a.out/'informative_surrogate_signal_numeric_checks.json',{'cells':len(rows),'replications_per_cell':2000,
        'maximum_posterior_Bayes_identity_error':posterior_error,'population_risks':[
            {'signal':gamma,'flip_probability':e,'Brier_risk':risk,'critical_ratio':(norm.pdf(0)**2-s*s)/risk}
            for (gamma,e),(s,risk) in moments_by_case.items()],
        'exact_finite_signal_formula_claimed':False,'independent_novelty_verified':False})
    print('Nonnull proxy checks complete.',flush=True)


if __name__=='__main__':main()
