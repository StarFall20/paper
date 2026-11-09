"""Exact Gaussian finite-root probabilities with an informative binary proxy.

The predictor is the frozen conditional-mean oracle for a symmetric proxy.
Classical lift-zonoid geometry is an attributed ingredient, not a new result.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from scipy.special import gammainc
from scipy.stats import binom, norm, binomtest
from public_conjoint_optimizer_audit import save
from rectified_choice_existence_checks import exact_finite, radius


def mixture(m,epsilon):
    thresholds=[];weights=[]
    for k in range(m+1):
        a=np.arange(m-k+1)[:,None];b=np.arange(k+1)[None,:]
        t=.5+(epsilon*(2*a-(m-k))+(1-epsilon)*(2*b-k))/m
        variance=(m-k)*epsilon**2+k*(1-epsilon)**2
        w=binom.pmf(k,m,epsilon)*binom.pmf(a,m-k,.5)*binom.pmf(b,k,.5)
        threshold=m*m*radius(t)**2/variance
        threshold[(t<=0)|(t>=1)]=0.
        thresholds.append(threshold.reshape(-1));weights.append(w.reshape(-1))
    threshold=np.concatenate(thresholds);weight=np.concatenate(weights)
    assert abs(weight.sum()-1)<1e-12
    return threshold,weight


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--replications',type=int,default=2000);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261110);rows=[];reduction=0.
    ratios=[.1,.25,.5,.75,1.,1.5,2.,3.,4.];epsilons=[.05,.1,.25,.4,.5]
    for m in [20,60,120]:
        dimensions=sorted({max(1,int(round(r*m))) for r in ratios});counts={(e,p):0 for e in epsilons for p in dimensions}
        for start in range(0,a.replications,50):
            size=min(50,a.replications-start);x=rng.normal(size=(size,m,max(dimensions)))
            y=rng.integers(0,2,size=(size,m));u=rng.random((size,m))
            for epsilon in epsilons:
                z=np.logical_xor(y,u<epsilon).astype(float);g=epsilon+(1-2*epsilon)*z
                residual=y-g;t=.5+residual.mean(1)
                slope=(x*residual[:,:,None]).mean(1);norms=np.cumsum(slope*slope,axis=1)
                bound=radius(t)**2;valid=(t>0)&(t<1)
                for d in dimensions:counts[(epsilon,d)]+=int(np.sum(valid&(norms[:,d-1]<bound)))
        for epsilon in epsilons:
            threshold,weights=mixture(m,epsilon)
            for d in dimensions:
                truth=float(weights@gammainc(d/2,threshold/2));count=counts[(epsilon,d)]
                if epsilon==.5:reduction=max(reduction,abs(truth-exact_finite(m,d,.5)))
                rows.append({'labels':m,'dimensions':d,'dimension_label_ratio':d/m,'proxy_flip_probability':epsilon,
                    'replications':a.replications,'finite_count':count,'finite_frequency':count/a.replications,
                    'exact_finite_probability':truth,'Brier_risk':epsilon*(1-epsilon),
                    'critical_ratio':1/(2*np.pi*epsilon*(1-epsilon)),
                    'population_optimal_weight_infinite_auxiliary':1. if epsilon<.5 else None,
                    'absolute_error':abs(count/a.replications-truth),
                    'exact_binomial_p':binomtest(count,a.replications,truth).pvalue})
        print('informative proxy complete for labels',m,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(a.out/'informative_surrogate_exact_checks.csv',index=False)
    assert reduction<1e-12
    save(a.out/'informative_surrogate_numeric_checks.json',{'cells':len(rows),'replications_per_cell':a.replications,
        'direct_X_Y_Z_draws':True,'uninformative_proxy_reduction_max_error':reduction,
        'maximum_probability_absolute_error':frame.absolute_error.max(),
        'minimum_exact_binomial_p':frame.exact_binomial_p.min(),
        'minimum_Bonferroni_p':min(1.,len(frame)*frame.exact_binomial_p.min()),
        'independent_priority_verified':False,'Gaussian_binary_scope':True})
    print('Informative-surrogate exact probability checks completed.',flush=True)


if __name__=='__main__':main()
