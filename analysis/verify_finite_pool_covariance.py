"""Closed-form intercept benchmarks for the implemented covariance formulas."""
from pathlib import Path
import argparse,numpy as np
from scipy.optimize import brentq
from scipy.special import expit,logit
from finite_pool_choice_validation import covariance,interior_fit
from power_tuned_choice_audit import WeightedLogit
from public_conjoint_optimizer_audit import save


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    m=10;N=4;x=np.ones((m+N,1));y=np.r_[np.array([1.,1.,1.,0.,0.,0.,0.,0.,0.,0.]),np.zeros(N)]
    g=np.r_[np.linspace(.15,.75,m),[.2,.4,.6,.8]];maximum=0.
    for lam in [0.,.3,1.]:
        p=y[:m].mean()+lam*(g[m:].mean()-g[:m].mean());b=np.array([logit(p)])
        actual=covariance(x,y,g,m,b,lam,'PPI_unit',0.)[0,0]
        closed=(np.var(y[:m]-lam*g[:m],ddof=1)/m+lam*lam*np.var(g[m:],ddof=1)/N)/(p*(1-p))**2
        maximum=max(maximum,abs(actual-closed));assert abs(actual-closed)<1e-12
    p=(sum(y[:m])+sum(g[m:]))/(m+N);b=np.array([logit(p)])
    actual=covariance(x,y,g,m,b,None,'soft_oracle_pool',0.)[0,0]
    closed=(m*np.var(y[:m],ddof=1)+N*np.var(g[m:],ddof=1))/(m+N)**2/(p*(1-p))**2
    maximum=max(maximum,abs(actual-closed));assert abs(actual-closed)<1e-12
    exterior=WeightedLogit(np.ones((4,1)),np.full(4,.25),np.array([1.2]))
    b,check=interior_fit(exterior,np.zeros(1),rho=.1)
    independent=brentq(lambda b:expit(b)-1.2+.1*b,0.,20.)
    assert b is not None and abs(b[0]-independent)<1e-7
    save(args.out/'finite_pool_covariance_verification.json',{'intercept_variance_max_closed_form_error':maximum,
        'ridge_exterior_root_difference_from_brent':abs(b[0]-independent),
        'unpenalized_base_has_no_finite_root':True,'penalty_changes_the_objective':True,
        'interval_coverage_claim':'No coverage guarantee follows from these numerical identities'})
    print('Closed-form intercept variance and penalized exterior-root checks passed.')


if __name__=='__main__':main()
