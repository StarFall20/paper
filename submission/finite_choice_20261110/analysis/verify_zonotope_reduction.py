"""Verify the exact classical Gaussian-zonotope special-case reduction."""
from pathlib import Path
import argparse
import numpy as np
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
from power_tuned_choice_audit import objective,support
from public_conjoint_optimizer_audit import save


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261126);counts={};residual=0.
    for j in range(200):
        m=12+j%9;N=8+j%15;d=min(N,1+j%10)
        x=rng.normal(size=(m+N,d));y=rng.integers(0,2,m+N).astype(float);g=np.full(m+N,.5)
        original=support(objective(x,y,g,m,1.))['status']
        standard_gaussian=x[:m].T@(2*y[:m]-1)/np.sqrt(m)
        target=N/np.sqrt(m)*standard_gaussian
        # Independent signed-cube coefficient LP, with upper/lower slack one.
        aub=np.vstack([np.c_[np.eye(N),np.ones(N)],np.c_[-np.eye(N),np.ones(N)]])
        fit=linprog(np.r_[np.zeros(N),-1.],A_ub=aub,b_ub=np.ones(2*N),
            A_eq=np.c_[x[m:].T,np.zeros(d)],b_eq=target,
            bounds=[(None,None)]*N+[(0,1)],method='highs')
        if fit.status==2:other='outside_mean_support'
        else:
            assert fit.success,(j,fit.message)
            other='interior' if fit.x[-1]>1e-8 else 'boundary_or_near_boundary'
            residual=max(residual,float(max(abs(x[m:].T@fit.x[:-1]-target))))
        assert original==other,(j,original,other)
        counts[other]=counts.get(other,0)+1
    report={'original_corrected_objective_vs_signed_cube_LP_matches':200,'statuses':counts,
        'maximum_absorption_moment_residual':residual,'Gaussian_target_scale':'N/sqrt(m)',
        'prior_absorption_parameter_sigma_squared':'N**2/m',
        'scope':'unit power,constant prediction one half,balanced Gaussian slopes,fixed zero intercept,N>=p',
        'this_special_case_is_classical':True,'full_informative_signal_power_problem_reduced_to_same_prior':False}
    save(args.out/'zonotope_reduction_checks.json',report);print(report)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
