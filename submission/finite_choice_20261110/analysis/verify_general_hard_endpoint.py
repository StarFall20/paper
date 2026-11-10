"""Post-plan analytic scope check: a hard anti-label endpoint has no strict gain.

This check was added after the frozen interior-predictor simulation plan. It
does not alter its draws or source and is not an additional phase experiment.
"""
from pathlib import Path
import argparse, json
import numpy as np
from finite_ratio_signal_boundary import Population, B_terms, tail_derivative
from finite_ratio_choice_boundary import tail
from general_joint_power_boundary import ordinary_boundary
from public_conjoint_optimizer_audit import save


def verify(out):
    error=0.;excess=0.;positive_derivative=0.;count=0
    for gamma in [0.,1.5,2.]:
      for tau in [1.,4.]:
        pop=Population(gamma,.25,tau,1.,400)
        for eta in [np.zeros(2),np.array([.2,-.3]),np.array([-.4,.7])]:
            # Z=1-Y,g=Z implies r=1,s_U=s_L,J_U=[-1/tau,0].
            h=pop.signL*(eta[0]+eta[1]*pop.s)
            at0=float(pop.weights@tail(h))
            for v in [0.,.01,.2,1.,3.]:
                aux,_,dv=B_terms(h,v,np.full_like(h,-1/tau),np.zeros_like(h))
                value=float(pop.weights@(1+h*h-tail(-h-v)+tau*aux))
                direct=tail_derivative(-h-v)-tail_derivative(-h+v/tau)
                actual=tail_derivative(-h-v)+tau*dv
                error=max(error,float(max(abs(actual-direct))))
                excess=max(excess,value-at0)
                positive_derivative=max(positive_derivative,float(max(actual)))
                if v==0:assert max(abs(actual))<1e-12 and abs(value-at0)<1e-12
                count+=1
    assert error<1e-11 and excess<1e-11 and positive_derivative<1e-11
    result={'post_plan_analytic_scope_check':True,'population_scale_evaluations':count,
            'hard_predictor':'g=Z=1-Y','label_residual_r':1,
            'scale_derivative_identity_max_error':error,'positive_derivative_tolerance_excess':positive_derivative,
            'positive_scale_value_over_zero_tolerance_excess':excess,
            'ordinary_signal1point5_boundary':ordinary_boundary(1.5,400),
            'analytic_conclusion':'Derivative equals E[f_prime(-h-v)-f_prime(-h+v/tau)]<=0;the supremum is at zero and the variational formula equals the ordinary boundary. Strict-interior uniform strict-gain claim cannot include this hard endpoint.',
            'interior_phase_sources_or_draws_changed':False,'priority_proven':False}
    save(out/'general_joint_power_hard_endpoint_checks.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);verify(a.out)
