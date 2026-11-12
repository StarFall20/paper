"""Established fixed-dimensional power tuning for hard-category menu scores.

The control-variate criterion is PPI++ prior work. This entry point calculates
its actual oracle power for the specified proxy and evaluates the candidate
existence boundary there; it does not assume the posterior-oracle power.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from menu_hard_prediction_boundary import HardMenuPopulation
from menu_corrected_logit import save


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    plan=json.loads(a.plan.read_text());rows=[]
    for seed in plan['seeds']:
      for J in [2,3]:
       for gamma in [0.,1.5]:
        for tau in [1.,4.]:
         for epsilon in [.25,1.]:
            model=HardMenuPopulation(J,gamma,epsilon,tau,1.,14,seed)
            mu=model.mu;R=float(np.mean(1-np.sum(mu*mu,axis=1)))
            slope=1-epsilon*J/(J-1);K=slope*R;V=2*epsilon+(2*slope-1)*R
            human=mu[:,None,:]-np.eye(J)[None,:,:]
            surrogate=mu[:,None,:]-model.g
            direct_K=float(np.mean(np.einsum('nzy,nyk,nzk->n',model.joint,human,surrogate)))
            direct_V=float(np.mean(np.sum(model.pZ*np.sum(surrogate**2,axis=-1),axis=1)))
            assert max(abs(K-direct_K),abs(V-direct_V))<1e-10
            unconstrained=tau*K/((1+tau)*V);power=float(np.clip(unconstrained,0.,1.))
            model.power=power
            result=model.critical(ordinary=True) if power==0 else model.critical()
            variance=lambda lam:(R-2*lam*K+(1+1/tau)*lam*lam*V)/(R*R)
            assert variance(power)<=variance(0.)+1e-10
            rows.append(dict(alternatives=J,gamma=gamma,tau=tau,epsilon=epsilon,seed=seed,bits=14,
                human_score_variance=R,score_cross_covariance=K,surrogate_score_variance=V,
                unconstrained_power=unconstrained,clipped_variance_power=power,
                covariance_formula_error=abs(K-direct_K),surrogate_variance_formula_error=abs(V-direct_V),
                ordinary_asymptotic_variance=variance(0.),optimal_asymptotic_variance=variance(power),
                half_power_asymptotic_variance=variance(.5),unit_power_asymptotic_variance=variance(1.),
                critical_ratio=result['critical_ratio'],signal_multiplier=result['signal_multiplier'],scale=result['scale']))
            pd.DataFrame(rows).to_csv(a.out/'hard_menu_variance_reference.csv',index=False)
            print(json.dumps({k:rows[-1][k] for k in ['alternatives','gamma','tau','epsilon','seed','clipped_variance_power','critical_ratio']}),flush=True)
    save(a.out/'hard_menu_variance_reference_checks.json',dict(state='complete',conditions=len(rows),
        maximum_covariance_formula_error=max(r['covariance_formula_error'] for r in rows),
        maximum_variance_formula_error=max(r['surrogate_variance_formula_error'] for r in rows),
        negative_covariance_zero_power_cases=sum(r['score_cross_covariance']<0 and r['clipped_variance_power']==0 for r in rows),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),
        scope='Existing fixed-dimensional oracle variance criterion on orthogonal common slopes; not a new tuning rule or high-dimensional risk guarantee'))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
