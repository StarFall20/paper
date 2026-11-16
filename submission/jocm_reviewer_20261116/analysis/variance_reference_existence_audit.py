"""Audit an established variance reference against the corrected-logit boundary.

Prediction strength shrinks a frozen conditional-mean proxy toward mu(S).
The estimator, fixed-dimensional weight and existence theorem are unchanged.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from imperfect_predictor_choice_boundary import ImperfectPopulation
from finite_ratio_signal_boundary import Population
from finite_ratio_choice_boundary import tail


class ShrunkPopulation(ImperfectPopulation):
    def __init__(self,gamma,epsilon,tau,lam,strength,nodes):
        Population.__init__(self,gamma,epsilon,tau,lam,nodes)
        mu=expit(gamma*self.s)
        self.oracle=self.g.copy()
        self.g=mu+strength*(self.oracle-mu)
        r=np.where(self.y==0,self.g,1-self.g)
        self.aL=lam*r;self.bL=1-lam+lam*r
        self.aU=lam/tau*np.where(self.z==1,-self.g,self.g-1)
        self.bU=lam/tau*np.where(self.z==1,1-self.g,self.g)


def ordinary_boundary(gamma,nodes):
    # Ordinary score geometry is E[(H+(1-2Y)*eta*S)_+^2].
    pop=Population(gamma,.25,1.,1.,nodes)
    fit=minimize_scalar(lambda b:float(pop.weights@tail((1-2*pop.y)*b*pop.s)),bounds=(-12.,12.),method='bounded',options={'xatol':1e-12})
    assert abs(fit.x)<11.9
    return float(fit.fun)


def scalar_null_unit(tau,epsilon,strength):
    confidence=strength*(.5-epsilon)
    risk=.25+confidence*confidence-confidence*(1-2*epsilon)
    def f(u):return tau*risk*u*u+tail((.5+confidence)*u)+tail((.5-confidence)*u)
    fit=minimize_scalar(f,bounds=(0.,40.),method='bounded',options={'xatol':1e-12})
    return float(tau*(1-fit.fun))


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());a.out.mkdir(parents=True,exist_ok=True)
    rows=[];cov_error=0.;quad_error=0.;scalar_error=0.
    for gamma in plan['gamma']:
      for epsilon in plan['epsilon']:
       for tau in plan['tau']:
        for strength in plan['prediction_strength']:
            lam=min(1.,tau/((1+tau)*strength));values=[]
            for nodes in plan['quadrature_nodes']:
                pop=ShrunkPopulation(gamma,epsilon,tau,lam,strength,nodes)
                mu=expit(gamma*pop.s);human=mu-pop.y;proxy=mu-pop.g
                K=float(pop.weights@((mu-pop.oracle)**2));R=float(pop.weights@(human*human))
                cross=float(pop.weights@(human*proxy));V=float(pop.weights@(proxy*proxy))
                cov_error=max(cov_error,abs(cross-strength*K),abs(V-strength*strength*K))
                assert K>0 and abs(lam-float(np.clip(tau/(1+tau)*cross/V,0,1)))<1e-10
                result=pop.critical_general();ordinary=ordinary_boundary(gamma,nodes)
                variance=R-2*lam*cross+(1+1/tau)*lam*lam*V
                assert variance<R
                value=dict(gamma=gamma,epsilon=epsilon,tau=tau,prediction_strength=strength,nodes=nodes,power=lam,
                    critical_ratio=result['critical_ratio'],ordinary_critical_ratio=ordinary,
                    difference_from_ordinary=result['critical_ratio']-ordinary,
                    human_score_variance=R,reference_score_variance=variance,
                    relative_reference_variance=variance/R,oracle_residual_score_variance=K,
                    maximum_multiplier_gradient=result['gradient_max'],intercept_multiplier=result['intercept_multiplier'],
                    signal_multiplier=result['signal_multiplier'],scale=result['scale'])
                values.append(value);rows.append(value)
                if gamma==0 and lam==1:scalar_error=max(scalar_error,abs(result['critical_ratio']-scalar_null_unit(tau,epsilon,strength)))
            quad_error=max(quad_error,abs(values[0]['critical_ratio']-values[-1]['critical_ratio']))
            pd.DataFrame(rows).to_csv(a.out/'variance_reference_population.csv',index=False)
    assert cov_error<1e-10 and quad_error<1e-7 and scalar_error<1e-7,(cov_error,quad_error,scalar_error)
    frame=pd.DataFrame(rows)
    fine=frame[frame.nodes==max(plan['quadrature_nodes'])]
    fine.to_csv(a.out/'variance_reference_population_primary.csv',index=False)
    result=dict(state='complete',population_evaluations=len(rows),conditions=len(fine),
        below_ordinary_conditions=int((fine.difference_from_ordinary< -1e-7).sum()),
        covariance_identity_maximum_error=cov_error,quadrature_maximum_difference=quad_error,
        independent_null_scalar_maximum_error=scalar_error,
        every_fixed_dimensional_reference_variance_below_human=bool((fine.relative_reference_variance<1).all()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),
        scope=plan['claim_limit'])
    (a.out/'variance_reference_population_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
