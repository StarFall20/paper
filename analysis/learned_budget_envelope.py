"""Existing all-power envelope and an elementary necessary label-budget bound.

The half-line calculation is the already stated Appendix D4, instantiated for
the learned score. The concavity bound is a consequence, not a new conic law.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.optimize import brentq,root
from scipy.special import expit,roots_legendre,ndtr
from scipy.stats import norm
from scipy.integrate import quad
from threadpoolctl import threadpool_limits
from learned_score_budget_boundary import LearnedPopulation as AttributePopulation
from learned_proxy_budget_boundary import LearnedPopulation as ProxyPopulation
from finite_ratio_signal_boundary import B_terms,tail,tail_derivative
from verify_learned_budget import clipped


def envelope_value(model,eta,v):
    u=model.V@eta;h=model.sign*u;r=np.where(model.y==0,model.g,1-model.g)
    arg=-h-v*r
    left=1+h*h-tail(arg);dl=2*h+tail_derivative(arg);tl=r*tail_derivative(arg)
    su=getattr(model,'signU',np.ones(len(u)))
    right,dr,tr=B_terms(su*u,v,model.ulo,model.uhi)
    return float(model.weights@(left+model.tau*right)),model.V.T@(model.weights*(model.sign*dl+model.tau*su*dr)),float(model.weights@(tl+model.tau*tr))


def calculate(base,out):
    out.mkdir(parents=True,exist_ok=True);plan=json.loads((base/'plan.json').read_text());proxy='proxy_error' in plan
    cls=ProxyPopulation if proxy else AttributePopulation
    class Envelope(cls):
        def value(self,eta,t):return envelope_value(self,eta,t)
        def maximize(self,eta):
            first=self.value(eta,0.)
            if first[2]<=0:return first[0],first[1],0.
            upper=1.
            while self.value(eta,upper)[2]>0:upper*=2
            v=brentq(lambda v:self.value(eta,v)[2],0,upper,xtol=1e-11)
            value,gradient,_=self.value(eta,v)
            return value,gradient,v
    # Independent Legendre formulation of both the half-line and auxiliary set.
    z,w=roots_legendre(100);z=11*z;w=11*w*norm.pdf(z)
    S,T=np.meshgrid(z,z,indexing='ij');S=S.ravel();T=T.ravel();W=np.outer(w,w).ravel();p=expit(plan['utility_signal']*S)
    pairs=[(0,0),(0,1),(1,0),(1,1)] if proxy else [(0,1),(1,1)]
    s=np.tile(S,len(pairs));t=np.tile(T,len(pairs));y=np.repeat([a for a,b in pairs],len(S));zz=np.repeat([b for a,b in pairs],len(S))
    epsilon=plan.get('proxy_error',0.)
    weights=np.concatenate([W*(p if a else 1-p)*((1-epsilon if a==b else epsilon) if proxy else 1.) for a,b in pairs])
    a=quad(lambda x:x*np.tanh(plan['utility_signal']*x/2)*norm.pdf(x),-12,12,epsabs=1e-12)[0]
    rows=[];risks=[];maxdiff=maxquad=0.
    for rho in plan['population_dimension_ratios']:
      for alpha in plan['training_fractions']:
        g=expit(a*s+np.sqrt(rho/alpha)*t+((1-2*epsilon)*(2*zz-1) if proxy else 0.))
        r=np.where(y==0,g,1-g);r1=float(weights@r);r2=float(weights@(r*r))
        bound=.5+(1-r1)**2/(np.pi*r2)
        # Classification integrates T analytically; avoids a thresholded grid.
        def err(x):
            mu=expit(plan['utility_signal']*x)
            total=0.
            for Y,Z in pairs:
                mass=(mu if Y else 1-mu)*((1-epsilon if Y==Z else epsilon) if proxy else 1.)
                prob=ndtr((a*x+((1-2*epsilon)*(2*Z-1) if proxy else 0.))/np.sqrt(rho/alpha))
                total+=mass*(1-prob if Y else prob)
            return norm.pdf(x)*total
        error,errorerr=quad(err,-12,12,epsabs=1e-11,epsrel=1e-11)
        risks.append(dict(total_slope_ratio=rho,training_fraction=alpha,classification_error=error,
            classification_quad_error=errorerr,absolute_prediction_error=r1,Brier_risk=r2,
            zero_multiplier_envelope_upper_bound=bound))
        for nu in plan['population_auxiliary_rows_per_total_label']:
            tau=nu/(1-alpha);model=Envelope(plan['utility_signal'],a,np.sqrt(rho/alpha),tau,1.,80);res=model.critical()
            low=Envelope(plan['utility_signal'],a,np.sqrt(rho/alpha),tau,1.,40).critical();maxquad=max(maxquad,abs(low['critical_ratio']-res['critical_ratio']))
            sl=1-2*y;su=2*zz-1;matrix=np.c_[s,t]
            ulo=np.where(zz==1,-g,g-1)/tau;uhi=np.where(zz==1,1-g,g)/tau
            def direct(e):
                u=matrix@e[:2];v=e[2];h=sl*u;lower=v*r;A=h+lower
                pa=ndtr(A);sf=ndtr(-A);fa=norm.pdf(A)
                left=2*lower*(-fa-h*pa)-lower*lower*pa+(1+h*h)*sf+(A-2*h)*fa
                dl=-2*(lower*pa+fa-h*sf);dv=2*r*(-fa-(h+lower)*pa)
                rr,dr,tr=clipped(su*u,v,ulo,uhi)
                value=float(weights@(left+tau*rr))
                grad=matrix.T@(weights*(sl*dl+tau*su*dr))
                return value,np.r_[grad,weights@(dv+tau*tr)]
            fit=root(lambda e:direct(e)[1],[res['signal_multiplier'],res['training_noise_multiplier'],res['scale']],options={'xtol':1e-9})
            val,grad=direct(fit.x);assert fit.x[2]>0 and max(abs(grad))<1e-7
            maxdiff=max(maxdiff,abs(val-res['critical_ratio']))
            assert res['critical_ratio']<=bound+1e-8
            rows.append(dict(total_slope_ratio=rho,training_fraction=alpha,auxiliary_per_total_label=nu,
                envelope=res['critical_ratio'],independent_envelope=val,correction_ratio=rho/(1-alpha),
                all_power_gap=res['critical_ratio']-rho/(1-alpha),upper_bound=bound))
    pd.DataFrame(rows).to_csv(out/'all_power_envelopes.csv',index=False)
    pd.DataFrame(risks).to_csv(out/'learned_prediction_risks.csv',index=False)
    assert maxdiff<1e-7 and maxquad<2e-6
    result=dict(envelope_conditions=len(rows),prediction_risk_conditions=len(risks),
        independent_half_line_integral_max_difference=maxdiff,GH_40_80_max_difference=maxquad,
        generic_concavity_upper_bound_checked=True,all_power_selector_new_estimator_claimed=False)
    (out/'envelope_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):calculate(a.base,a.out)
