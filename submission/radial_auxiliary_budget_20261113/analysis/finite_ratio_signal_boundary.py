"""Finite-ratio signal candidate: independent LPs, cone projection and phase checks.

Only simulated Gaussian choice rows are generated. The generic conic machinery
and ordinary-logit endpoint are established prior work, not contribution claims.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.optimize import brentq,linprog,minimize,LinearConstraint,Bounds
from scipy.special import expit,ndtr
from scipy.stats import norm
from scipy.special import roots_hermitenorm
from scipy.linalg import null_space
from finite_ratio_choice_boundary import tail,critical_ratio
from power_tuned_choice_audit import objective,support
from public_conjoint_optimizer_audit import save


def tail_derivative(a):return 2*(a*ndtr(-a)-norm.pdf(a))


def B_terms(h,t,a,b):
    lo=-h-t*a;hi=h+t*b
    value=1+h*h-tail(lo)-tail(hi)
    dh=2*h+tail_derivative(lo)-tail_derivative(hi)
    dt=a*tail_derivative(lo)-b*tail_derivative(hi)
    return value,dh,dt


class Population:
    def __init__(self,gamma,epsilon,tau,lam,nodes=100):
        s,w=roots_hermitenorm(nodes);w=w/np.sqrt(2*np.pi)
        self.s=np.tile(s,4);self.weights=[];self.y=[];self.z=[]
        p=expit(gamma*s)
        for y in [0,1]:
            for z in [0,1]:
                self.y.extend([y]*nodes);self.z.extend([z]*nodes)
                self.weights.extend(w*(p if y else 1-p)*((1-epsilon) if y==z else epsilon))
        self.y=np.array(self.y);self.z=np.array(self.z);self.weights=np.array(self.weights)
        self.g=expit(gamma*self.s+(2*self.z-1)*np.log((1-epsilon)/epsilon))
        self.signL=1-2*self.y;self.signU=2*self.z-1
        r=np.where(self.y==0,self.g,1-self.g)
        self.aL=lam*r;self.bL=1-lam+lam*r
        self.aU=lam/tau*np.where(self.z==1,-self.g,self.g-1)
        self.bU=lam/tau*np.where(self.z==1,1-self.g,self.g)
        self.tau=tau

    def value(self,eta,t):
        h0,h1=eta;hL=self.signL*(h0+h1*self.s);hU=self.signU*(h0+h1*self.s)
        vL,dL,tL=B_terms(hL,t,self.aL,self.bL);vU,dU,tU=B_terms(hU,t,self.aU,self.bU)
        v=self.weights@(vL+self.tau*vU)
        d=self.weights@(dL*self.signL+self.tau*dU*self.signU)
        d1=self.weights@((dL*self.signL+self.tau*dU*self.signU)*self.s)
        dt=self.weights@(tL+self.tau*tU)
        return float(v),np.array([d,d1]),float(dt)

    def maximize(self,eta):
        at0=self.value(eta,0.)
        if at0[2]<=0:return 0.,np.zeros(2),0.
        upper=1.
        while self.value(eta,upper)[2]>=0:
            upper*=2
            if upper>1e6:raise RuntimeError('signal-scale maximizer not bracketed')
        t=brentq(lambda t:self.value(eta,t)[2],0.,upper,xtol=1e-11)
        val,grad,_=self.value(eta,t)
        return val,grad,t

    def critical(self):
        grad0=self.maximize((0.,0.))[1][1]
        if abs(grad0)<1e-11:eta=0.
        else:
            lower=-1.;upper=1.
            while self.maximize((0.,lower))[1][1]>=0:lower*=2
            while self.maximize((0.,upper))[1][1]<=0:upper*=2
            eta=brentq(lambda e:self.maximize((0.,e))[1][1],lower,upper,xtol=1e-10)
        value,grad,t=self.maximize((0.,eta))
        return {'critical_ratio':value,'signal_multiplier':eta,'scale':t,'gradient_max':float(max(abs(grad))),'scale_derivative':self.value((0.,eta),t)[2]}


def intervals(s,y,z,m,gamma,epsilon,lam):
    N=len(y)-m;tau=N/m;g=expit(gamma*s+(2*z-1)*np.log((1-epsilon)/epsilon))
    r=np.where(y[:m]==0,g[:m],1-g[:m]);sign=np.r_[1-2*y[:m],2*z[m:]-1]
    a=np.r_[lam*r,lam/tau*np.where(z[m:]==1,-g[m:],g[m:]-1)]
    b=np.r_[1-lam+lam*r,lam/tau*np.where(z[m:]==1,1-g[m:],g[m:])]
    V=np.column_stack([sign,sign*s])
    return a,b,V,sign,g


def box_feasibility(s,y,z,orth,m,gamma,epsilon,lam):
    a,b,V,sign,_=intervals(s,y,z,m,gamma,epsilon,lam)
    A=np.column_stack([V,orth*sign[:,None]]).T
    fit=linprog(np.zeros(len(y)),A_eq=A,b_eq=np.zeros(A.shape[0]),bounds=list(zip(a,b)),method='highs')
    if fit.status==2:return 'outside'
    if not fit.success or max(abs(A@fit.x))>1e-6:return 'unresolved'
    return 'feasible'


def cone_projection_identity(a,b,V,H):
    n=len(H)
    def empirical(eta,t):
        h=H-V@eta
        q=np.clip(h,t*a,t*b);value=np.sum(2*h*q-q*q)
        return value,-2*V.T@q,2*np.sum(np.where(h<t*a,a*(h-t*a),np.where(h>t*b,b*(h-t*b),0.)))
    def maximized(eta):
        if empirical(eta,0.)[2]<=0:return 0.,np.zeros(V.shape[1]),0.
        upper=1.
        while empirical(eta,upper)[2]>=0:upper*=2
        t=brentq(lambda t:empirical(eta,t)[2],0.,upper,xtol=1e-11)
        val,grad,_=empirical(eta,t);return val,grad,t
    dual=minimize(lambda e:maximized(e)[:2],np.zeros(V.shape[1]),method='BFGS',jac=True,options={'gtol':1e-9,'maxiter':2000})
    val,grad,t=maximized(dual.x)
    fixed=np.isclose(a,b,atol=1e-15,rtol=0.)
    eq=np.column_stack([V.T,np.zeros(V.shape[1])])
    if fixed.any():eq=np.vstack([eq,np.column_stack([np.eye(n)[fixed],-a[fixed]])])
    nonfixed=~fixed
    inequalities=np.vstack([np.column_stack([np.eye(n)[nonfixed],-a[nonfixed]]),np.column_stack([-np.eye(n)[nonfixed],b[nonfixed]])])
    constraints=[LinearConstraint(eq,np.zeros(len(eq)),np.zeros(len(eq))),LinearConstraint(inequalities,np.zeros(len(inequalities)),np.full(len(inequalities),np.inf))]
    def primal(w):return .5*np.sum((w[:n]-H)**2),np.r_[w[:n]-H,0.]
    fit=minimize(primal,np.zeros(n+1),jac=True,method='SLSQP',constraints=constraints,bounds=Bounds(np.r_[np.full(n,-np.inf),0.],np.full(n+1,np.inf)),options={'ftol':1e-11,'maxiter':2000})
    assert fit.success,(fit.message,dual.message)
    # Polish the attained active face independently of the saddle solution.
    # A tiny objective error can otherwise produce a larger norm-squared error.
    active=abs(inequalities@fit.x)<1e-7
    face=np.vstack([eq,inequalities[active]])
    if fit.x[-1]<1e-7:face=np.vstack([face,np.r_[np.zeros(n),1.]])
    basis=null_space(face)
    candidate=basis@np.linalg.lstsq(basis[:n],H,rcond=None)[0] if basis.shape[1] else np.zeros(n+1)
    if (candidate[-1]>=-1e-8 and max(abs(eq@candidate))<1e-8
        and min(inequalities@candidate)>-1e-8 and primal(candidate)[0]<=fit.fun+1e-10):fit.x=candidate
    projected=fit.x[:n];primal_value=float(projected@projected)
    assert max(abs(eq@fit.x))<1e-7 and np.min(inequalities@fit.x)>-1e-7
    error=abs(primal_value-val)
    assert error<1e-6,(primal_value,val,grad,dual.message)
    return error,float(max(abs(grad)))


def checks():
    nullerror=0.;quadrature=0.;stationarity=0.;symmetry=0.;ordinary=[];boundaries=[]
    for tau in [1.,4.]:
        for lam in [1.,tau/(1+tau)]:
            for gamma in [0.,1.,2.]:
                low=Population(gamma,.25,tau,lam,100);high=Population(gamma,.25,tau,lam,200)
                result=low.critical();other=high.critical();quadrature=max(quadrature,abs(result['critical_ratio']-other['critical_ratio']))
                stationarity=max(stationarity,result['gradient_max'],abs(result['scale_derivative']))
                eta=result['signal_multiplier'];plus=low.maximize((.3,eta))[0];minus=low.maximize((-.3,eta))[0]
                symmetry=max(symmetry,abs(plus-minus));assert result['critical_ratio']<=plus+1e-10
                if gamma==0:nullerror=max(nullerror,abs(result['critical_ratio']-critical_ratio(tau,.25,lam)[0]))
                boundaries.append({'gamma':gamma,'epsilon':.25,'tau':tau,'power':lam,**other})
    assert nullerror<1e-8 and quadrature<1e-7 and stationarity<1e-7 and symmetry<1e-9
    # Classical ordinary-logit endpoint, integrated independently over Y and S.
    s,w=roots_hermitenorm(200);w=w/np.sqrt(2*np.pi)
    for gamma in [0.,1.,2.]:
        p=expit(gamma*s)
        fn=lambda eta:float(w@((1-p)*tail(eta*s)+p*tail(-eta*s)))
        fp=lambda eta:float(w@((1-p)*tail_derivative(eta*s)*s-p*tail_derivative(-eta*s)*s))
        eta=brentq(fp,-20.,20.)
        ordinary.append({'gamma':gamma,'critical_ratio':fn(eta),'signal_multiplier':eta,'attributed_to':'classical ordinary-logit boundary'})
    assert abs(ordinary[0]['critical_ratio']-.5)<1e-12
    rng=np.random.default_rng(20261115);matches=0;counts={};projectionerror=0.;projectiongrad=0.
    for j in range(120):
        m=12+j%7;N=m*(1+j%3);q=2+j%16;gamma=[0.,1.,2.][j%3];lam=[.3,.7,1.][j%3];epsilon=.25
        s=rng.normal(size=m+N);y=(rng.random(m+N)<expit(gamma*s)).astype(float)
        z=np.logical_xor(y,rng.random(m+N)<epsilon).astype(float);orth=rng.normal(size=(m+N,q))
        a,b,V,sign,g=intervals(s,y,z,m,gamma,epsilon,lam)
        x=np.column_stack([np.ones(m+N),s,orth])
        original=support(objective(x,y,g,m,lam))['status'];box=box_feasibility(s,y,z,orth,m,gamma,epsilon,lam)
        expected='feasible' if original=='interior' else 'outside' if original=='outside_mean_support' else 'unresolved'
        assert box==expected,(j,original,box);matches+=1;counts[original]=counts.get(original,0)+1
        if j<12:
            pe,pg=cone_projection_identity(a,b,V,rng.normal(size=m+N));projectionerror=max(projectionerror,pe);projectiongrad=max(projectiongrad,pg)
    return {'null_reduction_max_error':nullerror,'100_200_node_boundary_max_error':quadrature,'saddle_stationarity_max_error':stationarity,'intercept_symmetry_max_error':symmetry,'original_interval_LP_matches':matches,'LP_status_counts':counts,'cone_projection_identity_cases':12,'cone_projection_max_error':projectionerror,'cone_dual_gradient_max':projectiongrad,'ordinary_logit_endpoint':ordinary,'independent_priority_verified':False},boundaries


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--checks-only',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    result,bounds=checks();save(args.out/'finite_ratio_signal_numeric_checks.json',result);pd.DataFrame(bounds).to_csv(args.out/'finite_ratio_signal_boundaries.csv',index=False)
    print(json.dumps(result),flush=True)
    if args.checks_only:return
    rng=np.random.default_rng(20261116);rows=[]
    for m in [40,80]:
        for tau in [1,4]:
            N=m*tau
            for gamma in [1.,2.]:
                for name,lam in [('unit',1.),('ratio_power',tau/(1+tau))]:
                    boundary=Population(gamma,.25,tau,lam,200).critical()['critical_ratio']
                    dims=sorted({max(1,int(round(m*(boundary+offset)))) for offset in [-.15,-.075,0.,.075,.15]})
                    counts={q:{'feasible':0,'outside':0,'unresolved':0} for q in dims}
                    for rep in range(300):
                        s=rng.normal(size=m+N);y=(rng.random(m+N)<expit(gamma*s)).astype(float)
                        z=np.logical_xor(y,rng.random(m+N)<.25).astype(float);orth=rng.normal(size=(m+N,max(dims)))
                        for q in dims:counts[q][box_feasibility(s,y,z,orth[:,:q],m,gamma,.25,lam)]+=1
                    for q,count in counts.items():rows.append({'labels':m,'auxiliary_rows':N,'tau':tau,'gamma':gamma,'epsilon':.25,'power_name':name,'power':lam,'orthogonal_dimensions':q,'orthogonal_dimension_label_ratio':q/m,'critical_ratio':boundary,'replications':300,**count,'finite_frequency':count['feasible']/300})
                    pd.DataFrame(rows).to_csv(args.out/'finite_ratio_signal_phase.csv',index=False)
                    save(args.out/'finite_ratio_signal_status.json',{'completed_cells':len(rows),'requested_cells':80,'replications_per_cell':300,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_priority_verified':False})
                    print('Signal phase cells complete',len(rows),'of80',flush=True)
    print('All signal phase cells complete.',flush=True)


if __name__=='__main__':main()
