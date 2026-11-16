"""Matched prediction marginals with an external or fitted Gaussian index.

This extends the same corrected-loss cone calculation by retaining the actual
prediction index among fitted columns. Generic teacher/feature alignment and
Gaussian rotation are established ideas, not new algorithms.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.optimize import brentq,minimize,minimize_scalar,linprog
from scipy.special import roots_legendre,ndtr
from scipy.stats import norm
from threadpoolctl import threadpool_limits
from finite_ratio_signal_boundary import B_terms,tail_derivative,cone_projection_identity
from finite_ratio_choice_boundary import tail
from power_tuned_choice_audit import objective,support
from public_conjoint_optimizer_audit import save


class AlignedPopulation:
    def __init__(self,c,tau,power,nodes=80):
        u,w=roots_legendre(nodes);positive=(u+1)*6
        halfweight=w*6*norm.pdf(positive)
        index=np.r_[-positive,positive];weight=np.r_[halfweight,halfweight]
        self.T=np.tile(index,2);self.y=np.repeat([0.,1.],len(index))
        self.weights=np.tile(weight,2)/2
        self.g=.5+c*np.sign(self.T);self.sign=1-2*self.y
        self.c=c;self.tau=tau;self.power=power
        r=np.where(self.y==0,self.g,1-self.g)
        self.aL=power*r;self.bL=1-power+power*r
        self.aU=-power*self.g/tau;self.bU=power*(1-self.g)/tau
        self.V=np.c_[np.ones(len(self.T)),self.T]
        assert abs(self.weights.sum()-1)<1e-12

    def value(self,eta,t):
        linear=self.V@eta
        left,dl,tl=B_terms(self.sign*linear,t,self.aL,self.bL)
        right,du,tu=B_terms(linear,t,self.aU,self.bU)
        value=float(self.weights@(left+self.tau*right))
        gradient=self.V.T@(self.weights*(self.sign*dl+self.tau*du))
        dt=float(self.weights@(tl+self.tau*tu))
        return value,gradient,dt

    def maximize(self,eta):
        first=self.value(eta,0.)
        if first[2]<=0:return 0.,np.zeros(2),0.
        upper=1.
        while self.value(eta,upper)[2]>0:
            upper*=2
            if upper>1e6:raise ValueError('Scale not bracketed')
        t=brentq(lambda v:self.value(eta,v)[2],0,upper,xtol=1e-12)
        value,gradient,_=self.value(eta,t)
        return value,gradient,t

    def critical(self,aligned):
        if aligned:
            # Joint sign symmetry makes the intercept multiplier zero.
            start=self.maximize(np.zeros(2))[1][1]
            if abs(start)<1e-12:index_eta=0.
            else:
                lower,upper=-1.,1.
                while self.maximize(np.array([0.,lower]))[1][1]>=0:lower*=2
                while self.maximize(np.array([0.,upper]))[1][1]<=0:upper*=2
                index_eta=brentq(lambda e:self.maximize(np.array([0.,e]))[1][1],lower,upper,xtol=1e-11)
            eta=np.array([0.,index_eta])
        else:eta=np.zeros(2)
        value,gradient,t=self.maximize(eta)
        return {'critical_ratio':value,'intercept_multiplier':float(eta[0]),'index_multiplier':float(eta[1]),
                'scale':t,'intercept_gradient':float(gradient[0]),'index_gradient':float(gradient[1]),
                'scale_derivative':self.value(eta,t)[2]}


def intervals(index,y,g,m,power):
    tau=(len(y)-m)/m
    sign=np.r_[1-2*y[:m],np.ones(len(y)-m)]
    r=np.where(y[:m]==0,g[:m],1-g[:m])
    lower=np.r_[power*r,-power*g[m:]/tau]
    upper=np.r_[1-power+power*r,power*(1-g[m:])/tau]
    V=sign[:,None]*np.c_[np.ones(len(y)),index]
    return lower,upper,V,sign


def diagnostic_checks(out):
    boundaries=[];max_quad=0.;max_gradient=0.;max_symmetry=0.;max_scalar=0.;max_derivative=0.
    for c in [.1,.25,.4]:
      for tau in [1.,4.]:
       for power in [1.,.5,.1]:
        coarse=AlignedPopulation(c,tau,power,80);fine=AlignedPopulation(c,tau,power,160)
        ext=coarse.critical(False);inside=coarse.critical(True);fine_ext=fine.critical(False);fine_in=fine.critical(True)
        max_quad=max(max_quad,abs(ext['critical_ratio']-fine_ext['critical_ratio']),abs(inside['critical_ratio']-fine_in['critical_ratio']))
        assert inside['critical_ratio']<=ext['critical_ratio']+1e-10
        eta=np.array([inside['intercept_multiplier'],inside['index_multiplier']]);step=1e-5;eye=np.eye(2)
        fd=np.array([(coarse.maximize(eta+step*e)[0]-coarse.maximize(eta-step*e)[0])/(2*step) for e in eye])
        max_gradient=max(max_gradient,float(max(abs(fd-coarse.maximize(eta)[1]))))
        max_symmetry=max(max_symmetry,abs(coarse.maximize(eta+np.array([.3,0.]))[0]-coarse.maximize(eta-np.array([.3,0.]))[0]))
        assert max(abs(coarse.maximize(eta)[1]))<1e-7
        row={'confidence_amplitude':c,'tau':tau,'power':power,
             'external_critical_ratio':fine_ext['critical_ratio'],'aligned_critical_ratio':fine_in['critical_ratio'],
             'boundary_gap':fine_ext['critical_ratio']-fine_in['critical_ratio'],
             'aligned_index_multiplier':fine_in['index_multiplier'],'aligned_scale':fine_in['scale'],
             'external_scale':fine_ext['scale'],'external_envelope_index_gradient':fine_ext['index_gradient'],
             'Brier_risk_both':.25+c*c,'label_prediction_covariance_both':0.,'prediction_variance_both':c*c}
        if power==1:
            fn=lambda u:tau*(.25+c*c)*u*u+tail((.5+c)*u)+tail((.5-c)*u)
            opt=minimize_scalar(fn,bounds=(0,30),method='bounded',options={'xatol':1e-12})
            exact=tau*(1-opt.fun);max_scalar=max(max_scalar,abs(exact-fine_ext['critical_ratio']))
            t=fine_ext['scale']
            derivative=np.sqrt(2/np.pi)*(-2*t*c+tau*(tail_derivative(t*(.5+c)/tau)-tail_derivative(t*(.5-c)/tau)))
            max_derivative=max(max_derivative,abs(derivative-fine_ext['index_gradient']))
            assert derivative<0 and row['boundary_gap']>0
            row['analytic_external_index_derivative']=derivative
        boundaries.append(row)
    pd.DataFrame(boundaries).to_csv(out/'aligned_prediction_boundaries.csv',index=False)
    numeric={'population_conditions':len(boundaries),'80_160_node_max_error':max_quad,
             'envelope_gradient_finite_difference_error':max_gradient,'intercept_symmetry_error':max_symmetry,
             'external_scalar_unit_boundary_error':max_scalar,'negative_unit_derivative_identity_error':max_derivative,
             'two_sided_normal_tail_mass_bound':float(2*norm.sf(12)),
             'analytic_strict_unit_difference_cases':6,
             'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'new_human_or_AI_responses':0,'new_cross_fitting_algorithm':False,'priority_or_submission_readiness_proven':False}
    save(out/'aligned_prediction_population_attempt.json',numeric)
    assert max_quad<1e-7 and max_gradient<1e-7 and max_symmetry<1e-9 and max_scalar<1e-9 and max_derivative<1e-8,numeric
    rng=np.random.default_rng(20261201);matches=0;pair_counts={};max_projection=0.
    for trial in range(120):
        m=14+trial%9;N=m*(1+trial%3);n=m+N;d=3+trial%14
        index=rng.normal(size=n);other=rng.normal(size=n);W=rng.normal(size=(n,d-1))
        y=rng.integers(0,2,size=n).astype(float);c=[.1,.25,.4][trial%3];g=.5+c*np.sign(index)
        power=[1.,.5,.1][trial%3]
        states=[]
        for label,attribute in [('external',other),('aligned',index)]:
            x=np.c_[np.ones(n),attribute,W]
            original=support(objective(x,y,g,m,power))['status']
            lower,upper,V,sign=intervals(attribute,y,g,m,power)
            A=np.c_[V,sign[:,None]*W].T
            check=linprog(np.zeros(n),A_eq=A,b_eq=np.zeros(A.shape[0]),bounds=list(zip(lower,upper)),method='highs')
            box='outside_mean_support' if check.status==2 else 'interior' if check.success and max(abs(A@check.x))<1e-7 else 'unresolved'
            assert original==box,(trial,label,original,check.message)
            matches+=1;states.append(original)
            if trial<8:max_projection=max(max_projection,cone_projection_identity(lower,upper,V,rng.normal(size=n))[0])
        key='external:'+states[0]+'|aligned:'+states[1];pair_counts[key]=pair_counts.get(key,0)+1
    numeric.update(paired_finite_designs=120,original_oriented_LP_matches=matches,finite_paired_states=pair_counts,
                   independent_projection_cases=16,maximum_projection_norm_squared_error=max_projection)
    save(out/'aligned_prediction_numeric_checks.json',numeric)
    print(json.dumps(numeric),flush=True)
    print(pd.DataFrame(boundaries)[['confidence_amplitude','tau','power','external_critical_ratio','aligned_critical_ratio','boundary_gap']].to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):diagnostic_checks(a.out)
