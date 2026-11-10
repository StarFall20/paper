"""General frozen-low-index all-power boundary and independent population checks.

Generic conic geometry,ordinary-logit separation and tuning are prior machinery.
This file evaluates a specific compensated loss and proves no inference guarantee.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.optimize import brentq,minimize
from scipy.special import expit,ndtr,roots_hermitenorm
from scipy.stats import norm
from threadpoolctl import threadpool_limits
from finite_ratio_signal_boundary import Population,B_terms,tail_derivative
from finite_ratio_choice_boundary import tail
from imperfect_predictor_choice_boundary import ImperfectPopulation
from vanishing_power_choice import positive_limit,rescaled_support
from power_tuned_choice_audit import objective,support
from public_conjoint_optimizer_audit import save

PREDICTORS=[('oracle',1.,0.),('overconfident',2.,0.),('intercept_shifted',1.,1.),
            ('reversed',-1.,0.),('constant_one_half',0.,0.)]


def ordinary_boundary(gamma,nodes=200):
    s,w=roots_hermitenorm(nodes);w=w/np.sqrt(2*np.pi);p=expit(gamma*s)
    fun=lambda eta:float(w@((1-p)*tail(eta*s)+p*tail(-eta*s)))
    grad=lambda eta:float(w@((1-p)*tail_derivative(eta*s)*s-p*tail_derivative(-eta*s)*s))
    eta=brentq(grad,-30.,30.)
    return fun(eta)


class GeneralLimitPopulation(Population):
    def __init__(self,gamma,epsilon,tau,a,b,nodes=100):
        super().__init__(gamma,epsilon,tau,1.,nodes)
        logit=a*(gamma*self.s+(2*self.z-1)*np.log((1-epsilon)/epsilon))+b
        self.g=expit(logit);complement=expit(-logit);self.complement=complement
        # These algebraic complements avoid cancellation at extreme quadrature nodes.
        self.r=expit(self.signL*logit)
        self.aU=np.where(self.z==1,-self.g,-complement)/tau
        self.bU=np.where(self.z==1,complement,self.g)/tau

    def value(self,eta,v):
        h=eta[0]+eta[1]*self.s;hL=self.signL*h;hU=self.signU*h
        arg=-hL-v*self.r
        label=1+hL*hL-tail(arg);label_h=2*hL+tail_derivative(arg)
        label_v=self.r*tail_derivative(arg)
        aux,aux_h,aux_v=B_terms(hU,v,self.aU,self.bU)
        d=label_h*self.signL+self.tau*aux_h*self.signU
        return float(self.weights@(label+self.tau*aux)),np.array([self.weights@d,self.weights@(d*self.s)]),float(self.weights@(label_v+self.tau*aux_v))

    def maximize(self,eta):
        at0=self.value(eta,0.)
        if at0[2]<=0:
            raise ArithmeticError(('Population zero-scale derivative must be positive',eta,at0))
        hi=1.
        while self.value(eta,hi)[2]>=0:
            hi*=2
            if hi>1e8:raise ArithmeticError('Limiting scale not bracketed')
        v=brentq(lambda t:self.value(eta,t)[2],0.,hi,xtol=1e-11)
        val,grad,_=self.value(eta,v)
        return val,grad,v

    def envelope_hessian(self,eta,v):
        VL=np.column_stack([self.signL,self.signL*self.s]);VU=np.column_stack([self.signU,self.signU*self.s])
        hL=VL@eta;hU=VU@eta;ff=2*ndtr(hL+v*self.r)
        hh=VL.T@((self.weights*(2-ff))[:,None]*VL)
        ht=VL.T@(-self.weights*self.r*ff);tt=-float(self.weights@(self.r*self.r*ff))
        lo=2*ndtr(hU+v*self.aU);hi=2*ndtr(-hU-v*self.bU);w=self.tau*self.weights
        hh+=VU.T@((w*(2-lo-hi))[:,None]*VU)
        ht+=VU.T@(w*(-self.aU*lo-self.bU*hi))
        tt+=float(w@(-self.aU*self.aU*lo-self.bU*self.bU*hi));assert tt<0
        return hh-np.outer(ht,ht)/tt

    def critical_general(self):
        fit=minimize(lambda eta:self.maximize(eta)[:2],np.zeros(2),jac=True,method='BFGS',options={'gtol':1e-10,'maxiter':1200})
        eta=fit.x.copy()
        for _ in range(10):
            val,grad,v=self.maximize(eta)
            if max(abs(grad))<1e-10:break
            step=np.linalg.solve(self.envelope_hessian(eta,v),grad);scale=1.
            while scale>1e-9 and self.maximize(eta-scale*step)[0]>val+1e-12:scale/=2
            if scale<=1e-9:break
            eta-=scale*step
        val,grad,v=self.maximize(eta)
        assert max(abs(grad))<1e-7 and 0<val<.5+self.tau,(fit.message,eta,val,grad)
        return {'critical_ratio':val,'intercept_multiplier':float(eta[0]),'signal_multiplier':float(eta[1]),
                'scale':v,'gradient_max':float(max(abs(grad))),'scale_derivative':self.value(eta,v)[2]}

    def positive_derivative_identity(self,eta):
        h=eta[0]+eta[1]*self.s
        M=lambda z:norm.pdf(z)+z*ndtr(z)
        rhs=2*float(self.weights@(self.y*self.g*M(h)+(1-self.y)*self.complement*M(-h)))
        return self.value(eta,0.)[2],rhs


def population_checks(out):
    rows=[];quaderr=0.;gradienterr=0.;hessianerr=0.;identityerr=0.;nullerr=0.;fixederr=0.;monotone_excess=0.
    for gamma in [0.,1.5,2.]:
      ordinary=ordinary_boundary(gamma,400)
      for tau in [1.,4.]:
       for name,a,b in PREDICTORS:
        low=GeneralLimitPopulation(gamma,.25,tau,a,b,200);high=GeneralLimitPopulation(gamma,.25,tau,a,b,400)
        coarse=low.critical_general();fine=high.critical_general();quaderr=max(quaderr,abs(coarse['critical_ratio']-fine['critical_ratio']))
        eta=np.array([fine['intercept_multiplier'],fine['signal_multiplier']]);v=fine['scale']
        for location in [np.array([.2,-.3]),eta]:
            lhs,rhs=high.positive_derivative_identity(location);identityerr=max(identityerr,abs(lhs-rhs));assert rhs>0
        step=1e-5;eye=np.eye(2)
        fd=np.array([(high.maximize(eta+e*step)[0]-high.maximize(eta-e*step)[0])/(2*step) for e in eye])
        gradienterr=max(gradienterr,float(max(abs(fd-high.maximize(eta)[1]))))
        fdh=np.column_stack([(high.maximize(eta+e*step)[1]-high.maximize(eta-e*step)[1])/(2*step) for e in eye])
        hessianerr=max(hessianerr,float(abs(fdh-high.envelope_hessian(eta,v)).max()))
        assert fine['critical_ratio']>ordinary+1e-8
        if gamma==0 and name in ['oracle','constant_one_half']:
            expected=positive_limit(tau,.25 if name=='oracle' else .5)[0]
            nullerr=max(nullerr,abs(expected-fine['critical_ratio']))
        prev=-np.inf
        for lam in [1.,.5,.1,.01]:
            value=ImperfectPopulation(gamma,.25,tau,lam,a,b,400).critical_general()
            monotone_excess=max(monotone_excess,prev-value['critical_ratio'],value['critical_ratio']-fine['critical_ratio']);prev=value['critical_ratio']
            if lam==.01:fixederr=max(fixederr,abs(value['critical_ratio']-fine['critical_ratio']))
            rows.append({'gamma':gamma,'tau':tau,'prediction':name,'a':a,'b':b,'power':lam,
                'finite_power_critical_ratio':value['critical_ratio'],'all_power_critical_ratio':fine['critical_ratio'],
                'ordinary_critical_ratio':ordinary,'limiting_intercept_multiplier':fine['intercept_multiplier'],
                'limiting_signal_multiplier':fine['signal_multiplier'],'limiting_scale':v,
                'zero_scale_derivative':high.value(eta,0.)[2]})
    save(out/'general_joint_power_population_refined_attempt.json',{'200_400_node_max_error':quaderr,'gradient_error':gradienterr,'hessian_error':hessianerr,'identity_error':identityerr,'null_error':nullerr,'monotonicity_excess':monotone_excess,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    assert quaderr<1e-7 and gradienterr<1e-7 and hessianerr<1e-6,(quaderr,gradienterr,hessianerr)
    assert identityerr<1e-9 and nullerr<1e-8 and monotone_excess<1e-8
    pd.DataFrame(rows).to_csv(out/'general_joint_power_boundaries.csv',index=False)
    result={'population_conditions':30,'finite_power_comparisons':120,'200_400_node_max_error':quaderr,
        'envelope_gradient_fd_error':gradienterr,'envelope_hessian_fd_error':hessianerr,
        'zero_scale_derivative_identity_error':identityerr,'balanced_scalar_reduction_error':nullerr,
        'monotonicity_numeric_tolerance_excess':monotone_excess,'maximum_power_point01_limit_gap':fixederr,
        'ordinary_existence_boundary_strictly_lower_in_all30conditions':True,
        'coefficient_error_or_interval_guarantee':False,'priority_proven':False}
    save(out/'general_joint_power_population_checks.json',result);return result


def phase(out):
    frame=pd.read_csv(out/'general_joint_power_boundaries.csv');rng=np.random.default_rng(20261127)
    rows=[];done=0;unknowns=[];conflicts=[]
    for m in [40,80]:
      for tau in [1,4]:
       N=tau*m
       for name,a,b in PREDICTORS[:3]:
        spec=frame[(frame.gamma==1.5)&(frame.tau==tau)&(frame.prediction==name)].iloc[0]
        dims=sorted({max(1,round(m*(spec.all_power_critical_ratio+offset))) for offset in [-.1,0.,.1]})
        assert len(dims)==3
        names=['ordinary_zero','unit','inverse_sqrt_m','inverse_m','positive_power_limit']
        counts={(q,power):{} for q in dims for power in names}
        for rep in range(500):
            s=rng.normal(size=m+N);y=(rng.random(m+N)<expit(1.5*s)).astype(float)
            z=np.logical_xor(y,rng.random(m+N)<.25).astype(float);orth=rng.normal(size=(m+N,max(dims)))
            g=expit(a*(1.5*s+(2*z-1)*np.log(3.))+b)
            for q in dims:
                x=np.column_stack([np.ones(m+N),s,orth[:,:q]])
                statuses={}
                for power in names:
                    lam={'ordinary_zero':0.,'unit':1.,'inverse_sqrt_m':1/np.sqrt(m),'inverse_m':1/m,'positive_power_limit':None}[power]
                    raw=support(objective(x,y,g,m,0.)) if power=='ordinary_zero' else rescaled_support(x,y,g,m,lam)
                    status=raw['status'];known={'interior','outside_mean_support','boundary_or_near_boundary','rank_deficient'}
                    if status not in known:
                        unknowns.append({'labels':m,'tau':tau,'prediction':name,'orthogonal_dimensions':q,'replication':rep,'power_name':power,'power':lam,'raw_solver_result':raw})
                        status='unresolved'
                    c=counts[q,power];c[status]=c.get(status,0)+1
                    statuses[power]=status
                chain=[statuses[p] for p in ['unit','inverse_sqrt_m','inverse_m','positive_power_limit']]
                for first,second in zip(chain,chain[1:]):
                    if first=='interior' and second in ['outside_mean_support','boundary_or_near_boundary']:
                        conflicts.append({'labels':m,'tau':tau,'prediction':name,'orthogonal_dimensions':q,'replication':rep,'statuses':statuses})
                if statuses['ordinary_zero']=='interior' and statuses['positive_power_limit'] in ['outside_mean_support','boundary_or_near_boundary']:
                    conflicts.append({'labels':m,'tau':tau,'prediction':name,'orthogonal_dimensions':q,'replication':rep,'statuses':statuses})
        for (q,power),c in counts.items():
            rows.append({'labels':m,'auxiliary_rows':N,'tau':tau,'gamma':1.5,'prediction':name,
                'orthogonal_dimensions':q,'orthogonal_dimension_label_ratio':q/m,'power_name':power,'replications':500,
                'interior':c.get('interior',0),'outside':c.get('outside_mean_support',0),
                'boundary_or_near_boundary':c.get('boundary_or_near_boundary',0),'rank_deficient':c.get('rank_deficient',0),
                'unresolved':c.get('unresolved',0),'finite_frequency':c.get('interior',0)/500,
                'all_power_critical_ratio':spec.all_power_critical_ratio,'ordinary_critical_ratio':spec.ordinary_critical_ratio})
        done+=1;pd.DataFrame(rows).to_csv(out/'general_joint_power_phase_raw.csv',index=False)
        save(out/'general_joint_power_unresolved_ids.json',{'cases':unknowns,'possible_nesting_conflicts':conflicts,'raw_outputs_overwritten':False})
        save(out/'general_joint_power_status.json',{'completed_DGP_blocks':done,'requested_DGP_blocks':12,
            'completed_aggregate_cells':len(rows),'requested_aggregate_cells':180,'replications_per_block':500,
            'independent_Gaussian_simulation_draws':done*500,'new_humans_or_AI_responses':False,
            'unresolved':int(pd.DataFrame(rows).unresolved.sum()),'possible_nesting_conflicts':len(conflicts),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        print('General joint-power DGP blocks complete',done,'of12',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--checks-only',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        print(json.dumps(population_checks(a.out)),flush=True)
        if not a.checks_only:phase(a.out)
