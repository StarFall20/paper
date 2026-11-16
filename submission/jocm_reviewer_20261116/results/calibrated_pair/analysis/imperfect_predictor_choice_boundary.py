"""Validate the frozen low-index boundary beyond an oracle predictor."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.optimize import minimize,brentq,linprog,minimize_scalar
from scipy.special import expit,ndtr
from finite_ratio_signal_boundary import Population,intervals,cone_projection_identity,B_terms
from finite_ratio_choice_boundary import tail
from power_tuned_choice_audit import objective,support
from public_conjoint_optimizer_audit import save


class ImperfectPopulation(Population):
    def __init__(self,gamma,epsilon,tau,lam,a,b,nodes=100):
        super().__init__(gamma,epsilon,tau,lam,nodes)
        self.g=expit(a*(gamma*self.s+(2*self.z-1)*np.log((1-epsilon)/epsilon))+b)
        r=np.where(self.y==0,self.g,1-self.g)
        self.aL=lam*r;self.bL=1-lam+lam*r
        self.aU=lam/tau*np.where(self.z==1,-self.g,self.g-1)
        self.bU=lam/tau*np.where(self.z==1,1-self.g,self.g)

    def moments(self,gamma):
        mu=expit(gamma*self.s);A=mu-self.y;B=mu-self.g
        covariance=float(self.weights@(A*B));pred_variance=float(self.weights@(B*B))
        power=float(np.clip(self.tau/(1+self.tau)*covariance/pred_variance,0,1)) if pred_variance>1e-14 else 0.
        return {'Brier_risk':float(self.weights@((self.y-self.g)**2)),
            'orthogonal_score_covariance':covariance,'orthogonal_prediction_score_variance':pred_variance,
            'orthogonal_variance_optimal_power':power,'weight_formula_is_prior_work':True}

    def envelope_hessian(self,eta,t):
        V1=np.column_stack([self.signL,self.signL*self.s]);V2=np.column_stack([self.signU,self.signU*self.s])
        hh=np.zeros((2,2));ht=np.zeros(2);tt=0.
        for V,a,b,w in [(V1,self.aL,self.bL,self.weights),(V2,self.aU,self.bU,self.tau*self.weights)]:
            h=V@eta;flo=2*ndtr(h+t*a);fhi=2*ndtr(-h-t*b)
            d2=2-flo-fhi;cross=-a*flo-b*fhi;dtt=-a*a*flo-b*b*fhi
            hh+=V.T@((w*d2)[:,None]*V);ht+=V.T@(w*cross);tt+=float(w@dtt)
        assert tt<0
        return hh-np.outer(ht,ht)/tt

    def critical_general(self):
        fit=minimize(lambda eta:self.maximize(eta)[:2],np.zeros(2),jac=True,method='BFGS',options={'gtol':1e-10,'maxiter':1000})
        eta=fit.x.copy()
        for _ in range(8):
            val,gradient,t=self.maximize(eta)
            if max(abs(gradient))<1e-10:break
            step=np.linalg.solve(self.envelope_hessian(eta,t),gradient);scale=1.
            while scale>1e-8 and self.maximize(eta-scale*step)[0]>val+1e-12:scale/=2
            if scale<=1e-8:break
            eta-=scale*step
        val,grad,t=self.maximize(eta)
        assert max(abs(grad))<1e-7,(fit.message,eta,grad)
        assert 0<val<1+self.tau
        return {'critical_ratio':val,'intercept_multiplier':float(eta[0]),'signal_multiplier':float(eta[1]),'scale':t,
            'gradient_max':float(max(abs(grad))),'scale_derivative':self.value(eta,t)[2]}


def imperfect_intervals(s,y,z,m,gamma,epsilon,lam,a,b):
    lower,upper,V,sign,_=intervals(s,y,z,m,gamma,epsilon,lam)
    g=expit(a*(gamma*s+(2*z-1)*np.log((1-epsilon)/epsilon))+b);N=len(y)-m;tau=N/m
    r=np.where(y[:m]==0,g[:m],1-g[:m])
    lower=np.r_[lam*r,lam/tau*np.where(z[m:]==1,-g[m:],g[m:]-1)]
    upper=np.r_[1-lam+lam*r,lam/tau*np.where(z[m:]==1,1-g[m:],g[m:])]
    return lower,upper,V,sign,g


def feasible(s,y,z,orth,m,gamma,epsilon,lam,a,b):
    lower,upper,V,sign,_=imperfect_intervals(s,y,z,m,gamma,epsilon,lam,a,b)
    A=np.column_stack([V,orth*sign[:,None]]).T
    fit=linprog(np.zeros(len(y)),A_eq=A,b_eq=np.zeros(A.shape[0]),bounds=list(zip(lower,upper)),method='highs')
    if fit.status==2:return 'outside'
    if not fit.success or max(abs(A@fit.x))>1e-6:return 'unresolved'
    return 'feasible'


def matched_moment_values():
    # Fixed deterministic construction, not an estimate from simulation.
    c=.4;high=.499;p_high=c*c/(high*high);cov=.9*p_high*high/2
    R=.25+c*c-2*cov;rows=[]
    for tau in [.05,.1,.25,1.,4.]:
        for name,amplitudes,weights in [('constant_confidence',[c],[1.]),('mixed_confidence',[0.,high],[1-p_high,p_high])]:
            fn=lambda u:tau*R*u*u+sum(w*(tail((.5+v)*u)+tail((.5-v)*u)) for v,w in zip(amplitudes,weights))
            fit=minimize_scalar(fn,bounds=(0.,80.),method='bounded',options={'xatol':1e-12})
            rows.append({'tau':tau,'prediction':name,'Brier_risk':R,'prediction_variance':c*c,'label_prediction_covariance':cov,
                'unit_power_score_variance_times_m':R+c*c/tau,'critical_ratio':tau*(1-fit.fun),'scaled_minimizer':fit.x,
                'high_confidence_probability':1. if name=='constant_confidence' else p_high,
                'high_confidence_amplitude':c if name=='constant_confidence' else high,
                'high_confidence_proxy_flip_error':.5-cov/c if name=='constant_confidence' else .05})
    return rows


def checks():
    cases=[('oracle',1.,0.),('weakened',.5,0.),('overconfident',2.,0.),('intercept_bias',1.,1.),('reversed',-1.,0.)]
    values=[];quadrature=0.;gradienterror=0.;hessianerror=0.
    for tau in [1.,4.]:
        for name,a,b in cases:
            moments=ImperfectPopulation(1.5,.25,tau,1.,a,b,200).moments(1.5)
            for power_name,lam in [('unit',1.),('orthogonal_optimal',moments['orthogonal_variance_optimal_power'])]:
                if lam==0:
                    values.append({'tau':tau,'prediction':name,'a':a,'b':b,'power_name':power_name,'power':lam,'status':'ordinary_endpoint_separate',**moments});continue
                low=ImperfectPopulation(1.5,.25,tau,lam,a,b,100);high=ImperfectPopulation(1.5,.25,tau,lam,a,b,200)
                result=low.critical_general();fine=high.critical_general();quadrature=max(quadrature,abs(result['critical_ratio']-fine['critical_ratio']))
                eta=np.array([result['intercept_multiplier'],result['signal_multiplier']]);h=1e-5
                analytic=low.maximize(eta)[1];fd=np.array([(low.maximize(eta+np.eye(2)[j]*h)[0]-low.maximize(eta-np.eye(2)[j]*h)[0])/(2*h) for j in range(2)])
                gradienterror=max(gradienterror,float(max(abs(analytic-fd))))
                matrix=low.envelope_hessian(eta,result['scale']);fdh=np.column_stack([(low.maximize(eta+np.eye(2)[j]*h)[1]-low.maximize(eta-np.eye(2)[j]*h)[1])/(2*h) for j in range(2)])
                hessianerror=max(hessianerror,float(max(abs(matrix-fdh).ravel())))
                if b==0:assert abs(fine['intercept_multiplier'])<1e-7
                if b==1:assert abs(fine['intercept_multiplier'])>.01
                values.append({'tau':tau,'prediction':name,'a':a,'b':b,'power_name':power_name,'power':lam,'status':'positive_power',**fine,**moments})
    assert quadrature<1e-7 and gradienterror<1e-7 and hessianerror<1e-6,(quadrature,gradienterror,hessianerror)
    rng=np.random.default_rng(20261118);counts={};projection_error=0.;matches=0
    for j in range(120):
        m=12+j%7;N=m*(1+j%3);q=2+j%16;name,a,b=cases[j%len(cases)];lam=[.3,.7,1.][j%3]
        s=rng.normal(size=m+N);y=(rng.random(m+N)<expit(1.5*s)).astype(float)
        z=np.logical_xor(y,rng.random(m+N)<.25).astype(float);orth=rng.normal(size=(m+N,q))
        lower,upper,V,sign,g=imperfect_intervals(s,y,z,m,1.5,.25,lam,a,b)
        x=np.column_stack([np.ones(m+N),s,orth]);original=support(objective(x,y,g,m,lam))['status'];box=feasible(s,y,z,orth,m,1.5,.25,lam,a,b)
        expected='feasible' if original=='interior' else 'outside' if original=='outside_mean_support' else 'unresolved'
        assert box==expected,(j,original,box);matches+=1;counts[original]=counts.get(original,0)+1
        if j<12:projection_error=max(projection_error,cone_projection_identity(lower,upper,V,rng.normal(size=m+N))[0])
    moments=matched_moment_values()
    frame=pd.DataFrame(moments)
    for tau,part in frame.groupby('tau'):
        assert part.Brier_risk.nunique()==part.prediction_variance.nunique()==part.label_prediction_covariance.nunique()==1
        assert part.iloc[0].critical_ratio>part.iloc[1].critical_ratio
    return {'100_200_node_max_error':quadrature,'envelope_gradient_finite_difference_error':gradienterror,'envelope_hessian_finite_difference_error':hessianerror,'original_interval_LP_matches':matches,'LP_status_counts':counts,'cone_projection_cases':12,'cone_projection_max_error':projection_error,'matched_moment_examples':len(moments),'independent_priority_verified':False},values,moments


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--checks-only',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    numeric,values,matched=checks();save(args.out/'imperfect_prediction_numeric_checks.json',numeric)
    pd.DataFrame(values).to_csv(args.out/'imperfect_prediction_boundaries.csv',index=False);pd.DataFrame(matched).to_csv(args.out/'matched_moment_confidence_boundaries.csv',index=False);print(json.dumps(numeric),flush=True)
    if args.checks_only:return
    rng=np.random.default_rng(20261119);rows=[];frame=pd.DataFrame(values)
    for m in [40,80]:
        for tau in [1,4]:
            N=tau*m
            for name,a,b in [('overconfident',2.,0.),('intercept_bias',1.,1.)]:
                for power_name in ['unit','orthogonal_optimal']:
                    spec=frame[(frame.tau==tau)&(frame.prediction==name)&(frame.power_name==power_name)].iloc[0]
                    lam=spec.power;boundary=spec.critical_ratio;assert lam>0
                    dims=sorted({max(1,int(round(m*(boundary+offset)))) for offset in [-.15,-.075,0.,.075,.15]})
                    counts={q:{'feasible':0,'outside':0,'unresolved':0} for q in dims}
                    for rep in range(300):
                        s=rng.normal(size=m+N);y=(rng.random(m+N)<expit(1.5*s)).astype(float)
                        z=np.logical_xor(y,rng.random(m+N)<.25).astype(float);orth=rng.normal(size=(m+N,max(dims)))
                        for q in dims:counts[q][feasible(s,y,z,orth[:,:q],m,1.5,.25,lam,a,b)]+=1
                    for q,count in counts.items():rows.append({'labels':m,'auxiliary_rows':N,'tau':tau,'prediction':name,'a':a,'b':b,'power_name':power_name,'power':lam,'orthogonal_dimensions':q,'orthogonal_dimension_label_ratio':q/m,'critical_ratio':boundary,'replications':300,**count,'finite_frequency':count['feasible']/300})
                    pd.DataFrame(rows).to_csv(args.out/'imperfect_prediction_phase.csv',index=False)
                    save(args.out/'imperfect_prediction_status.json',{'completed_cells':len(rows),'requested_cells':80,'replications_per_cell':300,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_priority_verified':False})
                    print('Imperfect-prediction phase cells complete',len(rows),'of80',flush=True)
    print('All imperfect-prediction phase cells complete.',flush=True)


if __name__=='__main__':main()
