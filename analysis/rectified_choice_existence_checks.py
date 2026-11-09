"""Direct-draw checks of a candidate rectified-logit existence formula.

Simulates mathematical Gaussian benchmarks only. It generates no LLM answers
or new questionnaire data and makes no originality claim.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.integrate import quad,quad_vec
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm,binom,chi2
from threadpoolctl import threadpool_limits
from public_conjoint_optimizer_audit import save


def radius(t):
    a=np.asarray(t,float);out=np.zeros_like(a);inside=(a>0)&(a<1)
    out[inside]=norm.pdf(norm.ppf(a[inside]));return out


def exact_finite(m,p,q):
    k=np.arange(1,m);v=k*(1-q)**2+(m-k)*q*q
    argument=m*m*radius(k/m)**2/v
    return float(np.sum(binom.pmf(k,m,.5)*chi2.cdf(argument,p)))


def moments(gamma):
    density=lambda x:np.exp(-x*x/2)/np.sqrt(2*np.pi)
    a=quad(lambda x:x*expit(gamma*x)*density(x),-12,12,epsabs=1e-12)[0]
    b=quad(lambda x:expit(gamma*x)*expit(-gamma*x)*density(x),-12,12,epsabs=1e-12)[0]
    hard=2*quad(lambda x:expit(-gamma*x)*density(x),0,12,epsabs=1e-12)[0]
    assert abs(a-gamma*b)<1e-10
    return a,b,hard


def population_numeric_checks():
    rows=[]
    for target_mean in [.3,.5,.7]:
        rad=float(radius(target_mean));cut=norm.ppf(1-target_mean)
        for factor in [0.,.5,.95,1.05,1.5]:
            s=factor*rad
            def objective(theta):
                alpha,r=theta
                def integrand(g):
                    u=alpha+r*g;p=expit(u);w=p*(1-p);density=norm.pdf(g)
                    return density*np.array([np.logaddexp(0,u),p,p*g,w,w*g,w*g*g])
                value,_=quad_vec(integrand,-12,12,epsabs=1e-10,epsrel=1e-10)
                loss=value[0]-target_mean*alpha-s*r
                gradient=value[1:3]-[target_mean,s]
                return float(loss),gradient,np.array([[value[3],value[4]],[value[4],value[5]]])
            if factor<1:
                opt=minimize(lambda t:objective(t)[:2],[np.log(target_mean/(1-target_mean)),0.],jac=True,
                    method='BFGS',options={'gtol':1e-9,'maxiter':200})
                theta=opt.x.copy()
                for _ in range(5):
                    loss,g,h=objective(theta)
                    if max(abs(g))<1e-9:break
                    theta-=np.linalg.solve(h,g)
                loss,g,h=objective(theta);assert max(abs(g))<1e-7
                rows.append({'target_mean':target_mean,'radius_factor':factor,'case':'interior',
                    'finite_alpha':theta[0],'finite_slope_norm':theta[1],'moment_gradient_max':max(abs(g)),
                    'ray_slope_theory':None,'ray_slope_numeric':None})
            else:
                values=[objective(np.array([-cut,1.])*t)[0] for t in [10.,100.,1000.]]
                slope=(values[2]-values[1])/900
                expected=rad-s;assert abs(slope-expected)<1e-4 and values[2]<values[1]<values[0]
                rows.append({'target_mean':target_mean,'radius_factor':factor,'case':'exterior',
                    'finite_alpha':None,'finite_slope_norm':None,'moment_gradient_max':None,
                    'ray_slope_theory':expected,'ray_slope_numeric':slope})
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--replications',type=int,default=2000);args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261109);rep=args.replications
    ratios=[.025,.05,.1,.15,.2,.3,.4,.5,.6,.7,.8];qs=[0.,.25,.5,.75,1.]
    records=[];phases=[];constants=[]
    for m in [20,60,120]:
        dimensions=sorted({max(1,int(round(r*m))) for r in ratios});pmax=max(dimensions)
        counts={(gamma,name,p):0 for gamma in [0.,1.,3.] for name in ['constant_half','constant_one','oracle_probability','hard_threshold','reversed_probability'] for p in dimensions}
        null_counts={(q,p):0 for q in qs for p in dimensions}
        for start in range(0,rep,100):
            size=min(100,rep-start);x=rng.normal(size=(size,m,pmax));uniform=rng.random((size,m))
            for gamma in [0.,1.,3.]:
                a,b,hard=moments(gamma);prob=expit(gamma*x[:,:,0]);y=(uniform<prob).astype(float)
                predictors={'constant_half':(np.full_like(y,.5),.5,0.,.25),
                    'constant_one':(np.ones_like(y),1.,0.,.5),
                    'oracle_probability':(prob,.5,a,b),
                    'hard_threshold':((x[:,:,0]>0).astype(float),.5,norm.pdf(0),hard),
                    'reversed_probability':(1-prob,.5,-a,1-3*b)}
                for name,(g,eg,exg,risk) in predictors.items():
                    residual=y-g;t=eg+residual.mean(1)
                    s=np.mean(x*residual[:,:,None],axis=1);s[:,0]+=exg
                    cumulative=np.cumsum(s*s,axis=1);r2=radius(t)**2;valid=(t>0)&(t<1)
                    for p in dimensions:counts[(gamma,name,p)]+=int(np.sum(valid & (cumulative[:,p-1]<r2)))
                if gamma==0:
                    for q in qs:
                        t=y.mean(1);s=np.mean(x*(y-q)[:,:,None],axis=1)
                        cumulative=np.cumsum(s*s,axis=1);r2=radius(t)**2;valid=(t>0)&(t<1)
                        for p in dimensions:null_counts[(q,p)]+=int(np.sum(valid & (cumulative[:,p-1]<r2)))
        for (q,p),count in null_counts.items():
            truth=exact_finite(m,p,q);estimate=count/rep
            records.append({'labels':m,'nonconstant_dimensions':p,'dimension_label_ratio':p/m,'q':q,
                'replications':rep,'finite_count':count,'finite_frequency':estimate,'exact_finite_probability':truth,
                'Monte_Carlo_SE_at_exact':np.sqrt(truth*(1-truth)/rep),'absolute_discrepancy':abs(estimate-truth),
                'classical_MLE_finite_probability':float(binom.sf(p,m-1,.5))})
        for (gamma,name,p),count in counts.items():
            a,b,hard=moments(gamma);risk={'constant_half':.25,'constant_one':.5,'oracle_probability':b,
                                        'hard_threshold':hard,'reversed_probability':1-3*b}[name]
            phases.append({'labels':m,'nonconstant_dimensions':p,'dimension_label_ratio':p/m,'signal':gamma,'predictor':name,
                'replications':rep,'finite_count':count,'finite_frequency':count/rep,'Brier_risk':risk,
                'critical_ratio_candidate':(norm.pdf(0)**2-a*a)/risk})
        print('direct Gaussian checks complete for labelled count',m,flush=True)
    exact=pd.DataFrame(records);phase=pd.DataFrame(phases)
    exact.to_csv(args.out/'exact_existence_Monte_Carlo.csv',index=False)
    phase.to_csv(args.out/'candidate_existence_phase.csv',index=False)
    numeric=population_numeric_checks();numeric.to_csv(args.out/'population_moment_numeric_checks.csv',index=False)
    for m in [20,60,120]:
        for p in exact[exact.labels.eq(m)].nonconstant_dimensions.unique():
            assert abs(exact_finite(m,int(p),0)-exact_finite(m,int(p),1))<1e-12
            assert abs(exact_finite(m,int(p),.25)-exact_finite(m,int(p),.75))<1e-12
    save(args.out/'candidate_theory_checks.json',{'exact_formula_cells':len(exact),'phase_cells':len(phase),
        'replications_per_cell':rep,'direct_original_X_Y_simulation':True,
        'maximum_exact_formula_MC_discrepancy':float(exact.absolute_discrepancy.max()),
        'population_interior_checks':int(numeric.case.eq('interior').sum()),
        'population_exterior_ray_checks':int(numeric.case.eq('exterior').sum()),
        'minimum_classical_support_claim_new':False,'independent_novelty_verified':False,
        'scope':'Gaussian benchmark and frozen low-index predictors; no universal real-DCE sample-size rule'})
    print('Candidate numerical checks completed; originality remains unverified.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
