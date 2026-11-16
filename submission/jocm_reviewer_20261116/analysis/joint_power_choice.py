"""Closed limiting cone checks and exact replay at jointly vanishing powers.

All cone and random-subspace machinery is classical. This verifies the stated
balanced corrected-loss calculation and adds no independent response samples.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.optimize import brentq,minimize
from threadpoolctl import threadpool_limits
from vanishing_power_choice import positive_limit,rescaled_support
from finite_ratio_choice_boundary import critical_ratio
from power_tuned_choice_audit import objective,support
from public_conjoint_optimizer_audit import save


def limiting_projection(h,r,tau,e):
    m=len(r);a=-(1-e)/tau;b=e/tau
    def point(t):return np.r_[np.maximum(h[:m],t*r),np.clip(h[m:],t*a,t*b)]
    def derivative(t):
        label=2*np.dot(r,np.maximum(t*r-h[:m],0))
        # For lower-active observations h<t*a: derivative=2*a*(t*a-h).
        aux=2*a*np.maximum(t*a-h[m:],0)+2*b*np.minimum(t*b-h[m:],0)
        return float(label+np.sum(aux))
    if derivative(0)>=0:t=0.
    else:
        hi=1.
        while derivative(hi)<0:hi*=2
        t=brentq(derivative,0,hi,xtol=1e-13)
    return point(t),t


def geometry_checks(out):
    rng=np.random.default_rng(20261125)
    max_qp=0.;max_polar=0.;max_orthogonal=0.;zero=0
    for j in range(48):
        m=6+j%7;N=m*(1+j%2);n=m+N;tau=N/m;e=[.25,.5][j%2]
        r=np.where(rng.random(m)<1-e,e,1-e);h=rng.normal(size=n)
        v,t=limiting_projection(h,r,tau,e);a=-(1-e)/tau;b=e/tau
        # Independent convex QP in all n+1 variables, started at the origin.
        matrix=np.zeros((m+2*N,n+1))
        matrix[:m,:m]=np.eye(m);matrix[:m,-1]=-r
        matrix[m:m+N,m:n]=np.eye(N);matrix[m:m+N,-1]=-a
        matrix[m+N:,m:n]=-np.eye(N);matrix[m+N:,-1]=b
        def fun(w):return .5*np.sum((w[:-1]-h)**2),np.r_[w[:-1]-h,0.]
        fit=minimize(fun,np.zeros(n+1),jac=True,method='SLSQP',
            constraints=[{'type':'ineq','fun':lambda w:matrix@w,'jac':lambda w:matrix}],
            bounds=[(None,None)]*n+[(0,None)],options={'ftol':1e-12,'maxiter':500})
        assert fit.success,(j,fit.message)
        max_qp=max(max_qp,float(max(abs(fit.x[:-1]-v))))
        residual=h-v
        polar=float(np.dot(r,residual[:m])+np.maximum(a*residual[m:],b*residual[m:]).sum())
        max_polar=max(max_polar,float(np.max(residual[:m])),polar)
        max_orthogonal=max(max_orthogonal,abs(float(residual@v)))
        zero+=t==0
    # This nonzero projection lies on the t=0 face of the CLOSED cone.
    h=np.array([3.,0.,-1.,0.,0.,0.]);r=np.array([.25,.75,.25])
    v,t=limiting_projection(h,r,1.,.25)
    assert t==0 and np.array_equal(v,np.array([3.,0.,0.,0.,0.,0.]))
    assert max_qp<2e-6 and max_polar<1e-9 and max_orthogonal<1e-8
    result={'independent_QP_projection_checks':48,'maximum_projection_coordinate_difference':max_qp,
        'maximum_polar_inequality_excess':max_polar,'maximum_Moreau_orthogonality_error':max_orthogonal,
        'random_zero_scale_optima':zero,'explicit_nonzero_zero_scale_projection_retained':True,
        'scope':'balanced Gaussian slopes with a fixed zero intercept',
        'error_distribution_or_coverage_theorem':False,'priority_proven':False}
    save(out/'joint_power_geometry_checks.json',result)
    return result


def replay(reference,out):
    old=pd.read_csv(reference);old=old[old.power.eq('limit')]
    assert len(old)==12 and old.replications.eq(500).all() and old.unresolved.sum()==0
    rng=np.random.default_rng(20261123);rows=[];resolved_rows=[];resolutions=[];done=0;matches=0;fits=0
    for m in [40,80]:
      for e in [.25,.5]:
       for ratio in [.45,.6,.7]:
        p=round(m*ratio);counts={name:{} for name in ['inverse_sqrt_m','inverse_m','limit']}
        verified={name:{} for name in counts};outside_violations=0
        for rep in range(500):
            x=rng.normal(size=(2*m,p));y=rng.integers(0,2,2*m).astype(float)
            z=np.logical_xor(y,rng.random(2*m)<e).astype(float);g=e+(1-2*e)*z
            limit=rescaled_support(x,y,g,m);assert limit['status']!='unresolved'
            cases=[('limit',None,limit)]
            for name,lam in [('inverse_sqrt_m',1/np.sqrt(m)),('inverse_m',1/m)]:
                raw=rescaled_support(x,y,g,m,lam);cases.append((name,lam,raw));fits+=1
            for name,lam,raw in cases:
                raw_status=raw['status'];status=raw_status
                counts[name][raw_status]=counts[name].get(raw_status,0)+1
                if raw_status=='unresolved':
                    other=support(objective(x,y,g,m,lam));status=other['status']
                    resolutions.append({'labels':m,'epsilon':e,'dimensions':p,'replication':rep,
                        'power_name':name,'power':lam,'raw':raw,'independent_original_moment':other})
                verified[name][status]=verified[name].get(status,0)+1
                if status=='interior' and limit['status']!='interior':outside_violations+=1
        reference_row=old[(old.labels==m)&(old.epsilon==e)&(old.dimensions==p)].iloc[0]
        assert counts['limit'].get('interior',0)==reference_row.interior
        assert counts['limit'].get('outside_mean_support',0)==reference_row.outside
        assert counts['limit'].get('boundary_or_near_boundary',0)==reference_row.boundary_or_near_boundary
        assert outside_violations==0;matches+=1
        limit_value=positive_limit(1.,e)[0]
        for target,collection in [(rows,counts),(resolved_rows,verified)]:
          for name,c in collection.items():
            lam={'inverse_sqrt_m':1/np.sqrt(m),'inverse_m':1/m,'limit':np.nan}[name]
            target.append({'labels':m,'epsilon':e,'dimensions':p,'dimension_label_ratio':p/m,
                'power_name':name,'power':lam,'replications':500,'interior':c.get('interior',0),
                'outside':c.get('outside_mean_support',0),'boundary_or_near_boundary':c.get('boundary_or_near_boundary',0),
                'unresolved':sum(v for k,v in c.items() if k in ['unresolved','rank_deficient']),
                'finite_frequency':c.get('interior',0)/500,'joint_limit_boundary':limit_value,
                'finite_power_boundary':limit_value if name=='limit' else critical_ratio(1.,e,lam)[0]})
        done+=1
        pd.DataFrame(rows).to_csv(out/'joint_power_phase_raw.csv',index=False)
        pd.DataFrame(resolved_rows).to_csv(out/'joint_power_phase_verified.csv',index=False)
        save(out/'joint_power_resolutions.json',{'cases':resolutions,'raw_outputs_overwritten':False})
        save(out/'joint_power_status.json',{'completed_DGP_blocks':done,'requested_DGP_blocks':12,
            'replications_per_block':500,'original_draws_replayed':500*done,'new_positive_power_fits':fits,
            'original_limit_aggregate_matches':matches,'interior_outside_envelope_violations':outside_violations,
            'raw_unresolved':int(pd.DataFrame(rows).unresolved.sum()),
            'remaining_unresolved':int(pd.DataFrame(resolved_rows).unresolved.sum()),
            'new_independent_response_samples':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'reference_sha256':hashlib.sha256(reference.read_bytes()).hexdigest()})
        print('Joint-power original DGP blocks complete',done,'of12',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--reference',type=Path);parser.add_argument('--checks-only',action='store_true')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        print(json.dumps(geometry_checks(args.out)),flush=True)
        if not args.checks_only:
            assert args.reference is not None;replay(args.reference,args.out)
