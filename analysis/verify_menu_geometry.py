"""Independent constrained-cone QPs and finite-phase result reconciliation."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.optimize import brentq,minimize,least_squares,LinearConstraint
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from menu_corrected_logit import simplex_projection,save,MenuLoss,menu_features


def projection_check(rng,J,power,case):
    m=7+case%3;N=m*(1+case%2);n=m+N;tau=N/m
    S=rng.normal(size=(n,J));S-=S.mean(1,keepdims=True)
    mu=softmax(1.5*S,axis=1);Y=np.sum(rng.random(n)[:,None]>np.cumsum(mu,axis=1),axis=1)
    Z=(Y+(rng.random(n)<.25)*rng.integers(1,J,size=n))%J
    g=softmax(1.5*S+np.eye(J)[Z]*np.log(.75/(.25/(J-1))),axis=1)
    a=np.vstack([power*g[:m]-np.eye(J)[Y[:m]],-power*g[m:]/tau])
    mass=np.r_[np.full(m,1-power),np.full(N,power/tau)]
    H=rng.normal(size=(n,J));H-=H.mean(1,keepdims=True)
    B=helmert(J,full=False).T
    V=np.concatenate([np.broadcast_to(B,(n,J,J-1)),S[:,:,None]],axis=2).reshape(n*J,J)

    def project(w,t):
        # Separate masses keep this implementation distinct from the population helper.
        p=np.empty_like(w)
        for i in range(n):p[i]=simplex_projection(w[i]-t*a[i],float(t*mass[i]))+t*a[i]
        return p

    def scale(eta):
        w=H-(V@eta).reshape(n,J)
        d0=2*float(np.sum(mass*np.max(w,axis=1)+np.sum(a*w,axis=1)))
        if d0<=0:return 0.,np.zeros(J),np.zeros_like(w),0.
        def derivative(t):
            if not t:return d0
            p=project(w,t);return 2*float(np.sum((w-p)*p))/t
        upper=1.
        while derivative(upper)>0:
            upper*=2
            if upper>1e6:raise RuntimeError('Finite cone scale not bracketed')
        t=brentq(derivative,0.,upper,xtol=1e-12);p=project(w,t)
        return float(np.sum(w*w)-np.sum((w-p)**2)),-2*V.T@p.ravel(),p,t

    fit=minimize(lambda eta:scale(eta)[:2],np.zeros(J),jac=True,method='BFGS',options={'gtol':1e-10,'maxiter':1500})
    eta=fit.x
    if np.max(abs(scale(eta)[1]))>1e-8:
        polished=least_squares(lambda e:scale(e)[1],eta,gtol=1e-13,ftol=1e-13,xtol=1e-13,max_nfev=500)
        eta=polished.x
    value,gradient,p,t=scale(eta)
    size=n*J;eq=[]
    if power==1:
        for i in range(m):
            for j in range(J):
                row=np.zeros(size+1);row[i*J+j]=1.;row[-1]=-a[i,j];eq.append(row)
        menus=range(m,n)
    else:menus=range(n)
    for i in menus:
        row=np.zeros(size+1);row[i*J:(i+1)*J]=1.;eq.append(row)
    for col in range(J):eq.append(np.r_[V[:,col],0.])
    eq=np.array(eq);indices=np.arange(m*J,size) if power==1 else np.arange(size)
    inequality=np.c_[np.eye(size)[indices],-a.ravel()[indices]]
    def distance(z):return float(np.sum((z[:-1]-H.ravel())**2)/2),np.r_[z[:-1]-H.ravel(),0.]
    qp=minimize(distance,np.zeros(size+1),jac=True,method='SLSQP',
                constraints=[LinearConstraint(eq,np.zeros(len(eq)),np.zeros(len(eq))),LinearConstraint(inequality,0,np.inf)],
                bounds=[(None,None)]*size+[(0,None)],options={'ftol':1e-13,'maxiter':2500})
    u=qp.x[:-1].reshape(n,J)
    checks=dict(case=case,alternatives=J,power=power,menus=n,
                saddle_vs_QP_projection_error=float(np.max(abs(p-u))),
                squared_norm_difference=abs(float(np.sum(p*p)-np.sum(u*u))),
                saddle_value_vs_projection_norm=abs(value-float(np.sum(p*p))),
                nuisance_residual=float(np.max(abs(V.T@p.ravel()))),
                cone_constraint_violation=max(0.,float(np.max(t*a-p))),
                cone_scale=t,orthogonality_error=abs(float(np.sum(p*(H-p)))),
                QP_constraint_error=max(float(np.max(abs(eq@qp.x))),max(0.,float(-np.min(inequality@qp.x)))),
                QP_reported_success=bool(qp.success),saddle_reported_success=bool(fit.success))
    return checks


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(a.out/'menu_phase_trials.csv');shown=pd.read_csv(a.out/'menu_phase_summary.csv')
    status=json.loads((a.out/'menu_phase_status.json').read_text())
    assert status['state']=='complete' and len(raw)==6000 and len(shown)==12
    assert not raw.duplicated(['label_menus','trial','power','orthogonal_slopes']).any()
    assert raw.groupby(['label_menus','trial']).size().eq(6).all()
    assert raw.independent_residual_status.notna().sum()==120
    max_frequency=max_se=0.;nesting=0
    keys=['label_menus','auxiliary_menus','power','orthogonal_slopes','orthogonal_dimension_ratio','critical_ratio']
    for key,part in raw.groupby(keys):
        row=shown[(shown.label_menus==key[0])&(shown.power==key[2])&(shown.orthogonal_slopes==key[3])].iloc[0]
        assert len(part)==row.replications==500
        for state,column in [('interior','finite_fits'),('outside','outside'),('boundary_or_near_boundary','boundary'),('rank_deficient','rank_deficient'),('unresolved','unresolved')]:assert part.status.eq(state).sum()==row[column]
        frequency=float(part.status.eq('interior').mean());se=np.sqrt(frequency*(1-frequency)/500)
        max_frequency=max(max_frequency,abs(frequency-row.finite_frequency));max_se=max(max_se,abs(se-row.binomial_MCSE))
    for key,part in raw.groupby(['label_menus','trial','power']):
        events=part.sort_values('orthogonal_slopes').status.eq('interior').to_numpy(int)
        assert np.all(np.diff(events)<=0);nesting+=1
    assert max(max_frequency,max_se)<1e-12 and not raw.status.eq('unresolved').any()
    assert raw.loc[raw.status.eq('interior'),'interior_margin'].min()>1e-8
    assert raw.loc[raw.status.eq('outside'),'separation_gap'].min()>1e-8
    rng=np.random.default_rng(2026111161)
    projections=[projection_check(rng,J,power,i) for i,(J,power) in enumerate((J,p) for J in [2,3,4] for p in [1.,.5,.1,1.])]
    save(a.out/'menu_constrained_projection_checks.json',projections)
    maximum=max(max(r[k] for k in ['squared_norm_difference','saddle_value_vs_projection_norm','nuisance_residual','cone_constraint_violation','orthogonality_error','QP_constraint_error']) for r in projections)
    assert maximum<1e-6,projections
    result=dict(all_phase_rows_reconciled=len(raw),aggregate_cells_reconciled=len(shown),
                maximum_frequency_error=max_frequency,maximum_MCSE_error=max_se,
                samplewise_dimension_nesting_cases=nesting,independent_full_constrained_cone_QPs=len(projections),
                maximum_QP_norm_or_certificate_error=maximum,
                maximum_QP_point_difference=max(r['saddle_vs_QP_projection_error'] for r in projections),
                zero_projection_cases=int(sum(r['cone_scale']<1e-8 for r in projections)),
                minimum_interior_margin=float(raw.loc[raw.status.eq('interior'),'interior_margin'].min()),
                minimum_exterior_gap=float(raw.loc[raw.status.eq('outside'),'separation_gap'].min()),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Independent algebra,projection and phase-record checks; does not establish originality or publication readiness')
    save(a.out/'menu_independent_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
