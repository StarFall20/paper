"""Independent finite-cone projection and all-positive-power recession checks."""
from pathlib import Path
import argparse
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import brentq,linprog,minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits
from vanishing_power_choice import rescaled_support
from power_tuned_choice_audit import objective,support
from public_conjoint_optimizer_audit import save


def limiting_projection(r,a,b,V,H):
    m=len(r);n=len(H);N=n-m
    def empirical(eta,t):
        w=H-V@eta;q=np.r_[np.maximum(w[:m],t*r),np.clip(w[m:],t*a,t*b)]
        value=float(2*w@q-q@q);gradient=-2*V.T@q
        dv=2*float(r@np.minimum(w[:m]-t*r,0))
        dv+=2*float(np.sum(np.where(w[m:]<t*a,a*(w[m:]-t*a),np.where(w[m:]>t*b,b*(w[m:]-t*b),0))))
        return value,gradient,dv,q
    def maximized(eta):
        val,grad,dv,_=empirical(eta,0.)
        if dv<=0:return val,grad,0.
        hi=1.
        while empirical(eta,hi)[2]>=0:
            hi*=2
            if hi>1e9:raise ArithmeticError('Empirical limiting scale not bracketed')
        t=brentq(lambda v:empirical(eta,v)[2],0,hi,xtol=1e-12)
        val,grad,_,_=empirical(eta,t);return val,grad,t
    dual=minimize(lambda eta:maximized(eta)[:2],np.zeros(V.shape[1]),jac=True,method='BFGS',options={'gtol':1e-10,'maxiter':2500})
    value,gradient,t=maximized(dual.x)
    inequality=np.zeros((m+2*N,n+1));inequality[:m,:m]=np.eye(m);inequality[:m,-1]=-r
    inequality[m:m+N,m:n]=np.eye(N);inequality[m:m+N,-1]=-a
    inequality[m+N:,m:n]=-np.eye(N);inequality[m+N:,-1]=b
    equality=np.c_[V.T,np.zeros(V.shape[1])]
    def cost(w):return .5*np.sum((w[:-1]-H)**2),np.r_[w[:-1]-H,0.]
    primal=minimize(cost,np.zeros(n+1),jac=True,method='SLSQP',
        constraints=[{'type':'eq','fun':lambda w:equality@w,'jac':lambda w:equality},
                     {'type':'ineq','fun':lambda w:inequality@w,'jac':lambda w:inequality}],
        bounds=[(None,None)]*n+[(0,None)],options={'ftol':1e-12,'maxiter':2000})
    # Refine the QP active face without using the dual projected point.
    active=abs(inequality@primal.x)<1e-7;face=np.vstack([equality,inequality[active]])
    if primal.x[-1]<1e-7:face=np.vstack([face,np.r_[np.zeros(n),1.]])
    basis=null_space(face)
    candidate=basis@np.linalg.lstsq(basis[:-1],H,rcond=None)[0] if basis.shape[1] else np.zeros(n+1)
    if candidate[-1]>=-1e-8 and max(abs(equality@candidate))<1e-8 and min(inequality@candidate)>-1e-8 and cost(candidate)[0]<=primal.fun+1e-10:
        primal.x=candidate
    projected=primal.x[:-1];norm2=float(projected@projected);gap=abs(norm2-value)
    eqerr=float(max(abs(equality@primal.x)));ineqerr=max(0.,-float(min(inequality@primal.x)))
    orth=abs(float((H-projected)@projected))
    assert eqerr<1e-7 and ineqerr<1e-7 and primal.x[-1]>=-1e-8
    assert gap<1e-6 and orth<1e-6,(primal.message,dual.message,gap,orth,gradient)
    # A feasible primal point and its independent dual upper bound certify the
    # numerical value even if a solver ends with a non-success message.
    return {'projection_norm_squared_difference':gap,'constraint_error':max(eqerr,ineqerr),
            'Moreau_orthogonality_error':orth,'dual_gradient_max':float(max(abs(gradient))),
            'primal_solver_success':bool(primal.success),'dual_solver_success':bool(dual.success),'zero_scale':bool(abs(primal.x[-1])<1e-8)}


def weak_separator_certificate(x,y,g,m):
    n,p=x.shape;N=n-m;XL=x[:m];XU=x[m:];signed=(2*y[:m]-1)[:,None]*XL
    linear=XL.T@(g[:m]-y[:m])/m-XU.T@g[m:]/N
    cost=np.r_[linear,np.full(N,1/N),np.zeros(p)]
    aub=np.vstack([np.c_[-signed,np.zeros((m,N+p))],
                   np.c_[XU,-np.eye(N),np.zeros((N,p))],
                   np.c_[np.eye(p),np.zeros((p,N)),-np.eye(p)],
                   np.c_[-np.eye(p),np.zeros((p,N)),-np.eye(p)],
                   np.r_[np.zeros(p+N),np.ones(p)][None,:]])
    rhs=np.r_[np.zeros(m+N+2*p),1.]
    fit=linprog(cost,A_ub=aub,b_ub=rhs,bounds=[(None,None)]*p+[(0,None)]*(N+p),method='highs')
    assert fit.success,fit.message
    d=fit.x[:p];margin=float(min(signed@d));u=x@d
    a=float(np.mean(np.maximum(u[m:],0)-g[m:]*u[m:])+np.mean((g[:m]-y[:m])*u[:m]))
    assert margin>=-1e-8 and a<-1e-8,(margin,a)
    identity=0.
    for lam in [.01,.3,1.]:
        loss=objective(x,y,g,m,lam);recession=float(np.sum(loss.weights*np.maximum(u,0))-loss.moment@d)
        identity=max(identity,abs(recession/lam-a));assert recession<0
    return a,margin,identity


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261128);matches=0;certificates=0;rescues=0;projections=[];types={};max_identity=0.
    cases=[(1.,0.),(2.,0.),(1.,1.),(-1.,0.),(0.,0.)]
    for j in range(120):
        m=16+j%7;N=m*(1+j%3);q=2+j%18;gamma=[0.,1.5,2.][j%3];a,b=cases[j%5];lam=[.05,.3,1.][j%3]
        s=rng.normal(size=m+N);y=(rng.random(m+N)<expit(gamma*s)).astype(float)
        z=np.logical_xor(y,rng.random(m+N)<.25).astype(float)
        logit=a*(gamma*s+(2*z-1)*np.log(3.))+b;g=expit(logit);complement=expit(-logit)
        x=np.c_[np.ones(m+N),s,rng.normal(size=(m+N,q))]
        original=support(objective(x,y,g,m,lam))['status'];rescaled=rescaled_support(x,y,g,m,lam)['status']
        assert original==rescaled,(j,original,rescaled);matches+=1
        limit=rescaled_support(x,y,g,m);status=limit['status'];types[status]=types.get(status,0)+1
        if status=='interior':
            power=limit['certified_positive_power'];assert support(objective(x,y,g,m,power))['status']=='interior';rescues+=1
        elif status=='outside_mean_support':
            av,margin,error=weak_separator_certificate(x,y,g,m);certificates+=1;max_identity=max(max_identity,error)
        else:raise ArithmeticError(('Unresolved limiting check',j,limit))
        if j<24:
            signs=np.r_[1-2*y[:m],2*z[m:]-1];V=np.c_[signs,signs*s]
            r=expit((1-2*y[:m])*logit[:m]);tau=N/m
            lower=np.where(z[m:]==1,-g[m:],-complement[m:])/tau
            upper=np.where(z[m:]==1,complement[m:],g[m:])/tau
            projections.append(limiting_projection(r,lower,upper,V,rng.normal(size=m+N)))
    report={'original_rescaled_LP_matches':matches,'limiting_support_statuses':types,
        'independent_weak_separator_negative_cost_certificates':certificates,
        'independent_original_moment_positive_power_witnesses':rescues,
        'maximum_all_power_recession_identity_error':max_identity,'independent_limiting_projection_cases':len(projections),
        'maximum_projection_norm_squared_difference':max(p['projection_norm_squared_difference'] for p in projections),
        'maximum_projection_constraint_error':max(p['constraint_error'] for p in projections),
        'maximum_projection_orthogonality_error':max(p['Moreau_orthogonality_error'] for p in projections),
        'projection_solver_statuses':[{'primal_success':p['primal_solver_success'],'dual_success':p['dual_solver_success'],'zero_scale':p['zero_scale']} for p in projections],
        'priority_or_inferential_guarantee_proven':False}
    save(args.out/'general_joint_power_geometry_checks.json',report);print(report)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
