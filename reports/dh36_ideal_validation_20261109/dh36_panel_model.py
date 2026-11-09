"""Panel mixed logit: participant-level integrated likelihood, no provenance editing."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import qmc, norm


def draws(power=9, seed=20261109):
    u=qmc.Sobol(2, scramble=True, seed=seed).random_base2(power)
    return norm.ppf(np.clip(u,1e-12,1-1e-12))


def panel_eval(b, y, v, base, normal_draws, interaction=False, scores=False):
    """Return NLL and analytic gradient; optional independent participant scores."""
    alpha=b[:2]; mu=b[2]; gamma=b[3:5]
    sb,sp=np.exp(b[5:7]); z=normal_draws
    ri=sb*z[:,0]; price=-np.exp(mu+sp*z[:,1])
    n=len(y); r=len(z); lp_person=np.zeros(n); score=np.zeros((n,len(b)))
    for group in (0,1):
        use=np.where(v==group)[0]
        if len(use)==0: continue
        a=alpha+(b[7:9]*group if interaction else 0)
        u=np.zeros((r,8,3))
        u[:,:,:2]=a[None,None,:]+ri[:,None,None]
        u+=price[:,None,None]*base[None,:,:,2]
        u+=np.einsum('tjk,k->tj',base[:,:,3:5],gamma)[None,:,:]
        lp=u-logsumexp(u,axis=2,keepdims=True); prob=np.exp(lp)
        chosen=np.eye(3)[y[use]]
        lj=np.einsum('ntj,rtj->nr',chosen,lp,optimize=True)
        lm=logsumexp(lj,axis=1)-np.log(r); lp_person[use]=lm
        post=np.exp(lj-logsumexp(lj,axis=1,keepdims=True))
        observed=chosen.sum(axis=1)[:,:2]
        expected=prob.sum(axis=1)[:,:2]
        for j in (0,1):
            score[use,j]=np.sum(post*(observed[:,j,None]-expected[None,:,j]),axis=1)
        obsx=np.einsum('ntj,tjk->nk',chosen,base[:,:,2:5],optimize=True)
        ex=np.einsum('rtj,tjk->rk',prob,base[:,:,2:5],optimize=True)
        dp=obsx[:,0,None]-ex[None,:,0]
        score[use,2]=np.sum(post*dp*price[None,:],axis=1)
        for k in (1,2):
            score[use,k+2]=np.sum(post*(obsx[:,k,None]-ex[None,:,k]),axis=1)
        ds=observed.sum(axis=1)[:,None]-expected.sum(axis=1)[None,:]
        score[use,5]=np.sum(post*ds*ri[None,:],axis=1)
        score[use,6]=np.sum(post*dp*price[None,:]*sp*z[None,:,1],axis=1)
        if interaction: score[use,7:9]=score[use,:2]*group
    value=-float(lp_person.sum()); grad=-score.sum(axis=0)
    if scores: return value,grad,score,lp_person
    return value,grad


def fit_panel(y,v,base,power=9,seed=20261109,interaction=False,start=None,multistart=True):
    z=draws(power,seed); k=9 if interaction else 7
    initial=np.array([0.,-.4,np.log(.22),-.2,.1,np.log(.8),np.log(.4)]+([0.,0.] if interaction else []))
    starts=[initial] if start is None else [np.array(start)]
    if multistart:
        a=starts[0].copy(); a[5:7]=[np.log(.3),np.log(.9)]; starts.append(a)
        a=starts[0].copy(); a[5:7]=[np.log(1.6),np.log(.15)]; starts.append(a)
    bounds=[(-8,8),(-8,8),(-5,1.5),(-5,5),(-5,5),(-4,2),(-4,1)]+([(-8,8),(-8,8)] if interaction else [])
    opts=[]
    for b0 in starts:
        opts.append(minimize(panel_eval,b0,args=(y,v,base,z,interaction),jac=True,method='L-BFGS-B',bounds=bounds,options={'ftol':1e-13,'gtol':1e-6,'maxiter':800,'maxls':40}))
    opt=min(opts,key=lambda x:x.fun)
    value,g,scores,lpi=panel_eval(opt.x,y,v,base,z,interaction,True)
    projected=g.copy()
    for j,(lo,hi) in enumerate(bounds):
        if opt.x[j]<=lo+1e-6 and g[j]>0: projected[j]=0
        if opt.x[j]>=hi-1e-6 and g[j]<0: projected[j]=0
    eps=2e-5; eye=np.eye(k)
    h=np.column_stack([(panel_eval(opt.x+eps*eye[j],y,v,base,z,interaction)[1]-panel_eval(opt.x-eps*eye[j],y,v,base,z,interaction)[1])/(2*eps) for j in range(k)])
    h=(h+h.T)/2; eig=np.linalg.eigvalsh(h)
    hinv=np.linalg.pinv(h); cov=hinv@scores.T@scores@hinv*len(y)/(len(y)-1)
    bcheck=initial*.9
    fd=np.array([(panel_eval(bcheck+eps*eye[j],y,v,base,z,interaction)[0]-panel_eval(bcheck-eps*eye[j],y,v,base,z,interaction)[0])/(2*eps) for j in range(k)])
    error=float(np.max(abs(fd-panel_eval(bcheck,y,v,base,z,interaction)[1])))
    names=['ASC_A','ASC_B','mu_log_negative_price','G1_vs_G0','G2_vs_G0','log_sd_service_intercept','log_sd_log_price']+(['V2_x_ASC_A','V2_x_ASC_B'] if interaction else [])
    return {'beta':opt.x,'cov':cov,'loglik':-value,'participant_loglik':lpi,
            'parameters':[{'term':nm,'estimate':float(b),'cluster_se':float(np.sqrt(max(0,cov[j,j])))} for j,(nm,b) in enumerate(zip(names,opt.x))],
            'validation':{'sobol_draws':len(z),'seed':seed,'solver_success':bool(opt.success),'projected_gradient_max':float(max(abs(projected))),'gradient_check_error':error,'hessian_min_eigenvalue':float(eig.min()),'hessian_max_eigenvalue':float(eig.max()),'start_logliks':[-float(o.fun) for o in opts],'at_bound':[names[j] for j,(lo,hi) in enumerate(bounds) if min(abs(opt.x[j]-lo),abs(opt.x[j]-hi))<1e-5]}}


def fixed_price_eval(b,y,v,base,nodes=61,interaction=False,scores=False):
    """Random service intercept, fixed price: Gauss-Hermite panel likelihood."""
    gh,weights=np.polynomial.hermite.hermgauss(nodes); zz=np.sqrt(2)*gh
    lw=np.log(weights)-.5*np.log(np.pi); sd=np.exp(b[5]); ri=sd*zz
    lpi=np.zeros(len(y)); ss=np.zeros((len(y),len(b)))
    for group in (0,1):
        use=np.where(v==group)[0]
        if len(use)==0: continue
        a=b[:2]+(b[6:8]*group if interaction else 0)
        u=np.zeros((nodes,8,3)); u[:,:,:2]=a[None,None,:]+ri[:,None,None]
        u+=np.einsum('tjk,k->tj',base[:,:,2:5],b[2:5])[None,:,:]
        lp=u-logsumexp(u,axis=2,keepdims=True); prob=np.exp(lp)
        obs=np.eye(3)[y[use]]; lj=np.einsum('ntj,rtj->nr',obs,lp,optimize=True)+lw
        lpi[use]=logsumexp(lj,axis=1); post=np.exp(lj-lpi[use,None])
        observed=obs.sum(1)[:,:2]; expected=prob.sum(1)[:,:2]
        for j in (0,1): ss[use,j]=np.sum(post*(observed[:,j,None]-expected[None,:,j]),axis=1)
        obsx=np.einsum('ntj,tjk->nk',obs,base[:,:,2:5],optimize=True)
        ex=np.einsum('rtj,tjk->rk',prob,base[:,:,2:5],optimize=True)
        for j in range(3): ss[use,j+2]=np.sum(post*(obsx[:,j,None]-ex[None,:,j]),axis=1)
        ss[use,5]=np.sum(post*(observed.sum(1)[:,None]-expected.sum(1)[None,:])*ri,axis=1)
        if interaction: ss[use,6:8]=ss[use,:2]*group
    if scores: return -float(lpi.sum()),-ss.sum(0),ss,lpi
    return -float(lpi.sum()),-ss.sum(0)


def fit_fixed_price(y,v,base,nodes=61,interaction=False):
    initial=np.array([0.,-.4,-.22,-.2,.1,np.log(.8)]+([0.,0.] if interaction else [])); k=len(initial)
    starts=[]
    for sd in [.3,.8,1.6]:
        a=initial.copy(); a[5]=np.log(sd); starts.append(a)
    bounds=[(-8,8),(-8,8),(-2,0),(-5,5),(-5,5),(-4,2)]+([(-8,8),(-8,8)] if interaction else [])
    opts=[minimize(fixed_price_eval,b,args=(y,v,base,nodes,interaction),jac=True,method='L-BFGS-B',bounds=bounds,options={'ftol':1e-13,'gtol':1e-7,'maxiter':1000,'maxls':40}) for b in starts]
    opt=min(opts,key=lambda o:o.fun); value,g,ss,lpi=fixed_price_eval(opt.x,y,v,base,nodes,interaction,True)
    eps=2e-5; eye=np.eye(k)
    h=np.column_stack([(fixed_price_eval(opt.x+eps*e,y,v,base,nodes,interaction)[1]-fixed_price_eval(opt.x-eps*e,y,v,base,nodes,interaction)[1])/(2*eps) for e in eye]); h=(h+h.T)/2
    hi=np.linalg.inv(h); cov=len(y)/(len(y)-1)*hi@ss.T@ss@hi
    fd=np.array([(fixed_price_eval(initial+eps*e,y,v,base,nodes,interaction)[0]-fixed_price_eval(initial-eps*e,y,v,base,nodes,interaction)[0])/(2*eps) for e in eye])
    err=float(max(abs(fd-fixed_price_eval(initial,y,v,base,nodes,interaction)[1])))
    names=['ASC_A','ASC_B','price_per_10_CNY','G1_vs_G0','G2_vs_G0','log_sd_service_intercept']+(['V2_x_ASC_A','V2_x_ASC_B'] if interaction else [])
    return {'beta':opt.x,'cov':cov,'loglik':-value,'participant_loglik':lpi,'parameters':[{'term':nm,'estimate':float(b),'cluster_se':float(np.sqrt(cov[j,j])),'ci_low':float(b-1.96*np.sqrt(cov[j,j])),'ci_high':float(b+1.96*np.sqrt(cov[j,j]))} for j,(nm,b) in enumerate(zip(names,opt.x))],
            'validation':{'quadrature_nodes':nodes,'solver_success':bool(opt.success),'gradient_max':float(max(abs(g))),'gradient_check_error':err,'hessian_min_eigenvalue':float(np.linalg.eigvalsh(h).min()),'start_logliks':[-float(o.fun) for o in opts],'fixed_start_rule':[.3,.8,1.6],'at_bound':[names[j] for j,(lo,hi) in enumerate(bounds) if min(abs(opt.x[j]-lo),abs(opt.x[j]-hi))<1e-5]}}
