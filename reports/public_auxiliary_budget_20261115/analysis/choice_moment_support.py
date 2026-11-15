"""Classical convex-support checks for a binary logit estimating equation."""
import numpy as np
from scipy import sparse
from scipy.optimize import linprog, minimize
from public_conjoint_optimizer_audit import Logistic


def support_margin(x, target):
    n,d=x.shape
    if np.linalg.matrix_rank(x)<d:
        return {'status':'rank_deficient','interior_margin':None}
    scale=np.maximum(np.max(abs(x),axis=0),1e-12);xx=x/scale
    # Maximize epsilon subject to epsilon <= p_i <= 1-epsilon and
    # sum_i x_i p_i = sum_i x_i target_i. No target clipping is used.
    identity=sparse.eye(n,format='csr');one=sparse.csr_matrix(np.ones((n,1)))
    aub=sparse.vstack([sparse.hstack([-identity,one]),sparse.hstack([identity,one])],format='csr')
    aeq=sparse.hstack([sparse.csr_matrix(xx.T),sparse.csr_matrix((d,1))],format='csr')
    result=linprog(np.r_[np.zeros(n),-1.],A_ub=aub,b_ub=np.r_[np.zeros(n),np.ones(n)],
        A_eq=aeq,b_eq=xx.T@target,bounds=[(0.,1.)]*n+[(0.,.5)],method='highs')
    if result.status==2:
        return {'status':'outside_mean_support','interior_margin':None}
    if not result.success:
        return {'status':'support_check_unresolved','interior_margin':None,'solver_message':result.message}
    margin=float(result.x[-1])
    residual=float(max(abs(xx.T@(result.x[:-1]-target))))
    if residual>1e-6:
        return {'status':'support_check_unresolved','interior_margin':margin,'equality_residual':residual}
    return {'status':'interior' if margin>1e-8 else 'boundary_or_near_boundary',
            'interior_margin':margin,'equality_residual':residual}


def solve_logit(x,target,check_support=True):
    support=support_margin(x,target) if check_support else {'status':'not_requested','interior_margin':None}
    if check_support and support['status']!='interior':
        return None,support
    loss=Logistic(x,np.asarray(target,float));opt=minimize(loss.value,np.zeros(x.shape[1]),
        jac=True,method='BFGS',options={'gtol':1e-9,'maxiter':1800})
    b=opt.x.copy()
    for _ in range(7):
        val,g=loss.value(b)
        if max(abs(g))<1e-9:break
        try:delta=np.linalg.solve(loss.hessian(b),g)
        except np.linalg.LinAlgError:break
        step=1.
        while step>1e-8 and loss.value(b-step*delta)[0]>val+1e-12:step/=2
        if step<=1e-8:break
        b-=step*delta
    val,g=loss.value(b);eig=np.linalg.eigvalsh(loss.hessian(b))
    support.update(mean_score_max=float(max(abs(g))),minimum_information_eigenvalue=float(min(eig)),
                   optimizer_success=bool(opt.success),objective=val)
    if not np.isfinite(b).all() or max(abs(g))>1e-7 or min(eig)<1e-11:
        support['status']='fit_unresolved'
        return None,support
    support['status']='finite_root'
    return b,support
