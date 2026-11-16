"""Classical parametric-support LPs for the specified corrected-logit power domain.

The closed feasible endpoint alone does not certify any finite root. A separate
strict witness,rank and endpoint checks are required. This is no tuning rule.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.special import expit
from threadpoolctl import threadpool_limits
from power_tuned_choice_audit import objective,support
from vanishing_power_choice import rescaled_support
from public_conjoint_optimizer_audit import save

TOL=1e-8


def recession_costs(x,y,g,m,d):
    u=x@d
    h=float(np.mean(np.maximum(u[:m],0)-y[:m]*u[:m]))
    a=float(np.mean(np.maximum(u[m:],0)-g[m:]*u[m:])+np.mean((g[:m]-y[:m])*u[:m]))
    return h,a


def weak_obstruction(x,y,g,m):
    """Nonzero weak-separator LP,including zero-cost boundary obstructions."""
    n,p=x.shape;N=n-m
    signed=(2*y[:m]-1)[:,None]*x[:m]
    linear=x[:m].T@(g[:m]-y[:m])/m-x[m:].T@g[m:]/N
    cost=np.r_[linear,np.full(N,1/N)]
    aub=sparse.vstack([sparse.hstack([-sparse.csr_matrix(signed),sparse.csr_matrix((m,N))]),
                       sparse.hstack([sparse.csr_matrix(x[m:]),-sparse.eye(N)])],format='csr')
    for coordinate in range(p):
      for sign in [-1.,1.]:
        bounds=[(-1.,1.)]*p+[(0.,None)]*N;bounds[coordinate]=(sign,sign)
        fit=linprog(cost,A_ub=aub,b_ub=np.zeros(n),bounds=bounds,method='highs')
        if fit.status==2:continue
        if not fit.success:continue
        d=fit.x[:p];h,a=recession_costs(x,y,g,m,d);margin=float(min(signed@d))
        if margin>=-1e-8 and h<=1e-8 and a<=1e-8:
            return {'direction':d.tolist(),'human_recession':h,'unit_recession':a,
                    'minimum_human_margin':margin,'direction_max_abs':float(max(abs(d))),
                    'kind':'negative_cost' if a<-TOL else 'boundary_or_numerical_proximity'}
    return None


def power_domain(x,y,g,m):
    x=np.asarray(x,float);y=np.asarray(y,float);g=np.asarray(g,float)
    n,p=x.shape;N=n-m
    assert m>0 and N>0 and len(y)==len(g)==n and np.isfinite(x).all()
    assert set(np.unique(y[:m]))<={0.,1.} and np.isfinite(g).all() and min(g)>=0 and max(g)<=1
    rank=int(np.linalg.matrix_rank(x))
    if rank<p:return {'state':'rank_deficient','stacked_rank':rank,'dimensions':p}
    scale=np.maximum(abs(x).max(axis=0),1e-12);xx=x/scale;tau=N/m
    b0=xx[:m].T@y[:m];b1=-xx[:m].T@g[:m]+xx[m:].T@g[m:]/tau
    eye=sparse.eye(n,format='csr');ones=sparse.csr_matrix(np.ones((n,1)))
    coefficient=sparse.csr_matrix(np.r_[np.ones(m),-np.ones(N)/tau][:,None])
    rhs=np.r_[np.ones(m),np.zeros(N)]
    equality=sparse.hstack([sparse.csr_matrix(xx.T),sparse.csr_matrix(-b1[:,None])],format='csr')
    endpoint=linprog(np.r_[np.zeros(n),-1.],A_ub=sparse.hstack([eye,coefficient]),b_ub=rhs,
                     A_eq=equality,b_eq=b0,bounds=[(0.,None)]*n+[(0.,1.)],method='highs')
    if not endpoint.success:return {'state':'solver_unresolved','stage':'closed_endpoint','message':endpoint.message}
    endpoint_u=endpoint.x[:n];upper=float(endpoint.x[-1])
    endpoint_residual=float(max(abs(xx.T@endpoint_u-b0-upper*b1)))
    endpoint_violation=max(0.,float(max(endpoint_u+np.r_[np.ones(m),-np.ones(N)/tau]*upper-rhs)),-float(min(endpoint_u)))
    lower=sparse.hstack([-eye,sparse.csr_matrix((n,1)),ones],format='csr')
    upper_constraints=sparse.hstack([eye,coefficient,ones],format='csr')
    witness=linprog(np.r_[np.zeros(n+1),-1.],A_ub=sparse.vstack([lower,upper_constraints]),b_ub=np.r_[np.zeros(n),rhs],
                    A_eq=sparse.hstack([equality,sparse.csr_matrix((p,1))]),b_eq=b0,
                    bounds=[(0.,None)]*n+[(0.,1.),(0.,.5)],method='highs')
    if not witness.success:return {'state':'solver_unresolved','stage':'interior_witness','message':witness.message}
    u=witness.x[:n];power=float(witness.x[n]);t=float(witness.x[-1])
    witness_residual=float(max(abs(xx.T@u-b0-power*b1)))
    gaps=np.r_[u,np.r_[np.full(m,1-power),np.full(N,power/tau)]-u]
    true_margin=float(min(gaps));violation=max(0.,t-true_margin)
    base={'closed_endpoint':upper,'interior_witness_power':power,'interior_witness_margin':t,
          'primal_endpoint_equality_residual':endpoint_residual,'primal_endpoint_bound_violation':endpoint_violation,
          'primal_witness_equality_residual':witness_residual,'primal_witness_margin_violation':violation,
          'stacked_rank':rank,'dimensions':p,'auxiliary_rank':int(np.linalg.matrix_rank(x[m:]))}
    if max(endpoint_residual,endpoint_violation,witness_residual,violation)>1e-7:
        return {'state':'solver_unresolved',**base,'stage':'primal_residual'}
    unit=support(objective(x,y,g,m,1.))
    base['unit_status']=unit['status']
    if unit['status'] not in ['interior','outside_mean_support','boundary_or_near_boundary','rank_deficient']:
        return {**base,'state':'solver_unresolved','stage':'unit_endpoint'}
    if t<=TOL:
        obstruction=weak_obstruction(x,y,g,m)
        return {'state':'no_verified_interior_power' if obstruction is None else 'weak_separator_obstruction',
                **base,'obstruction':obstruction}
    if not 0<power<1 or power>upper+TOL:return {'state':'solver_unresolved',**base,'stage':'strict_witness_power'}
    result={'state':'positive_interval',**base,'unit_endpoint_included':unit['status']=='interior'}
    if TOL<upper<1-TOL:
        d=np.asarray(endpoint.eqlin.marginals)/scale
        h,a=recession_costs(x,y,g,m,d)
        if h<=0 or h-a<=0:return {**result,'state':'solver_unresolved','stage':'endpoint_dual_sign'}
        ratio=h/(h-a);dualerr=max(abs(m*h-upper),abs(m*(a-h)+1),abs(ratio-upper))
        if dualerr>1e-7:return {**result,'state':'solver_unresolved','stage':'endpoint_dual_identity','dual_error':dualerr}
        result.update(endpoint_direction=d.tolist(),endpoint_human_recession=h,endpoint_unit_recession=a,
                      independent_recession_ratio=ratio,endpoint_dual_identity_error=dualerr)
    return result


def numeric_checks(out):
    one=np.ones((4,1));analytic=[]
    specs=[('rescued',[1.,1.],.75,.25,1.,True,True),
           ('all_boundary',[1.,1.],.75,.75,1.,False,False),
           ('all_exterior',[1.,1.],.25,.75,0.,False,False),
           ('cutoff_point625',[0.,1.],.1,.9,.625,True,False),
           ('unit_excluded',[0.,1.],.1,.6,1.,True,False)]
    for name,labels,gl,gu,expected,interior,unit in specs:
        yy=np.r_[labels,[0.,0.]];gg=np.array([gl,gl,gu,gu]);result=power_domain(one,yy,gg,2)
        assert abs(result['closed_endpoint']-expected)<1e-9,(name,result)
        assert (result['state']=='positive_interval')==interior,(name,result)
        if interior:assert result['unit_endpoint_included']==unit,(name,result)
        else:assert result['obstruction'] is not None,(name,result)
        analytic.append({'name':name,'state':result['state'],'closed_endpoint':result['closed_endpoint'],'unit_status':result['unit_status']})
    assert power_domain(np.ones((4,2)),np.array([0.,1.,0.,1.]),np.full(4,.5),2)['state']=='rank_deficient'
    rng=np.random.default_rng(20261130);matches=0;ratio_checks=0;obstructions=0;interiors=0;max_dual=0.;max_residual=0.;states={}
    for trial in range(120):
        m=12+trial%9;N=m*(1+trial%2);p=3+trial%14;n=m+N
        signal=rng.normal(size=n);x=np.c_[np.ones(n),signal,rng.normal(size=(n,p-2))]
        y=(rng.random(n)<expit(.9*signal)).astype(float);z=np.logical_xor(y,rng.random(n)<.25)
        slope,bias=[(1.,0.),(2.,0.),(1.,1.),(-1.,0.),(0.,0.)][trial%5]
        g=expit(slope*(.9*signal+(2*z.astype(float)-1)*np.log(3.))+bias)
        result=power_domain(x,y,g,m);state=result['state'];states[state]=states.get(state,0)+1
        assert state in ['positive_interval','weak_separator_obstruction'],(trial,result)
        max_residual=max(max_residual,*[v for k,v in result.items() if 'residual' in k or 'violation' in k])
        if state=='positive_interval':
            interiors+=1;w=result['interior_witness_power']
            assert support(objective(x,y,g,m,w))['status']=='interior'
            if 'endpoint_direction' in result:
                ratio_checks+=1;max_dual=max(max_dual,result['endpoint_dual_identity_error'])
                d=np.array(result['endpoint_direction']);cut=result['closed_endpoint']
                below=.99*cut;above=(1+cut)/2
                assert support(objective(x,y,g,m,below))['status']=='interior'
                actual=objective(x,y,g,m,above);u=x@d
                recession=float(actual.weights@np.maximum(u,0)-actual.moment@d)
                assert recession<-1e-8,(trial,result,recession)
        else:obstructions+=1
        for power in [.01,.1,.3,.5,.9,1.]:
            first=support(objective(x,y,g,m,power))['status'];second=rescaled_support(x,y,g,m,power)['status']
            assert first==second,(trial,power,first,second)
            expected=(state=='positive_interval' and (power<result['closed_endpoint']-TOL or (power==1 and result['unit_endpoint_included'])))
            assert (first=='interior')==expected,(trial,power,result,first)
            matches+=1
    report={'analytic_cases':analytic,'rank_deficiency_checked':True,'random_finite_designs':120,
            'original_rescaled_fixed_power_matches':matches,'positive_interval_cases':interiors,
            'independent_nonzero_weak_separator_obstructions':obstructions,'independent_endpoint_ratio_checks':ratio_checks,
            'maximum_endpoint_dual_identity_error':max_dual,'maximum_primal_residual_or_violation':max_residual,
            'random_case_states':states,'Gaussian_design_required_for_finite_LP':False,'new_tuning_rule_proposed':False,
            'novelty_or_inference_guarantee_proven':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(out/'power_domain_numeric_checks.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):numeric_checks(a.out)
