"""Global finite-support attainment under a regular covariance budget.

Concave objective plus a classical SOC constraint. All solver returns and
small feasibility repairs are retained. No new simulated choices are drawn.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import pandas as pd
from scipy.optimize import brentq,minimize
from scipy.special import ndtr


def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def normal_tail(x):
    x=np.asarray(x);return (1+x*x)*ndtr(-x)-x*np.exp(-x*x/2)/np.sqrt(2*np.pi)


def normal_tail_prime(x):
    x=np.asarray(x);return 2*(x*ndtr(-x)-np.exp(-x*x/2)/np.sqrt(2*np.pi))


def profile(x,e,lam,tau):
    b=(1-lam)/lam
    return 1+tau-(1-e)*(normal_tail(-e*x)+normal_tail((b+e)*x))-e*(normal_tail(-(1-e)*x)+normal_tail((b+1-e)*x))-tau*(normal_tail((1-e)*x/tau)+normal_tail(e*x/tau))


def gradient(x,e,lam,tau):
    b=(1-lam)/lam
    return -(1-e)*(-e*normal_tail_prime(-e*x)+(b+e)*normal_tail_prime((b+e)*x))-e*(-(1-e)*normal_tail_prime(-(1-e)*x)+(b+1-e)*normal_tail_prime((b+1-e)*x))-(1-e)*normal_tail_prime((1-e)*x/tau)-e*normal_tail_prime(e*x/tau)


def radial(shape,p,e,lam,tau):
    f=lambda v:float(p@(shape*gradient(v*shape,e,lam,tau)))
    hi=1.
    while f(hi)>0:hi*=2
    v=brentq(f,0,hi,xtol=1e-13)
    return v*shape,float(p@profile(v*shape,e,lam,tau))


def variance(t,d,p):return float((p@(t*t))/(p@(d*t))**2)


def power_comparator(d,p,c,e,lam,tau):
    cinv=1/float(p@d)**2
    if c>=cinv-1e-12:r=1.
    elif c==1:r=0.
    else:r=brentq(lambda r:variance(d**(1-r),d,p)-c,0,1,xtol=1e-13)
    x,k=radial(d**(1-r),p,e,lam,tau)
    return x,k,r


def optimal(d,p,c,e,lam,tau,power_x):
    norm=lambda x:float(np.sqrt(p@(x*x)))
    cone=lambda x:norm(x)-np.sqrt(c)*float(p@(d*x))
    if c==1:
        x,value=radial(d,p,e,lam,tau);return x,value,dict(method='exact_unweighted_endpoint',computed_upper_gap=0.,cone_residual=cone(x),stationarity_norm=None,multiplier=None),[]
    if c>=1/float(p@d)**2-1e-12:
        x,value=radial(np.ones(len(d)),p,e,lam,tau);return x,value,dict(method='exact_inverse_scale_endpoint',computed_upper_gap=0.,cone_residual=cone(x),stationarity_norm=None,multiplier=0.),[]
    unweighted,_=radial(d,p,e,lam,tau);returns=[];valid=[]
    def objective(x):return -float(p@profile(x,e,lam,tau))
    def jac(x):return -p*gradient(x,e,lam,tau)
    constraint=dict(type='ineq',fun=lambda x:-cone(x),jac=lambda x:p*(np.sqrt(c)*d-x/max(norm(x),1e-30)))
    for name,start in [('unweighted_ray',unweighted),('power_ray',power_x)]:
        fit=minimize(objective,start,jac=jac,method='SLSQP',bounds=[(1e-10,None)]*len(d),constraints=[constraint],options={'ftol':5e-14,'maxiter':2000})
        raw=np.array(fit.x);x=raw.copy();repair=0.;before=cone(x)
        if before>-1e-12:
            repair=min(1.,max(0.,(before+1.1e-12)/(before-cone(unweighted))))
            x=(1-repair)*x+repair*unweighted
        x,value=radial(x,p,e,lam,tau);g=cone(x);h=x/norm(x)-np.sqrt(c)*d
        eta=max(0.,float(p@(gradient(x,e,lam,tau)*h))/(p@(h*h)))
        residual=gradient(x,e,lam,tau)-eta*h;resnorm=float(np.sqrt(p@(residual*residual)))
        uppergap=float(max(0.,-eta*g+(p@(residual*residual))/(2*e*(1-e))))
        quality=bool(g<=1e-11 and min(x)>0 and uppergap<1e-7)
        record=dict(start=name,solver_success=bool(fit.success),solver_status=int(fit.status),solver_message=str(fit.message),iterations=int(fit.nit),raw_x=raw.tolist(),raw_cone_residual=before,feasibility_mixing_fraction=repair,x=x.tolist(),capacity=value,cone_residual=g,stationarity_norm=resnorm,multiplier=eta,computed_upper_gap=uppergap,quality_passed=quality)
        returns.append(record)
        if quality:valid.append((value,x,record))
    if not valid:return returns[0]['x'],returns[0]['capacity'],dict(method='unresolved_primary',**{k:returns[0][k] for k in ['computed_upper_gap','cone_residual','stationarity_norm','multiplier']}),returns
    value,x,best=max(valid,key=lambda r:r[0]);return x,value,dict(method='convex_SLSQP',**{k:best[k] for k in ['computed_upper_gap','cone_residual','stationarity_norm','multiplier']}),returns


def limiting_shape(d,p,c):
    mu=float(p@d);cinv=1/mu**2
    if c>=cinv-1e-12:return np.ones(len(d))
    if c==1:return d.copy()
    sigma=np.sqrt(1-mu*mu);b=np.sqrt(1-1/c)/sigma;a=1/np.sqrt(c)-mu*b
    return a*d+b


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'run_status.json').exists(),'Use a fresh output directory.'
    plan=json.loads(a.plan.read_text());e=plan['epsilon'];rows=[];weights=[];diagnostics=[];targets=[];started=time.monotonic()
    for law in plan['scale_laws']:
        p=np.array(law['probabilities']);d=np.array(law['raw_scales']);d=d/np.sqrt(p@(d*d));mu=float(p@d);cinv=1/mu**2
        budgets=sorted(set(plan['variance_budgets']+([cinv] if abs(cinv-1)>1e-12 else [])))
        for tau in plan['auxiliary_ratios']:
            lam=tau/(1+tau)
            for c in budgets:
                key=f'{law["name"]}_tau{tau:g}_C{c:.12g}'
                power_x,power_k,r=power_comparator(d,p,c,e,lam,tau)
                x,k,check,returns=optimal(d,p,c,e,lam,tau,power_x);x=np.array(x)
                limit_t=limiting_shape(d,p,c);limit_k=float((p@limit_t)**2/(p@(limit_t*limit_t))/(2*np.pi*e*(1-e)))
                _,finite_limit_weight_k=radial(limit_t,p,e,lam,tau)
                for attempt in returns:diagnostics.append(dict(cell_id=key,**attempt))
                weights.append(dict(cell_id=key,model=law['name'],auxiliary_ratio=tau,variance_budget=c,scales=d.tolist(),probabilities=p.tolist(),optimized_x=x.tolist(),optimized_row_weights=(x/d).tolist(),power_x=power_x.tolist(),power_exponent=r,limiting_effective_scale=limit_t.tolist()))
                rows.append(dict(cell_id=key,model=law['name'],scale_levels=len(d),auxiliary_ratio=tau,power=lam,variance_budget=c,inverse_scale_cost=cinv,mean_scale=mu,optimized_capacity=k,power_capacity=power_k,capacity_gain=k-power_k,optimized_variance=variance(x,d,p),power_variance=variance(power_x,d,p),power_exponent=r,limiting_capacity=limit_k,finite_tau_limiting_weight_capacity=finite_limit_weight_k,**check))
        for target in plan['limiting_boundary_targets']:
            u=np.sqrt(2*np.pi*e*(1-e)*target)
            if u>1:cost=None;status='unattainable'
            elif u<=mu:cost=1.;status='unweighted_sufficient_for_boundary_target'
            else:cost=float(1/(mu*u+np.sqrt(1-mu**2)*np.sqrt(1-u*u))**2);status='positive_cost_for_boundary_target'
            targets.append(dict(model=law['name'],target_boundary=target,minimal_regular_covariance_multiplier=cost,status=status))
    frame=pd.DataFrame(rows);frame.to_csv(a.out/'budget_population.csv',index=False);pd.DataFrame(targets).to_csv(a.out/'limiting_cost_targets.csv',index=False);save(a.out/'weight_functions.json',weights);save(a.out/'all_solver_returns.json',diagnostics)
    unknown=int(frame.method.eq('unresolved_primary').sum())
    save(a.out/'run_status.json',dict(complete=True,population_cells=len(rows),solver_returns=len(diagnostics),unresolved_cells=unknown,maximum_computed_upper_gap=float(frame.computed_upper_gap.max()),minimum_same_budget_gain=float(frame.capacity_gain.min()),maximum_same_budget_gain=float(frame.capacity_gain.max()),elapsed_seconds=time.monotonic()-started,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),new_human_or_LLM_responses=0))
    print(json.dumps(dict(population_cells=len(rows),unresolved=unknown,max_gain=float(frame.capacity_gain.max()),max_gap_diagnostic=float(frame.computed_upper_gap.max()))))


if __name__=='__main__':main()
