"""Independent KKT roots, original scalar profile and Gaussian integration.

This does not call the primary SLSQP program or its vector profile functions.
Limiting values are checked using a separate linear objective on a unit ball.
"""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np
import pandas as pd
from scipy.optimize import root,minimize
from radial_compensation_audit import profile,profile_derivative,boundary
from verify_radial_compensation import independent_normal_box


def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def integral(x,p,e,lam,tau):
    total=error=0.
    for value,mass in zip(x,p):
        for residual,probability in [(e,1-e),(1-e,e)]:
            lo=value*residual;hi=lo+value*(1-lam)/lam
            val,err=independent_normal_box(lo,hi);total+=mass*probability*val;error+=mass*probability*err
        val,err=independent_normal_box(-value*(1-e)/tau,value*e/tau)
        total+=tau*mass*val;error+=tau*mass*err
    return total,error


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'verification.json').exists(),'Use a fresh output directory.'
    plan=json.loads(a.plan.read_text());e=plan['epsilon'];frame=pd.read_csv(a.root/'results/budget_population.csv');functions={r['cell_id']:r for r in json.loads((a.root/'results/weight_functions.json').read_text())};status=json.loads((a.root/'results/run_status.json').read_text())
    assert status['population_cells']==len(frame)==195 and status['unresolved_cells']==0
    assert status['plan_sha256']==hashlib.sha256(a.plan.read_bytes()).hexdigest()
    assert status['source_sha256']==hashlib.sha256((Path(__file__).parent/'budgeted_attainment.py').read_bytes()).hexdigest()
    records=[];returns=[];limit_records=[];seen=set()
    for row in frame.itertuples(index=False):
        record=functions[row.cell_id];d=np.array(record['scales']);p=np.array(record['probabilities']);x=np.array(record['optimized_x']);xp=np.array(record['power_x']);c=row.variance_budget;lam=row.power;tau=row.auxiliary_ratio
        assert np.min(x)>0 and abs(p.sum()-1)<1e-12 and abs(p@(d*d)-1)<1e-12
        nrm=lambda z:float(np.sqrt(p@(z*z)))
        cone=lambda z:nrm(z)-math.sqrt(c)*float(p@(d*z))
        values=lambda z:np.array([profile(float(v),e,lam,tau) for v in z])
        gradients=lambda z:np.array([profile_derivative(float(v),e,lam,tau) for v in z])
        value=float(p@values(x));assert abs(value-row.optimized_capacity)<1e-11
        assert cone(x)<=1e-10
        variance=float((p@(x*x))/(p@(d*x))**2);assert variance<=c+1e-9 and abs(variance-row.optimized_variance)<1e-10
        power_value=float(p@values(xp));assert abs(power_value-row.power_capacity)<1e-11
        assert (p@(xp*xp))/(p@(d*xp))**2<=c+1e-9
        scalar_derivative=abs(float(p@(x*gradients(x))));assert scalar_derivative<1e-6
        independent_optimum=value;solution_difference=0.;eta=None;stationarity=None;gap=0.
        if row.method=='convex_SLSQP':
            def equations(z):
                xx=z[:-1];mult=z[-1];h=xx/max(nrm(xx),1e-30)-math.sqrt(c)*d
                return np.r_[np.sqrt(p)*(gradients(xx)-mult*h),cone(xx)]
            z0=np.r_[x*(1+np.linspace(-.01,.01,len(x))),row.multiplier*1.01]
            fit=root(equations,z0,method='hybr',options={'xtol':1e-10})
            residual=float(np.max(abs(equations(fit.x))));xx=fit.x[:-1];eta=float(fit.x[-1])
            returns.append(dict(cell_id=row.cell_id,solver_success=bool(fit.success),solver_message=str(fit.message),equation_residual=residual,x=xx.tolist(),multiplier=eta))
            assert residual<1e-8 and min(xx)>0 and eta>0,(row.cell_id,returns[-1])
            independent_optimum=float(p@values(xx));solution_difference=float(max(abs(xx-x)))
            assert abs(independent_optimum-value)<1e-8 and solution_difference<2e-5,(row.cell_id,independent_optimum-value,solution_difference)
            h=x/nrm(x)-math.sqrt(c)*d;eta0=max(0.,float(p@(gradients(x)*h))/(p@(h*h)))
            rr=gradients(x)-eta0*h;stationarity=float(nrm(rr));gap=float(-eta0*cone(x)+(p@(rr*rr))/(2*e*(1-e)))
            assert independent_optimum-value<=gap+1e-9 and gap<1e-7
        else:
            shape=d if row.method=='exact_unweighted_endpoint' else np.ones(len(d))
            old=boundary(shape,p,e,lam,tau);assert abs(old['critical_ratio']-value)<1e-10
        integ,quaderr=integral(x,p,e,lam,tau);pinteg,pquaderr=integral(xp,p,e,lam,tau)
        assert max(abs(integ-value),abs(pinteg-power_value))<1e-8
        if len(d)<=2:assert abs(value-power_value)<1e-8
        assert value>=power_value-1e-8
        records.append(dict(cell_id=row.cell_id,profile_formula_error=abs(value-row.optimized_capacity),independent_root_capacity=independent_optimum,capacity_solver_difference=independent_optimum-value,maximum_weight_scale_difference=solution_difference,
            integrated_capacity=integ,integral_discrepancy=abs(integ-value),quadrature_error_estimate=quaderr,power_integral_discrepancy=abs(pinteg-power_value),power_quadrature_error_estimate=pquaderr,
            radial_derivative_residual=scalar_derivative,stationarity_norm=stationarity,recomputed_upper_gap=gap,actual_variance_multiplier=variance))
        limit_key=(row.model,c)
        if limit_key not in seen:
            seen.add(limit_key);t=np.array(record['limiting_effective_scale']);t=t/nrm(t);limit_value=float((p@t)**2/(2*np.pi*e*(1-e)))
            assert abs(limit_value-row.limiting_capacity)<1e-10
            if 1<c<row.inverse_scale_cost-1e-12:
                # A different convex problem: maximize E T on the unit ball,
                # with a linear lower bound on its alignment with D.
                fit=minimize(lambda z:-float(p@z),t,jac=lambda z:-p,method='SLSQP',bounds=[(0,None)]*len(d),constraints=[dict(type='ineq',fun=lambda z:1-float(p@(z*z)),jac=lambda z:-2*p*z),dict(type='ineq',fun=lambda z:float(p@(d*z))-1/math.sqrt(c),jac=lambda z:p*d)],options={'ftol':1e-13,'maxiter':2000})
                assert fit.success and min(fit.x)>=0 and abs(float(p@fit.x)-float(p@t))<1e-7
                independent_limit=float((p@fit.x)**2/(2*np.pi*e*(1-e)))
            else:independent_limit=limit_value
            limit_records.append(dict(model=row.model,variance_budget=c,limiting_capacity=limit_value,independent_ball_capacity=independent_limit,difference=independent_limit-limit_value,variance_of_limiting_shape=float((p@(t*t))/(p@(d*t))**2)))
    pd.DataFrame(records).to_csv(a.out/'population_checks.csv',index=False);save(a.out/'all_independent_root_returns.json',returns);pd.DataFrame(limit_records).to_csv(a.out/'limiting_ball_checks.csv',index=False)
    cost_targets=pd.read_csv(a.root/'results/limiting_cost_targets.csv');target_checks=[]
    for row in cost_targets.itertuples(index=False):
        info=next(r for r in functions.values() if r['model']==row.model);d=np.array(info['scales']);p=np.array(info['probabilities']);mu=float(p@d);c=row.minimal_regular_covariance_multiplier
        if row.status=='positive_cost_for_boundary_target':
            sigma=math.sqrt(1-mu*mu);b=math.sqrt(1-1/c)/sigma;aa=1/math.sqrt(c)-mu*b;t=aa*d+b
            reached=float((p@t)**2/(p@(t*t))/(2*np.pi*e*(1-e)));assert abs(reached-row.target_boundary)<1e-10
        elif row.status=='unweighted_sufficient_for_boundary_target':reached=mu*mu/(2*np.pi*e*(1-e));assert reached>=row.target_boundary-1e-10 and c==1
        else:reached=None
        target_checks.append(dict(model=row.model,target_boundary=row.target_boundary,minimum_covariance_multiplier=c,reached_boundary=reached))
    pd.DataFrame(target_checks).to_csv(a.out/'inverse_budget_checks.csv',index=False)
    result=dict(date='2026-11-13',population_conditions=len(records),independent_intermediate_KKT_roots=len(returns),independent_Gaussian_profile_integrals=2*len(records),unique_limiting_ball_comparisons=len(limit_records),inverse_budget_conditions=len(target_checks),maximum_population_integral_discrepancy=max(max(r['integral_discrepancy'],r['power_integral_discrepancy']) for r in records),maximum_reported_quadrature_error=max(max(r['quadrature_error_estimate'],r['power_quadrature_error_estimate']) for r in records),maximum_independent_root_capacity_difference=max(abs(r['capacity_solver_difference']) for r in records),maximum_recomputed_upper_gap=max(r['recomputed_upper_gap'] for r in records),unresolved=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),priority_or_submission_certified=False)
    save(a.out/'verification.json',result);print(json.dumps(result))


if __name__=='__main__':main()
