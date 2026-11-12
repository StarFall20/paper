"""Known positive task scales with matched information and identical human separation.

This exploratory extension keeps the compensated loss and oracle predictor.
The row scale multiplies the entire utility, including the fitted intercept.
It is specified, not estimated from null-signal choices.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import pandas as pd
from scipy.optimize import brentq,linprog
from scipy.special import ndtr
from scipy.stats import norm


def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def tail(x):return (1+x*x)*ndtr(-x)-x*norm.pdf(x)
def tail_derivative(x):return 2*(x*ndtr(-x)-norm.pdf(x))


def profile(u,epsilon,power,tau):
    e=epsilon;extra=(1-power)/power
    return float(1+tau-(1-e)*(tail(-e*u)+tail((extra+e)*u))-e*(tail(-(1-e)*u)+tail((extra+1-e)*u))-tau*(tail((1-e)*u/tau)+tail(e*u/tau)))


def profile_derivative(u,epsilon,power,tau):
    e=epsilon;extra=(1-power)/power
    return float(-((1-e)*(-e*tail_derivative(-e*u)+(extra+e)*tail_derivative((extra+e)*u))+e*(-(1-e)*tail_derivative(-(1-e)*u)+(extra+1-e)*tail_derivative((extra+1-e)*u))+(1-e)*tail_derivative((1-e)*u/tau)+e*tail_derivative(e*u/tau)))


def boundary(values,probabilities,epsilon,power,tau):
    scales=np.array(values);mass=np.array(probabilities)
    if np.any(scales<=0) or abs(mass.sum()-1)>1e-12:raise ValueError('Positive scales and probability weights are required.')
    fun=lambda u:float(sum(w*profile(d*u,epsilon,power,tau) for d,w in zip(scales,mass)))
    derivative=lambda u:float(sum(w*d*profile_derivative(d*u,epsilon,power,tau) for d,w in zip(scales,mass)))
    upper=1.
    while derivative(upper)>=0:upper*=2
    optimum=brentq(derivative,0,upper,xtol=1e-13)
    ed=float(mass@scales);ed2=float(mass@(scales**2));r2=epsilon*(1-epsilon)
    return dict(critical_ratio=fun(optimum),scaled_minimizer=optimum,derivative_residual=abs(derivative(optimum)),mean_scale=ed,second_moment_scale=ed2,general_analytic_upper_bound=ed*ed/(np.pi*power**2*r2*ed2),unit_power_sharper_upper_bound=ed*ed/(2*np.pi*r2*ed2) if power==1 else None)


def two_point(p,small):return [small,float(np.sqrt((1-(1-p)*small*small)/p))],[1-p,p]


def residual_feasibility(x,y,g,m,power):
    tau=(len(y)-m)/m
    lower=np.r_[power*g[:m]-y[:m],-power*g[m:]/tau]
    upper=lower+np.r_[np.full(m,1-power),np.full(len(y)-m,power/tau)]
    fit=linprog(np.zeros(len(y)),A_eq=x.T,b_eq=np.zeros(x.shape[1]),bounds=np.column_stack([lower,upper]),method='highs')
    if fit.status==2:return 'outside'
    if not fit.success or np.max(abs(x.T@fit.x))>1e-7:return 'unknown'
    return 'feasible'


def ordinary_separation(x,y,m):
    signed=(2*y[:m]-1)[:,None]*x[:m]
    fit=linprog(np.zeros(x.shape[1]),A_ub=-signed,b_ub=-np.ones(m),bounds=[(None,None)]*x.shape[1],method='highs')
    if fit.status==2:return 'human_finite'
    if fit.success and np.min(signed@fit.x)>1-1e-7:return 'human_separated'
    return 'unknown'


def generated_design(seed,m,rep,q,epsilon,p,small):
    rng=np.random.default_rng(np.random.SeedSequence([seed,m,rep]))
    base=np.column_stack([np.ones(2*m),rng.normal(size=(2*m,q+1))])
    y=rng.integers(0,2,2*m).astype(float)
    z=np.logical_xor(y,rng.random(2*m)<epsilon).astype(float)
    g=epsilon+(1-2*epsilon)*z
    values,_=two_point(p,small)
    d=np.where(rng.random(2*m)<p,values[1],values[0])
    return base,y,g,d


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);assert not (args.out/'run_status.json').exists(),'Use a fresh result directory.'
    plan=json.loads(args.plan.read_text());e=plan['epsilon'];tau=plan['tau'];population=[]
    for power in plan['population_powers']:
        population.append(dict(model='constant_scale',power=power,large_probability=1,small_scale=1,**boundary([1.],[1.],e,power,tau)))
        for p in plan['population_large_probabilities']:
            for small in plan['population_small_scales']:
                values,weights=two_point(p,small)
                population.append(dict(model='two_point_scale',power=power,large_probability=p,small_scale=small,**boundary(values,weights,e,power,tau)))
    pd.DataFrame(population).to_csv(args.out/'population_boundaries.csv',index=False)
    rows=[];started=time.monotonic();mismatch=0
    for m in plan['label_budgets']:
        q=int(round(plan['noise_label_ratio']*m))
        for rep in range(plan['replications']):
            base,y,g,d=generated_design(plan['seed'],m,rep,q,e,plan['simulation_large_probability'],plan['simulation_small_scale'])
            human=[]
            for name,scale in [('constant_scale',np.ones(2*m)),('two_point_scale',d)]:
                x=scale[:,None]*base
                h=ordinary_separation(x,y,m);human.append(h)
                rows.append(dict(labels=m,replication=rep,noise_slopes=q,total_parameters=q+2,model=name,power=0.,status=h,criterion='human_separation'))
                for power in plan['simulation_powers']:
                    status=residual_feasibility(x,y,g,m,power)
                    rows.append(dict(labels=m,replication=rep,noise_slopes=q,total_parameters=q+2,model=name,power=power,status=status,criterion='compensated_residual_support'))
            mismatch+=human[0]!=human[1]
            if (rep+1)%50==0:
                pd.DataFrame(rows).to_csv(args.out/'phase_trials.csv',index=False)
                save(args.out/'run_status.json',dict(completed_budget=m,completed_replications_at_budget=rep+1,records=len(rows),requested_records=6000,complete=False,elapsed_seconds=time.monotonic()-started))
                print('radial labels',m,'replications',rep+1,'records',len(rows),flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(args.out/'phase_trials.csv',index=False)
    summary=[]
    for key,part in frame.groupby(['labels','model','power','criterion'],sort=False):
        counts=part.status.value_counts().to_dict();finite=counts.get('feasible',0)+counts.get('human_finite',0)
        summary.append(dict(zip(['labels','model','power','criterion'],key))|dict(replications=len(part),finite=finite,outside_or_separated=counts.get('outside',0)+counts.get('human_separated',0),unknown=counts.get('unknown',0),finite_frequency=finite/len(part)))
    pd.DataFrame(summary).to_csv(args.out/'phase_summary.csv',index=False)
    save(args.out/'run_status.json',dict(complete=True,records=len(rows),requested_records=6000,human_separation_invariance_mismatches=mismatch,unknowns=int((frame.status=='unknown').sum()),elapsed_seconds=time.monotonic()-started,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),new_human_or_LLM_answers=0))


if __name__=='__main__':main()
