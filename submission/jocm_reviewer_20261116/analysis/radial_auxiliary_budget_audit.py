"""Nested auxiliary rows on the original radial simulation draws.

The variance rule is established; the fixed-power arm separates the effect of
the sample size from changing that rule's power. No coefficient-risk ranking.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import pandas as pd
from radial_compensation_audit import boundary,two_point,generated_design,residual_feasibility
from power_tuned_choice_audit import objective,support
from finite_ratio_signal_boundary import cone_projection_identity
from scipy.optimize import linprog
from scipy.special import expit


def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def geometry_checks():
    rng=np.random.default_rng(2026111267);rows=[];projection_errors=[]
    for j in range(120):
        m=10+j%6;N=m*(1+j%2);q=2+j%6;n=m+N
        gamma=[0.,.75,1.5][j%3];power=[.2,.5,1.][j%3]
        s=rng.normal(size=n);orth=rng.normal(size=(n,q))
        values,_=two_point(.1,.1);d=np.where(rng.random(n)<.1,values[1],values[0])
        y=(rng.random(n)<expit(gamma*d*s)).astype(float)
        z=np.logical_xor(y,rng.random(n)<.25).astype(float);g=.25+.5*z
        signs=np.r_[1-2*y[:m],2*z[m:]-1]
        r=np.abs(y[:m]-g[:m]);ratio=N/m
        lower=d*np.r_[power*r,np.full(N,-power*.75/ratio)]
        upper=d*np.r_[1-power+power*r,np.full(N,power*.25/ratio)]
        V=np.column_stack([signs,signs*s])
        matrix=np.column_stack([V,orth*signs[:,None]]).T
        fit=linprog(np.zeros(n),A_eq=matrix,b_eq=np.zeros(q+2),bounds=np.column_stack([lower,upper]),method='highs')
        transformed='feasible' if fit.success and np.max(abs(matrix@fit.x))<1e-7 else 'outside' if fit.status==2 else 'unknown'
        x=d[:,None]*np.column_stack([np.ones(n),s,orth])
        original=support(objective(x,y,g,m,power))['status']
        expected='feasible' if original=='interior' else 'outside' if original=='outside_mean_support' else 'unknown'
        assert transformed==expected and expected!='unknown',(j,original,fit.message)
        rows.append(dict(case=j,gamma=gamma,power=power,original_status=original,transformed_status=transformed))
        if j<12:projection_errors.append(cone_projection_identity(lower,upper,V,rng.normal(size=n))[0])
    return rows,dict(original_to_scaled_interval_matches=len(rows),nonzero_signal_cases=sum(r['gamma']!=0 for r in rows),independent_constrained_projection_cases=len(projection_errors),maximum_projection_identity_error=max(projection_errors))


def nested_draw(plan,m,rep,maxN):
    q=int(round(plan['noise_label_ratio']*m))
    base,y,g,d=generated_design(plan['source_seed'],m,rep,q,plan['epsilon'],plan['large_scale_probability'],plan['small_scale'])
    extra=maxN-m
    if extra:
        rng=np.random.default_rng(np.random.SeedSequence([plan['extension_seed'],m,rep]))
        added=np.column_stack([np.ones(extra),rng.normal(size=(extra,q+1))])
        ay=rng.integers(0,2,extra).astype(float);az=np.logical_xor(ay,rng.random(extra)<plan['epsilon']).astype(float)
        ag=plan['epsilon']+(1-2*plan['epsilon'])*az
        values,_=two_point(plan['large_scale_probability'],plan['small_scale'])
        ad=np.where(rng.random(extra)<plan['large_scale_probability'],values[1],values[0])
        base=np.vstack([base,added]);y=np.r_[y,ay];g=np.r_[g,ag];d=np.r_[d,ad]
    return base,y,g,d


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    assert not (a.out/'run_status.json').exists(),'Use a fresh output directory.'
    plan=json.loads(a.plan.read_text());reference=pd.read_csv(a.reference)
    reference=reference[(reference.power==.5)&(reference.criterion=='compensated_residual_support')]
    lookup=reference.set_index(['labels','replication','model']).status.to_dict()
    checks,record=geometry_checks();pd.DataFrame(checks).to_csv(a.out/'scaled_interval_checks.csv',index=False);save(a.out/'geometry_checks.json',record)
    population=[]
    for name in plan['models']:
        values,mass=([1.],[1.]) if name=='constant_scale' else two_point(plan['large_scale_probability'],plan['small_scale'])
        for ratio in plan['population_ratios']:
            for rule,power in [('variance_rule',ratio/(1+ratio)),('held_power',1/41)]:
                result=boundary(values,mass,plan['epsilon'],power,ratio)
                variance=4*(1-2*power*.25+(1+1/ratio)*power**2*.25)
                population.append(dict(model=name,auxiliary_ratio=ratio,rule=rule,power=power,regular_covariance_times_labels=variance,**result))
    pd.DataFrame(population).to_csv(a.out/'population_budget_boundaries.csv',index=False)
    rows=[];baseline_matches=0;programs=0;started=time.monotonic()
    for m in plan['label_budgets']:
        maxN=int(round(m*max(plan['auxiliary_ratios'])))
        for rep in range(plan['replications']):
            base,y,g,d=nested_draw(plan,m,rep,maxN)
            for name in plan['models']:
                for ratio in plan['auxiliary_ratios']:
                    N=int(round(m*ratio));used=m+N
                    x=base[:used] if name=='constant_scale' else base[:used]*d[:used,None]
                    cached={}
                    for rule,power in [('variance_rule',N/(m+N)),('held_power',1/41)]:
                        if power not in cached:
                            cached[power]=residual_feasibility(x,y[:used],g[:used],m,power);programs+=1
                        status=cached[power]
                        if ratio==1 and rule=='variance_rule':
                            assert status==lookup[m,rep,name],(m,rep,name,status,lookup[m,rep,name]);baseline_matches+=1
                        rows.append(dict(labels=m,replication=rep,model=name,auxiliary_rows=N,auxiliary_ratio=N/m,rule=rule,power=power,noise_slopes=int(round(plan['noise_label_ratio']*m)),total_parameters=x.shape[1],status=status,duplicate_criterion_at_smallest_ratio=ratio==min(plan['auxiliary_ratios'])))
            if (rep+1)%50==0:
                pd.DataFrame(rows).to_csv(a.out/'nested_auxiliary_trials.csv',index=False)
                save(a.out/'run_status.json',dict(complete=False,completed_label_budget=m,replications_at_budget=rep+1,records=len(rows),distinct_programs=programs,elapsed_seconds=time.monotonic()-started))
                print('nested auxiliary labels',m,'replications',rep+1,'records',len(rows),flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(a.out/'nested_auxiliary_trials.csv',index=False)
    summary=[]
    for key,part in frame.groupby(['labels','model','auxiliary_ratio','rule'],sort=False):
        counts=part.status.value_counts().to_dict()
        summary.append(dict(zip(['labels','model','auxiliary_ratio','rule'],key))|dict(power=float(part.power.iloc[0]),replications=len(part),feasible=counts.get('feasible',0),outside=counts.get('outside',0),unknown=counts.get('unknown',0),finite_frequency=counts.get('feasible',0)/len(part)))
    pd.DataFrame(summary).to_csv(a.out/'nested_auxiliary_summary.csv',index=False)
    save(a.out/'run_status.json',dict(complete=True,records=len(rows),distinct_programs=programs,matched_original_ratio_one_records=baseline_matches,unknowns=int((frame.status=='unknown').sum()),elapsed_seconds=time.monotonic()-started,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),new_human_or_LLM_responses=0))


if __name__=='__main__':main()
