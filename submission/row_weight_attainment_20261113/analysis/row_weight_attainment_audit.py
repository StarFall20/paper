"""A fixed row-weight path for the existing known-scale compensation study.

All simulated cohorts are reconstructed from the preceding radial experiment.
The effective design is used for support only; it is not a substitute likelihood.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import pandas as pd
from radial_compensation_audit import boundary,profile_derivative,generated_design,residual_feasibility,two_point


def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def population(plan):
    rows=[]
    for law in plan['population_scale_laws']:
        if law['name']=='constant':d=np.ones(1);p=np.ones(1)
        else:
            d,p=two_point(law['large_probability'],law['small_scale']);d=np.array(d);p=np.array(p)
        moment=lambda k:float(p@d**k)
        tilted=lambda k:(p*d**k)/moment(k)
        def log_variance(k):
            mass=tilted(k);x=np.log(d);return float(mass@(x*x)-(mass@x)**2)
        for tau in plan['population_auxiliary_ratios']:
            lam=tau/(1+tau)
            for r in plan['population_exponents']:
                t=1-r;effective=d**t;result=boundary(effective,p,plan['epsilon'],lam,tau)
                variance=moment(2-2*r)/moment(2-r)**2
                v=result['scaled_minimizer']
                derivative=-v*sum(m*value*np.log(dd)*profile_derivative(v*value,plan['epsilon'],lam,tau) for dd,value,m in zip(d,effective,p))
                variance_derivative=2*variance*((tilted(2-r)@np.log(d))-(tilted(2-2*r)@np.log(d)))
                rows.append(dict(model=law['name'],large_probability=law['large_probability'],small_scale=law['small_scale'],auxiliary_ratio=tau,power=lam,row_exponent=r,
                    critical_ratio=result['critical_ratio'],profile_scale=v,capacity_derivative=float(derivative),
                    regular_covariance_multiplier=variance,regular_covariance_factor=4*(1-lam*(1-2*plan['epsilon'])**2)*variance,
                    covariance_multiplier_derivative=float(variance_derivative),small_exponent_quadratic_cost=log_variance(2),
                    mean_original_scale=moment(1),second_moment_original_scale=moment(2),effective_mean_scale=moment(t),effective_second_moment=moment(2*t)))
    return pd.DataFrame(rows)


def transitions(frame):
    rows=[]
    for m,part in frame.groupby('labels'):
        pivot=part.pivot(index='replication',columns='row_exponent',values='status')
        exponents=sorted(pivot.columns)
        for low,high in zip(exponents[:-1],exponents[1:]):
            before=pivot[low].eq('feasible').astype(int);after=pivot[high].eq('feasible').astype(int);diff=after-before
            rows.append(dict(labels=int(m),lower_exponent=low,higher_exponent=high,paired_draws=len(diff),
                finite_to_outside=int(((before==1)&(after==0)).sum()),outside_to_finite=int(((before==0)&(after==1)).sum()),
                finite_frequency_change=float(diff.mean()),paired_Monte_Carlo_SE=float(diff.std(ddof=1)/np.sqrt(len(diff)))))
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'run_status.json').exists(),'Use a fresh output directory.'
    plan=json.loads(a.plan.read_text());population(plan).to_csv(a.out/'population_weight_path.csv',index=False)
    reference=pd.read_csv(a.reference);reference=reference[(reference.power==plan['simulation_power'])&(reference.criterion=='compensated_residual_support')].set_index(['labels','replication','model'])
    rows=[];cohorts=[];started=time.monotonic()
    for m in plan['label_budgets']:
        q=int(round(plan['noise_label_ratio']*m))
        for rep in range(plan['replications']):
            base,y,g,d=generated_design(plan['source_seed'],m,rep,q,plan['epsilon'],plan['simulation_large_probability'],plan['simulation_small_scale'])
            digest=hashlib.sha256()
            for arr in [base,y,g,d]:digest.update(arr.tobytes())
            cohorts.append(dict(labels=m,replication=rep,source_draw_sha256=digest.hexdigest()))
            for r in plan['simulation_exponents']:
                status=residual_feasibility(d[:,None]**(1-r)*base,y,g,m,plan['simulation_power'])
                if r in [0,1]:assert status==reference.loc[(m,rep,'two_point_scale' if r==0 else 'constant_scale'),'status']
                rows.append(dict(labels=m,replication=rep,noise_slopes=q,total_parameters=q+2,row_exponent=r,power=plan['simulation_power'],auxiliary_ratio=plan['simulation_auxiliary_ratio'],status=status))
            if (rep+1)%100==0:
                pd.DataFrame(rows).to_csv(a.out/'weight_path_trials.csv',index=False)
                print('weighted support',m,rep+1,'records',len(rows),flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(a.out/'weight_path_trials.csv',index=False)
    pd.DataFrame(cohorts).to_csv(a.out/'source_draw_hashes.csv',index=False)
    summary=[]
    for (m,r),part in frame.groupby(['labels','row_exponent']):
        counts=part.status.value_counts().to_dict();summary.append(dict(labels=int(m),row_exponent=r,replications=len(part),finite=counts.get('feasible',0),outside=counts.get('outside',0),unknown=counts.get('unknown',0),finite_frequency=float(part.status.eq('feasible').mean())))
    pd.DataFrame(summary).to_csv(a.out/'weight_path_summary.csv',index=False);transitions(frame).to_csv(a.out/'paired_weight_transitions.csv',index=False)
    assert len(frame)==plan['expected_weighted_criteria'] and len(cohorts)==plan['expected_reused_cohorts']
    save(a.out/'run_status.json',dict(complete=True,weighted_criteria=len(frame),reused_cohorts=len(cohorts),endpoint_matches=2*len(cohorts),unknowns=int(frame.status.eq('unknown').sum()),elapsed_seconds=time.monotonic()-started,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),new_human_or_LLM_responses=0))


if __name__=='__main__':main()
