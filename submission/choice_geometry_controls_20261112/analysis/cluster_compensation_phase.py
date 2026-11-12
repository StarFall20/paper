"""Matched finite designs for the frozen three-task compensation plan."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from cluster_compensated_boundary import symmetric_root,save


def residual_support(x,y,g,label_rows,power):
    n=len(y);tau=(n-label_rows)/label_rows
    lower=np.r_[power*g[:label_rows]-y[:label_rows],-power*g[label_rows:]/tau]
    width=np.r_[np.full(label_rows,1-power),np.full(n-label_rows,power/tau)]
    result=linprog(np.zeros(n),A_eq=x.T,b_eq=np.zeros(x.shape[1]),bounds=np.column_stack([lower,lower+width]),method='highs')
    if result.status==2:return 'outside',None
    if not result.success:return 'unknown',None
    residual=float(np.max(np.abs(x.T@result.x)))
    if residual>1e-7:return 'unknown',residual
    return 'feasible',residual


def make_design(rng,persons,max_noise,dependence,epsilon):
    normal=rng.normal(size=(2*persons,3,max_noise+1))
    y=rng.integers(0,2,(2*persons,3)).astype(float)
    shared=rng.random(2*persons)<dependence
    common=rng.integers(0,2,2*persons)
    y[shared]=common[shared,None]
    z=np.logical_xor(y,rng.random(y.shape)<epsilon).astype(float)
    return normal,y.ravel(),(epsilon+(1-2*epsilon)*z).ravel()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--population',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    assert not (args.out/'phase_plan.json').exists(),'Do not overwrite a started phase experiment.'
    plan=json.loads(args.plan.read_text());table=pd.read_csv(args.population/'population_boundaries.csv')
    chosen=table[(table.exponent==max(plan['normal_exponents']))&(table.outcome_dependence==.75)&(table.model!='independent_attributes')]
    means=chosen.groupby(['power','model']).critical_slopes_per_labelled_person.mean().unstack()
    grids={}
    for persons in [40,80]:
        grids[persons]={}
        for power in plan['powers']:
            midpoint=float(means.loc[power].mean())
            grids[persons][power]=sorted({int(round(persons*(midpoint+offset))) for offset in [-.12,0.,.12]})
    derived=dict(date='2026-11-12',base_plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),population_sha256=hashlib.sha256((args.population/'population_boundaries.csv').read_bytes()).hexdigest(),seed=2026111209,replications=500,persons=[40,80],outcome_dependence=.75,epsilon=plan['epsilon'],tau=1,powers=plan['powers'],noise_dimensions=grids,paired_models=[m['name'] for m in plan['models'][1:]],fit_intercept=True,fit_unused_signal=True,method='Bounded residual feasibility; independent strict-interior and exterior-ray verification follows. Generic Gaussian position is invoked for the theoretical interior equivalence.',theory_selected_illustration=True,new_human_outcomes=0)
    save(args.out/'phase_plan.json',derived)
    started=time.monotonic();rows=[]
    for persons in [40,80]:
        max_noise=max(max(v) for v in grids[persons].values())
        for rep in range(500):
            rng=np.random.default_rng(np.random.SeedSequence([derived['seed'],persons,rep]))
            normal,y,g=make_design(rng,persons,max_noise,.75,plan['epsilon'])
            for model in plan['models'][1:]:
                root=symmetric_root(np.array(model['covariance']))
                transformed=np.einsum('ij,njk->nik',root,normal).reshape(6*persons,max_noise+1)
                for power in plan['powers']:
                    for q in grids[persons][power]:
                        x=np.column_stack([np.ones(6*persons),transformed[:,:q+1]])
                        status,residual=residual_support(x,y,g,3*persons,power)
                        rows.append(dict(persons=persons,replication=rep,model=model['name'],power=power,noise_slopes=q,total_parameters=q+2,slope_person_ratio=q/persons,labelled_choice_rows=3*persons,auxiliary_choice_rows=3*persons,status=status,maximum_equality_residual=residual))
            if (rep+1)%25==0:
                pd.DataFrame(rows).to_csv(args.out/'phase_trials.csv',index=False)
                save(args.out/'run_status.json',dict(completed_person_budget=persons,completed_replications_at_budget=rep+1,completed_records=len(rows),requested_records=24000,elapsed_seconds=time.monotonic()-started,complete=False))
                print('persons',persons,'replications',rep+1,'records',len(rows),flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(args.out/'phase_trials.csv',index=False)
    grouped=frame.groupby(['persons','model','power','noise_slopes'],sort=False)
    summaries=[]
    for key,part in grouped:
        counts=part.status.value_counts().to_dict()
        summaries.append(dict(zip(['persons','model','power','noise_slopes'],key))|dict(replications=len(part),feasible=counts.get('feasible',0),outside=counts.get('outside',0),unknown=counts.get('unknown',0),feasible_frequency=counts.get('feasible',0)/len(part)))
    pd.DataFrame(summaries).to_csv(args.out/'phase_summary.csv',index=False)
    save(args.out/'run_status.json',dict(completed_records=len(rows),requested_records=24000,elapsed_seconds=time.monotonic()-started,complete=True,unknowns=int((frame.status=='unknown').sum()),all_records_retained=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__=='__main__':main()
