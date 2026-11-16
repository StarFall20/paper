"""Paired finite-size validation for one transparently selected calibrated case."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import norm
from threadpoolctl import threadpool_limits
from power_tuned_choice_audit import objective,support
from public_conjoint_optimizer_audit import save


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    population_file=a.out/'calibrated_aligned_boundaries.csv';population=pd.read_csv(population_file)
    spec=population[(population.epsilon==.1)&(population.tau==1)&(population.power==1)].iloc[0]
    internal,external=spec.aligned_critical_ratio,spec.external_critical_ratio
    frozen={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'population_sha256':hashlib.sha256(population_file.read_bytes()).hexdigest(),
            'loss_source_sha256':hashlib.sha256((Path(__file__).parent/'power_tuned_choice_audit.py').read_bytes()).hexdigest()}
    rng=np.random.default_rng(20261202);rows=[];raw=[];unresolved=[];completed=0
    for m in [80,160]:
        dimensions=sorted({max(1,int(round(m*v))) for v in [internal-.08,internal,(internal+external)/2,external,external+.08]})
        count={d:{model:{s:0 for s in ['interior','outside_mean_support','boundary_or_near_boundary','rank_deficient','support_check_unresolved']} for model in ['external','aligned']} for d in dimensions}
        paired={d:[] for d in dimensions}
        for rep in range(500):
            n=2*m;T=rng.normal(size=n);R=rng.normal(size=n);W=rng.normal(size=(n,max(dimensions)-1));y=rng.integers(0,2,size=n).astype(float)
            threshold=np.where(y==1,norm.ppf(.1),norm.ppf(.9));z=(T>threshold).astype(float);g=.1+.8*z
            for d in dimensions:
                pair={}
                for model,index in [('external',R),('aligned',T)]:
                    x=np.c_[np.ones(n),index,W[:,:d-1]]
                    result=support(objective(x,y,g,m,1.));state=result['status']
                    if state not in count[d][model]:count[d][model][state]=0
                    count[d][model][state]+=1;pair[model]=state
                    raw.append({'labels':m,'replication':rep,'total_slopes':d,'model':model,'status':state})
                    if state=='support_check_unresolved':unresolved.append({'labels':m,'replication':rep,'total_slopes':d,'model':model,**result})
                if pair['external'] in ['interior','outside_mean_support'] and pair['aligned'] in ['interior','outside_mean_support']:
                    paired[d].append(float(pair['aligned']=='interior')-float(pair['external']=='interior'))
            if (rep+1)%100==0:
                pd.DataFrame(raw).to_csv(a.out/'aligned_phase_trial_diagnostics.csv',index=False)
                save(a.out/'aligned_phase_checkpoint.json',{**frozen,'completed_label_blocks':completed,'current_labels':m,'current_completed_repetitions':rep+1,'requested_repetitions_per_label':500,'unresolved':len(unresolved)})
                print(f'Aligned phase labels{m}: {rep+1}/500 paired repetitions; {len(dimensions)} dimensions; unresolved {len(unresolved)}',flush=True)
        for d in dimensions:
            diff=np.array(paired[d])
            for model in ['external','aligned']:
                rows.append({'labels':m,'tau':1,'epsilon':.1,'power':1,'total_slopes':d,'total_slope_label_ratio':d/m,
                             'conditional_remaining_ratio':(d-1)/m if model=='aligned' else d/m,'model':model,
                             'critical_ratio':internal if model=='aligned' else external,'replications':500,**count[d][model],
                             'finite_frequency':count[d][model]['interior']/500,
                             'paired_classified':len(diff),'paired_finite_difference_aligned_minus_external':float(diff.mean()) if len(diff) else None,
                             'paired_difference_MCSE':float(diff.std(ddof=1)/np.sqrt(len(diff))) if len(diff)>1 else None})
        completed+=1
        pd.DataFrame(rows).to_csv(a.out/'aligned_prediction_phase.csv',index=False)
        save(a.out/'aligned_phase_unresolved.json',unresolved)
        save(a.out/'aligned_phase_status.json',{**frozen,'completed_label_blocks':completed,'paired_draws':completed*500,'completed_cells':len(rows),
             'feasibility_calls':len(raw),'replications_per_cell':500,'unknowns':len(unresolved),
             'illustrative_condition_selected_after_population_grid':True,'new_human_or_AI_responses':0,'raw_private_questionnaire_used':False,
             'finite_size_can_resolve_analytic_gap_claimed':False,'submission_readiness_proven':False})
    assert hashlib.sha256(population_file.read_bytes()).hexdigest()==frozen['population_sha256']
    print('Completed all paired index-alignment phase cells.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
