"""Reproduce the retained population refinements or ordinary-boundary control."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from cluster_compensated_boundary import evaluate_boundary,ordinary_three_task_boundary,save

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--stage',choices=['refinement','matched'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    base=json.loads(a.plan.read_text());rows=[]
    if a.stage=='refinement':
        plan=json.loads((a.root/'population/refinement_plan.json').read_text());models=base['models'][1:];powers=plan['powers'];exponent=plan['normal_exponent'];seeds=plan['seeds']
    else:
        plan=json.loads((a.root/'matched_ordinary_control/plan.json').read_text());models=[]
        for i,(first,bracket) in enumerate(zip(plan['fixed_first_correlations'],plan['root_brackets'])):
            def cov(b):return np.array([[1,first,b],[first,1,1.5-first-b],[b,1.5-first-b,1.]])
            assert min(np.linalg.eigvalsh(cov(x)).min() for x in bracket)>0
            value=brentq(lambda b:ordinary_three_task_boundary(cov(b),1.)-1,*bracket,xtol=1e-14)
            models.append(dict(name='matched_ordinary_'+['A','B'][i],covariance=cov(value).tolist()))
        powers=plan['powers'];exponent=plan['normal_exponent'];seeds=[plan['normal_seed']]
    for model in models:
        for power in powers:
            for seed in seeds:
                result=evaluate_boundary(np.array(model['covariance']),.75,.25,power,1.,exponent,seed)
                record=dict(model=model['name'],outcome_dependence=.75,power=power) if a.stage=='refinement' else dict(model=model['name'],power=power)
                rows.append(record|result)
                print(model['name'],power,seed,flush=True)
    pd.DataFrame(rows).to_csv(a.out/('refined_boundaries.csv' if a.stage=='refinement' else 'boundary_screening.csv'),index=False)

if __name__=='__main__':main()
