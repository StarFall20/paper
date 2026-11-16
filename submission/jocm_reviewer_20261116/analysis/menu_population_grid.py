"""Run the recorded menu-boundary integration grid; no finite design sampling."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from menu_existence_population import MenuPopulation
from finite_ratio_signal_boundary import Population as BinaryPopulation
from menu_corrected_logit import save


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    plan=json.loads(a.plan.read_text());rows=[];source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for bits in plan['integration_bits']:
      for seed in plan['integration_scramble_seeds']:
       for gamma in plan['gamma']:
        for tau in plan['tau']:
            model=MenuPopulation(3,gamma,plan['epsilon'],tau,1.,bits,seed)
            for power in [1.,tau/(1+tau),.1]:
                model.power=power;r=model.critical()
                rows.append(dict(alternatives=3,gamma=gamma,tau=tau,epsilon=plan['epsilon'],power=power,event='fixed_positive',bits=bits,seed=seed,nodes=model.n,**r))
            r=model.critical(limiting=True)
            rows.append(dict(alternatives=3,gamma=gamma,tau=tau,epsilon=plan['epsilon'],power=np.nan,event='positive_limit',bits=bits,seed=seed,nodes=model.n,**r))
            if tau==plan['tau'][0]:
                r=model.critical(ordinary=True)
                rows.append(dict(alternatives=3,gamma=gamma,tau=np.nan,epsilon=plan['epsilon'],power=0.,event='ordinary',bits=bits,seed=seed,nodes=model.n,**r))
            pd.DataFrame(rows).to_csv(a.out/'menu_population_grid.csv',index=False)
            save(a.out/'menu_population_grid_status.json',dict(state='running',completed_evaluations=len(rows),planned_evaluations=72,source_sha256=source))
            print(json.dumps(dict(bits=bits,seed=seed,gamma=gamma,tau=tau,completed=len(rows))),flush=True)
    binary=[]
    for gamma in plan['gamma']:
      for tau in plan['tau']:
        model=MenuPopulation(2,gamma,plan['epsilon'],tau,1.,14,plan['integration_scramble_seeds'][0])
        for power in [1.,tau/(1+tau),.1]:
            model.power=power;new=model.critical();old=BinaryPopulation(np.sqrt(2)*gamma,plan['epsilon'],tau,power,nodes=200).critical()
            binary.append(dict(gamma_menu=gamma,gamma_binary=np.sqrt(2)*gamma,tau=tau,power=power,menu_ratio=new['critical_ratio'],binary_ratio=old['critical_ratio'],difference=new['critical_ratio']-old['critical_ratio']))
    pd.DataFrame(binary).to_csv(a.out/'menu_binary_boundary_reduction.csv',index=False)
    frame=pd.DataFrame(rows)
    selected=frame[(frame.bits==14)&(frame.seed==plan['integration_scramble_seeds'][0])]
    selected.to_csv(a.out/'menu_population_primary_values.csv',index=False)
    check={'state':'complete','completed_evaluations':len(rows),'planned_evaluations':72,
           'fixed_positive_cells_per_resolution_seed':12,'limiting_cells_per_resolution_seed':4,'ordinary_cells_per_resolution_seed':2,
           'binary_reduction_cells':len(binary),'maximum_binary_reduction_difference':max(abs(r['difference']) for r in binary),
           'source_sha256':source,'plan_sha256':hashlib.sha256(a.plan.read_bytes()).hexdigest(),
           'helper_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['menu_existence_population.py','menu_corrected_logit.py']},
           'phase_sampling_started':False,'rigorous_integration_error_bound_claimed':False,'novelty_or_submission_readiness_established':False}
    save(a.out/'menu_population_grid_status.json',check);print(json.dumps(check),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
