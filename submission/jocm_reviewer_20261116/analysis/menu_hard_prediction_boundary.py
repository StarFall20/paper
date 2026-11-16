"""Test the proposed whole-menu boundary with frozen hard category predictions.

This extends the existing population calculation to boundary-simplex g. It is
an existence calculation, not an accuracy or calibration advantage claim.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from menu_existence_population import MenuPopulation
from menu_corrected_logit import save,vertex_cone_projection


class HardMenuPopulation(MenuPopulation):
    def __init__(self,J,gamma,epsilon,tau,power,bits=12,seed=2026111171):
        super().__init__(J,gamma,epsilon,tau,power,bits,seed)
        self.g=np.broadcast_to(np.eye(J)[None,:,:],(self.n,J,J)).copy()
        assert np.min(self.g)==0 and np.max(abs(self.g.sum(-1)-1))==0


def checks(model):
    identity=binary_identity=0.;derivatives=[]
    for eta in [-.4,0.,.7]:
        left,right=model.zero_scale_identity(eta)
        identity=max(identity,abs(left-right));derivatives.append(left)
        if model.J==2:
            w=model.H-eta*model.S
            projection=vertex_cone_projection(w[:,None,:]+np.zeros((1,2,1)),np.arange(2)[None,:])
            chosen_mass=model.g  # axes are (integration node, proxy Z, true Y)
            # For a given (Z,Y), max(p)-g.p = 2*c*g_Y; c=max(p).
            expected=4*float(np.mean(np.sum(model.joint*np.max(projection,axis=-1)[:,None,:]*chosen_mass,axis=(1,2))))
            binary_identity=max(binary_identity,abs(left-expected))
    assert identity<1e-9 and binary_identity<1e-9
    return dict(maximum_menu_identity_error=identity,maximum_binary_gap_identity_error=binary_identity,
                minimum_tested_zero_scale_derivative=min(derivatives),maximum_tested_zero_scale_derivative=max(derivatives))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    plan=json.loads(a.plan.read_text());rows=[];diagnostics=[]
    source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for bits in plan['population']['bits']:
      for seed in plan['population']['seeds']:
       for J in plan['population']['alternatives']:
        for gamma in plan['population']['gamma']:
         for tau in plan['population']['tau']:
          for epsilon in plan['population']['proxy_error']:
            model=HardMenuPopulation(J,gamma,epsilon,tau,1.,bits,seed)
            diagnostic=checks(model)
            diagnostics.append(dict(alternatives=J,gamma=gamma,tau=tau,epsilon=epsilon,bits=bits,seed=seed,**diagnostic))
            for power in plan['population']['powers']:
                model.power=power;result=model.critical()
                rows.append(dict(alternatives=J,gamma=gamma,tau=tau,epsilon=epsilon,power=power,event='fixed_positive',bits=bits,seed=seed,**result))
            ordinary=model.critical(ordinary=True);limiting=model.critical(limiting=True)
            for event,power,result in [('ordinary',0.,ordinary),('positive_limit',np.nan,limiting)]:
                rows.append(dict(alternatives=J,gamma=gamma,tau=tau,epsilon=epsilon,power=power,event=event,bits=bits,seed=seed,**result))
            if J==2 and epsilon==1:
                assert max(abs(diagnostic['minimum_tested_zero_scale_derivative']),abs(diagnostic['maximum_tested_zero_scale_derivative']))<1e-9
                assert abs(ordinary['critical_ratio']-limiting['critical_ratio'])<1e-8
            else:
                assert diagnostic['minimum_tested_zero_scale_derivative']>0
                assert limiting['critical_ratio']>ordinary['critical_ratio']
            pd.DataFrame(rows).to_csv(a.out/'hard_menu_population_grid.csv',index=False)
            pd.DataFrame(diagnostics).to_csv(a.out/'hard_menu_derivative_checks.csv',index=False)
            save(a.out/'hard_menu_population_status.json',dict(state='running',population_evaluations=len(rows),planned_population_evaluations=320,source_sha256=source))
            print(json.dumps(dict(bits=bits,seed=seed,J=J,gamma=gamma,tau=tau,epsilon=epsilon,ordinary=ordinary['critical_ratio'],positive_limit=limiting['critical_ratio'],completed=len(rows))),flush=True)
    frame=pd.DataFrame(rows);assert len(frame)==320
    selected=frame[(frame.bits==max(plan['population']['bits']))&(frame.seed==plan['population']['seeds'][0])]
    selected.to_csv(a.out/'hard_menu_primary_values.csv',index=False)
    save(a.out/'hard_menu_population_status.json',dict(state='complete',population_evaluations=len(rows),event_conditions=80,
         resolutions=len(plan['population']['bits']),scramble_seeds=len(plan['population']['seeds']),
         diagnostic_conditions=len(diagnostics),source_sha256=source,plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),
         helper_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['menu_existence_population.py','menu_corrected_logit.py']},
         finite_phase_sampling_started=False,priority_or_publication_readiness_established=False,new_human_or_LLM_responses=0))
    print((a.out/'hard_menu_population_status.json').read_text(),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
