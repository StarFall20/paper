"""Recorded J=3 finite-menu validation of the compensated existence law."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from menu_corrected_logit import MenuLoss,menu_features,support,residual_support,save


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--plan',type=Path,required=True);a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    plan=json.loads(a.plan.read_text());J=plan['alternatives'];records=[];crosschecks=0
    source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();plan_hash=hashlib.sha256(a.plan.read_bytes()).hexdigest()
    dimensions={m:{power:sorted({max(1,round(m*(plan['critical_ratios'][str(power)]+off))) for off in plan['orthogonal_dimension_offsets']}) for power in plan['powers']} for m in plan['label_menus']}
    for m in plan['label_menus']:
      N=plan['tau']*m;n=m+N;largest=max(q for values in dimensions[m].values() for q in values)
      for trial in range(plan['replications_per_budget']):
        rng=np.random.default_rng(np.random.SeedSequence([plan['seed'],m,trial]))
        S=rng.normal(size=(n,J,1));W=rng.normal(size=(n,J,largest))
        probabilities=softmax(plan['gamma']*S[...,0],axis=1)
        y=np.sum(rng.random(n)[:,None]>np.cumsum(probabilities,axis=1),axis=1)
        z=(y+(rng.random(n)<plan['epsilon'])*rng.integers(1,J,size=n))%J
        g=softmax(plan['gamma']*S[...,0]+np.eye(J)[z]*np.log((1-plan['epsilon'])/(plan['epsilon']/(J-1))),axis=1)
        for power in plan['powers']:
          for q in dimensions[m][power]:
            f=menu_features(np.concatenate([S,W[...,:q]],axis=2));loss=MenuLoss(f,y,g,m,power);check=support(loss)
            independent=None
            if trial%50==0:
                independent=residual_support(f,y,g,m,power)
                expected='feasible' if check['status'] in ['interior','boundary_or_near_boundary'] else check['status']
                assert expected==independent,(m,trial,power,q,check,independent);crosschecks+=1
            records.append(dict(label_menus=m,auxiliary_menus=N,trial=trial,alternatives=J,power=power,
                                orthogonal_slopes=q,total_common_slopes=q+1,estimated_constants=J-1,
                                orthogonal_dimension_ratio=q/m,critical_ratio=plan['critical_ratios'][str(power)],
                                status=check['status'],interior_margin=check.get('margin'),
                                separation_gap=check.get('separation_gap'),independent_residual_status=independent))
        if (trial+1)%20==0 or trial+1==plan['replications_per_budget']:
            frame=pd.DataFrame(records);frame.to_csv(a.out/'menu_phase_trials.csv',index=False)
            summary=frame.groupby(['label_menus','auxiliary_menus','power','orthogonal_slopes','orthogonal_dimension_ratio','critical_ratio','status'],as_index=False,dropna=False).size()
            summary.to_csv(a.out/'menu_phase_status_counts.csv',index=False)
            save(a.out/'menu_phase_status.json',dict(state='running',completed_draws=(len(records)//6),planned_draws=1000,completed_calls=len(records),planned_calls=6000,source_sha256=source,plan_sha256=plan_hash))
            print(json.dumps(dict(label_menus=m,completed=trial+1,support_calls=len(records))),flush=True)
    frame=pd.DataFrame(records);groups=['label_menus','auxiliary_menus','power','orthogonal_slopes','orthogonal_dimension_ratio','critical_ratio']
    summary=frame.groupby(groups,as_index=False).agg(replications=('trial','size'),finite_fits=('status',lambda s:int(s.eq('interior').sum())),
              outside=('status',lambda s:int(s.eq('outside').sum())),boundary=('status',lambda s:int(s.eq('boundary_or_near_boundary').sum())),
              rank_deficient=('status',lambda s:int(s.eq('rank_deficient').sum())),unresolved=('status',lambda s:int(s.eq('unresolved').sum())))
    summary['finite_frequency']=summary.finite_fits/summary.replications
    summary['binomial_MCSE']=np.sqrt(summary.finite_frequency*(1-summary.finite_frequency)/summary.replications)
    summary.to_csv(a.out/'menu_phase_summary.csv',index=False)
    assert len(records)==6000 and len(summary)==12 and summary.replications.eq(500).all()
    save(a.out/'menu_phase_status.json',dict(state='complete',completed_draws=1000,planned_draws=1000,completed_calls=6000,planned_calls=6000,
           completed_cells=len(summary),replications_per_cell=500,independent_residual_LP_crosschecks=crosschecks,
           unresolved=int(summary.unresolved.sum()),boundary=int(summary.boundary.sum()),rank_deficient=int(summary.rank_deficient.sum()),
           source_sha256=source,helper_sha256=hashlib.sha256(Path(__file__).with_name('menu_corrected_logit.py').read_bytes()).hexdigest(),
           plan_sha256=plan_hash,independent_novelty_or_submission_readiness_established=False,new_human_or_LLM_responses=0))
    print((a.out/'menu_phase_status.json').read_text(),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
