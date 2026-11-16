"""Resolve numerical unknowns without altering the complete raw phase output.

Regenerate only recorded simulation cases from the frozen random stream. Finite
powers use the independent original corrected-moment support LP and its explicit
recession certificate. This is a numerical audit, not new participant data.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from general_joint_power_boundary import PREDICTORS
from power_tuned_choice_audit import objective, support
from vanishing_power_choice import rescaled_support
from verify_general_joint_geometry import weak_separator_certificate
from public_conjoint_optimizer_audit import save


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(out):
    raw_path=out/'general_joint_power_phase_raw.csv'
    raw_hash=sha(raw_path)
    status=json.loads((out/'general_joint_power_status.json').read_text())
    assert status['completed_DGP_blocks']==status['requested_DGP_blocks']==12
    frame=pd.read_csv(raw_path)
    assert len(frame)==180 and (frame.replications==500).all()
    assert (frame[['interior','outside','boundary_or_near_boundary','rank_deficient','unresolved']].sum(axis=1)==500).all()
    unknown=json.loads((out/'general_joint_power_unresolved_ids.json').read_text())
    assert not unknown['possible_nesting_conflicts']
    assert len(unknown['cases'])==frame.unresolved.sum()
    source=Path(__file__).with_name('general_joint_power_boundary.py')
    assert sha(source)==status['source_sha256']
    cases={}
    for case in unknown['cases']:
        key=(case['labels'],case['tau'],case['prediction'],case['replication'])
        cases.setdefault(key,[]).append(case)
    boundaries=pd.read_csv(out/'general_joint_power_boundaries.csv')
    rng=np.random.default_rng(20261127);resolutions=[];matched=set()
    for m in [40,80]:
      for tau in [1,4]:
       N=tau*m
       for name,a,b in PREDICTORS[:3]:
        spec=boundaries[(boundaries.gamma==1.5)&(boundaries.tau==tau)&(boundaries.prediction==name)].iloc[0]
        dims=sorted({max(1,round(m*(spec.all_power_critical_ratio+offset))) for offset in [-.1,0.,.1]})
        for rep in range(500):
            # Consume the exact frozen source stream even for unrequested cases.
            s=rng.normal(size=m+N);y=(rng.random(m+N)<expit(1.5*s)).astype(float)
            z=np.logical_xor(y,rng.random(m+N)<.25).astype(float)
            orth=rng.normal(size=(m+N,max(dims)))
            key=(m,tau,name,rep)
            if key not in cases:continue
            g=expit(a*(1.5*s+(2*z-1)*np.log(3.))+b)
            for case in cases[key]:
                q=case['orthogonal_dimensions'];assert q in dims
                x=np.c_[np.ones(m+N),s,orth[:,:q]]
                lam=case['power'];power=case['power_name']
                if lam is None:
                    check=rescaled_support(x,y,g,m)
                    if check['status']=='interior':
                        witness=check['certified_positive_power']
                        independent=support(objective(x,y,g,m,witness))
                        assert independent['status']=='interior'
                        check={'status':'interior','witness_positive_power':witness,
                               'independent_original_moment':independent}
                    else:
                        cost,margin,error=weak_separator_certificate(x,y,g,m)
                        check={'status':'outside_mean_support','negative_auxiliary_recession':cost,
                               'minimum_human_margin':margin,'all_power_identity_error':error}
                else:
                    check=support(objective(x,y,g,m,lam))
                    # No coefficient vectors are needed in the aggregate report.
                    check={k:v for k,v in check.items() if k!='direction'}
                    assert check['status'] in ['interior','outside_mean_support'],check
                    if check['status']=='outside_mean_support':assert check['separation_gap']>1e-8
                    else:assert check['interior_margin']>1e-8 and check['equality_residual']<1e-7
                mask=(frame.labels==m)&(frame.tau==tau)&(frame.prediction==name)&(frame.orthogonal_dimensions==q)&(frame.power_name==power)
                assert mask.sum()==1
                target='interior' if check['status']=='interior' else 'outside'
                frame.loc[mask,'unresolved']-=1;frame.loc[mask,target]+=1
                resolutions.append({'case':case,'independent_verification':check})
                matched.add((m,tau,name,q,rep,power))
    assert len(matched)==len(unknown['cases']) and frame.unresolved.sum()==0
    assert (frame[['interior','outside','boundary_or_near_boundary','rank_deficient']].sum(axis=1)==500).all()
    frame['finite_frequency']=frame.interior/500
    wide=frame.pivot(index=['labels','tau','prediction','orthogonal_dimensions'],columns='power_name',values='interior')
    assert (wide.unit<=wide.inverse_sqrt_m).all() and (wide.inverse_sqrt_m<=wide.inverse_m).all()
    assert (wide.inverse_m<=wide.positive_power_limit).all()
    assert (wide.ordinary_zero<=wide.positive_power_limit).all()
    frame.to_csv(out/'general_joint_power_phase_verified.csv',index=False)
    assert sha(raw_path)==raw_hash
    report={'independent_Gaussian_draws':6000,'aggregate_cells':len(frame),'replications_per_cell':500,
            'complete_feasibility_calls':90000,'initial_unknowns':len(unknown['cases']),
            'resolved_by_independent_certificates':len(resolutions),'remaining_unresolved':0,
            'ordinary_boundary_or_proximity_calls':int(frame.loc[frame.power_name=='ordinary_zero','boundary_or_near_boundary'].sum()),
            'positive_power_boundary_or_proximity_calls':int(frame.loc[frame.power_name!='ordinary_zero','boundary_or_near_boundary'].sum()),
            'recorded_draw_level_nesting_conflicts':len(unknown['possible_nesting_conflicts']),
            'aggregate_nesting_verified_after_resolution':True,'raw_aggregate_output_unchanged':True,
            'raw_aggregate_sha256':raw_hash,'source_sha256':sha(source),
            'resolutions':resolutions,'new_humans_or_AI_responses':False,
            'priority_or_interval_guarantee_proven':False}
    save(out/'general_joint_power_phase_verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='resolutions'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    with threadpool_limits(limits=1):verify(args.out)
