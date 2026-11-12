"""Replay every radial support event using the original weighted objective."""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np
import pandas as pd
from scipy.integrate import quad
from power_tuned_choice_audit import objective,support


def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def independent_normal_box(lower,upper):
    phi=lambda h:math.exp(-h*h/2)/math.sqrt(2*math.pi)
    left=lambda h:(h-lower)**2*phi(h)
    right=lambda h:(h-upper)**2*phi(h)
    pieces=[]
    if lower>0:pieces.extend([quad(left,-np.inf,0,epsabs=2e-12),quad(left,0,lower,epsabs=2e-12)])
    else:pieces.append(quad(left,-np.inf,lower,epsabs=2e-12))
    if upper<0:pieces.extend([quad(right,upper,0,epsabs=2e-12),quad(right,0,np.inf,epsabs=2e-12)])
    else:pieces.append(quad(right,upper,np.inf,epsabs=2e-12))
    return 1-sum(x for x,_ in pieces),sum(e for _,e in pieces)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    plan=json.loads(a.plan.read_text());original=pd.read_csv(a.root/'results/phase_trials.csv')
    assert len(original)==6000
    population=pd.read_csv(a.root/'results/population_boundaries.csv');integrals=[]
    e=plan['epsilon'];tau=plan['tau']
    for row in population.itertuples(index=False):
        if row.model=='constant_scale':values=[1.];weights=[1.]
        else:
            values=[row.small_scale,math.sqrt((1-(1-row.large_probability)*row.small_scale**2)/row.large_probability)];weights=[1-row.large_probability,row.large_probability]
        total=0.;error=0.
        for d,weight in zip(values,weights):
            for r,prob in [(e,1-e),(1-e,e)]:
                lower=row.scaled_minimizer*d*r;upper=lower+row.scaled_minimizer*d*(1-row.power)/row.power
                val,err=independent_normal_box(lower,upper);total+=weight*prob*val;error+=weight*prob*err
            lower=-row.scaled_minimizer*d*(1-e)/tau;upper=row.scaled_minimizer*d*e/tau
            val,err=independent_normal_box(lower,upper);total+=tau*weight*val;error+=tau*weight*err
        discrepancy=abs(total-row.critical_ratio);assert discrepancy<1e-9,(row,discrepancy)
        assert row.critical_ratio<=row.general_analytic_upper_bound+1e-10
        if row.power==1:assert row.critical_ratio<=row.unit_power_sharper_upper_bound+1e-10
        integrals.append(dict(model=row.model,power=row.power,large_probability=row.large_probability,small_scale=row.small_scale,independent_integral=total,formula_discrepancy=discrepancy,quadrature_error_estimate=error))
    pd.DataFrame(integrals).to_csv(a.out/'independent_population_integrals.csv',index=False)
    checks=[];rays=[];verified=original.copy();verified['initial_status']=verified.status
    for key,part in original.groupby(['labels','replication'],sort=True):
        m,rep=map(int,key);q=int(round(plan['noise_label_ratio']*m))
        rng=np.random.default_rng(np.random.SeedSequence([plan['seed'],m,rep]))
        base=np.concatenate([np.ones((2*m,1)),rng.standard_normal((2*m,q+1))],axis=1)
        y=rng.integers(0,2,size=2*m).astype(float)
        flips=rng.uniform(size=2*m)<e;z=np.not_equal(y,flips).astype(float);g=e+(1-2*e)*z
        p=plan['simulation_large_probability'];small=plan['simulation_small_scale'];large=math.sqrt((1-(1-p)*small**2)/p)
        scale=np.where(rng.uniform(size=2*m)<p,large,small)
        for index,row in part.iterrows():
            x=base.copy() if row.model=='constant_scale' else base*scale[:,None]
            loss=objective(x,y,g,m,float(row.power));result=support(loss)
            if result['status']=='interior':status='human_finite' if row.criterion=='human_separation' else 'feasible'
            elif result['status']=='outside_mean_support':status='outside'
            else:status='unknown'
            if row.status!='unknown':assert status==row.status,(index,row.to_dict(),result)
            assert status!='unknown',(index,result)
            verified.loc[index,'status']=status
            checks.append(dict(row_index=int(index),independent_status=result['status'],interior_margin=result.get('interior_margin'),separation_gap=result.get('separation_gap'),initial_status=row.status,verified_status=status))
            if 'direction' in result:
                direction=np.array(result['direction']);recession=float(loss.weights@np.maximum(loss.x@direction,0)-loss.moment@direction)
                assert recession<-1e-8
                rays.append(dict(row_index=int(index),recession_cost=recession,direction=direction.tolist()))
        if (rep+1)%100==0:print('radial original-objective replays',m,rep+1,'records',len(checks),flush=True)
    pd.DataFrame(checks).to_csv(a.out/'independent_support_replays.csv',index=False);save(a.out/'exterior_directions.json',rays)
    verified.to_csv(a.out/'phase_trials_verified.csv',index=False)
    summaries=[]
    for key,part in verified.groupby(['labels','model','power','criterion'],sort=False):
        finite=part.status.isin(['feasible','human_finite']).sum()
        summaries.append(dict(zip(['labels','model','power','criterion'],key))|dict(replications=len(part),finite=int(finite),outside_or_separated=int(part.status.isin(['outside','human_separated']).sum()),unknown=int((part.status=='unknown').sum()),finite_frequency=float(finite/len(part))))
    pd.DataFrame(summaries).to_csv(a.out/'phase_summary_verified.csv',index=False)
    humans=verified[verified.criterion=='human_separation'].pivot(index=['labels','replication'],columns='model',values='status')
    assert (humans.constant_scale==humans.two_point_scale).all()
    save(a.out/'verification.json',dict(date='2026-11-12',all_support_events_replayed=len(checks),independent_exterior_directions=len(rays),independent_population_integrals=len(integrals),maximum_integral_formula_discrepancy=max(x['formula_discrepancy'] for x in integrals),maximum_reported_quadrature_error=max(x['quadrature_error_estimate'] for x in integrals),original_unknowns_retained=int((original.status=='unknown').sum()),final_unknowns=0,initial_human_invariance_flags=int(json.loads((a.root/'results/run_status.json').read_text())['human_separation_invariance_mismatches']),human_invariance_flags_explanation='The initial strict-margin solver returned unknown on two different rows of the paired human comparisons. Every human objective is independently replayed; original unknowns are retained in initial_status.',final_human_separation_invariance_mismatches=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),originality_or_submission_certified=False))


if __name__=='__main__':main()
