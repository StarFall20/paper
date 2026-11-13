"""Population-selected illustration on previous choices and extended attributes."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import pandas as pd
from radial_compensation_audit import residual_feasibility,ordinary_separation


def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def reconstruct(plan,m,rep,d,p):
    q0=int(round(plan['original_noise_label_ratio']*m));q=int(round(plan['noise_label_ratio']*m));rng=np.random.default_rng(np.random.SeedSequence([plan['source_seed'],m,rep]))
    old=np.column_stack([np.ones(2*m),rng.normal(size=(2*m,q0+1))]);y=rng.integers(0,2,2*m).astype(float);z=np.logical_xor(y,rng.random(2*m)<plan['epsilon']).astype(float);g=plan['epsilon']+(1-2*plan['epsilon'])*z;u=rng.random(2*m)
    low=plan['original_small_scale'];prob=plan['original_large_probability'];high=np.sqrt((1-(1-prob)*low*low)/prob);original_d=np.where(u<prob,high,low)
    source=hashlib.sha256()
    for arr in [old,y,g,original_d]:source.update(arr.tobytes())
    extra=np.random.default_rng(np.random.SeedSequence([plan['extra_attribute_seed'],m,rep])).normal(size=(2*m,q-q0));base=np.column_stack([old,extra]);idx=np.searchsorted(np.cumsum(p),u,side='right');scales=d[idx]
    stop=m+int(round(plan['selected_cell_details']['auxiliary_ratio']*m));base=base[:stop];y=y[:stop];g=g[:stop];scales=scales[:stop];idx=idx[:stop]
    return base,y,g,scales,idx,source.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--reference-hashes',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'phase_trials.csv').exists(),'Use a fresh output directory.'
    plan=json.loads(a.plan.read_text());population=pd.read_csv(a.root/'results/budget_population.csv');eligible=population[(population.scale_levels>=3)&(population.auxiliary_ratio<=4)&(population.variance_budget<=2)];selected=eligible.sort_values('capacity_gain',ascending=False).iloc[0];assert selected.cell_id==plan['selected_cell']
    verification=json.loads((a.root/'verification/verification.json').read_text());assert verification['population_conditions']==195 and verification['unresolved']==0
    function=next(r for r in json.loads((a.root/'results/weight_functions.json').read_text()) if r['cell_id']==plan['selected_cell']);d=np.array(function['scales']);mass=np.array(function['probabilities']);optimized=np.array(function['optimized_row_weights']);r=function['power_exponent'];lam=plan['selected_cell_details']['power']
    reference=pd.read_csv(a.reference_hashes).set_index(['labels','replication']).source_draw_sha256
    rows=[];cohorts=[];started=time.monotonic()
    for m in plan['label_budgets']:
        for rep in range(plan['replications']):
            base,y,g,scale,idx,source=reconstruct(plan,m,rep,d,mass);assert source==reference.loc[m,rep]
            digest=hashlib.sha256()
            for arr in [base,y,g,scale]:digest.update(arr.tobytes())
            cohorts.append(dict(labels=m,replication=rep,previous_source_sha256=source,current_design_sha256=digest.hexdigest()))
            for name in plan['methods']:
                if name=='human_only':status=ordinary_separation(base*scale[:,None],y,m);weight=np.ones(len(y));power=0.
                else:
                    power=lam
                    weight={'unweighted_compensation':np.ones(len(y)),'power_at_budget':scale**(-r),'optimized_at_budget':optimized[idx],'inverse_scale':1/scale}[name]
                    status=residual_feasibility(base*(weight*scale)[:,None],y,g,m,power)
                rows.append(dict(labels=m,replication=rep,noise_slopes=int(round(m*plan['noise_label_ratio'])),total_parameters=base.shape[1],auxiliary_rows=len(y)-m,method=name,power=power,status=status))
            if (rep+1)%100==0:pd.DataFrame(rows).to_csv(a.out/'phase_trials.csv',index=False);print('budget illustration',m,rep+1,'records',len(rows),flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(a.out/'phase_trials.csv',index=False);pd.DataFrame(cohorts).to_csv(a.out/'source_cohort_hashes.csv',index=False)
    summary=[]
    for (m,method),part in frame.groupby(['labels','method'],sort=False):
        finite=int(part.status.isin(['feasible','human_finite']).sum());unknown=int(part.status.eq('unknown').sum())
        summary.append(dict(labels=int(m),method=method,replications=len(part),finite=finite,outside_or_separated=int(part.status.isin(['outside','human_separated']).sum()),unknown=unknown,finite_frequency=finite/len(part) if not unknown else None))
    pd.DataFrame(summary).to_csv(a.out/'phase_summary.csv',index=False)
    paired=[]
    for m,part in frame.groupby('labels'):
        states=part.pivot(index='replication',columns='method',values='status');pivot=states.isin(['feasible','human_finite']).astype(int)
        for first,second in [('power_at_budget','optimized_at_budget'),('human_only','optimized_at_budget'),('optimized_at_budget','inverse_scale'),('unweighted_compensation','power_at_budget')]:
            resolved=states[first].ne('unknown')&states[second].ne('unknown')
            before=pivot.loc[resolved,first];after=pivot.loc[resolved,second];diff=after-before
            paired.append(dict(labels=int(m),first_method=first,second_method=second,total_cohorts=len(pivot),paired_draws=len(diff),unresolved_pairs=int((~resolved).sum()),finite_to_outside=int(((before==1)&(after==0)).sum()),outside_to_finite=int(((before==0)&(after==1)).sum()),finite_frequency_change=float(diff.mean()),paired_Monte_Carlo_SE=float(diff.std(ddof=1)/np.sqrt(len(diff)))))
    pd.DataFrame(paired).to_csv(a.out/'paired_transitions.csv',index=False)
    save(a.out/'run_status.json',dict(complete=True,cohorts=len(cohorts),records=len(rows),previous_cohort_hashes_matched=len(cohorts),unknowns=int(frame.status.eq('unknown').sum()),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),population_weights_sha256=hashlib.sha256((a.root/'results/weight_functions.json').read_bytes()).hexdigest(),reference_hashes_sha256=hashlib.sha256(a.reference_hashes.read_bytes()).hexdigest(),elapsed_seconds=time.monotonic()-started,new_human_or_LLM_responses=0))


if __name__=='__main__':main()
