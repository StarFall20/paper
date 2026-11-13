"""Replay every selected-illustration objective using actual weighted logit.

The residual classifier and draw-construction functions in the generator are
not called. Ordinary boundaries receive explicit human separating directions.
"""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from power_tuned_choice_audit import WeightedLogit,support


def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def reconstruct(plan,m,rep,d,p):
    q0=round(m*plan['original_noise_label_ratio']);q=round(m*plan['noise_label_ratio']);n=int(round(m*plan['selected_cell_details']['auxiliary_ratio']))
    rng=np.random.default_rng(np.random.SeedSequence([plan['source_seed'],m,rep]));old=np.concatenate([np.ones((2*m,1)),rng.standard_normal((2*m,q0+1))],axis=1);y=rng.integers(0,2,2*m).astype(float)
    z=np.not_equal(y,rng.uniform(size=2*m)<plan['epsilon']).astype(float);g=plan['epsilon']+(1-2*plan['epsilon'])*z;uniforms=rng.uniform(size=2*m)
    probability=plan['original_large_probability'];small=plan['original_small_scale'];large=math.sqrt((1-(1-probability)*small**2)/probability);old_d=np.where(uniforms<probability,large,small)
    original=hashlib.sha256()
    for arr in [old,y,g,old_d]:original.update(arr.tobytes())
    extra_rng=np.random.default_rng(np.random.SeedSequence([plan['extra_attribute_seed'],m,rep]));base=np.concatenate([old,extra_rng.standard_normal((2*m,q-q0))],axis=1)
    # Count CDF cut points below each uniform, independently of searchsorted.
    index=(uniforms[:,None]>=np.cumsum(p)[None,:-1]).sum(axis=1);scale=np.take(d,index)
    stop=m+n;base=base[:stop];y=y[:stop];g=g[:stop];scale=scale[:stop];index=index[:stop]
    current=hashlib.sha256()
    for arr in [base,y,g,scale]:current.update(arr.tobytes())
    return base,scale,index,y,g,original.hexdigest(),current.hexdigest()


def weighted_loss(x,y,g,w,m,lam):
    n=len(y)-m;curvature=w*np.concatenate([np.full(m,(1-lam)/m),np.full(n,lam/n)])
    target=x[:m].T@(w[:m]*(y[:m]-lam*g[:m]))/m+lam*x[m:].T@(w[m:]*g[m:])/n
    normalizer=float(curvature.sum())
    return WeightedLogit(x,curvature/normalizer,target/normalizer),normalizer


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--reference-hashes',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'verification.json').exists(),'Use a fresh output directory.'
    plan=json.loads(a.plan.read_text());frame=pd.read_csv(a.root/'phase/phase_trials.csv');cohorts=pd.read_csv(a.root/'phase/source_cohort_hashes.csv').set_index(['labels','replication']);reference=pd.read_csv(a.reference_hashes).set_index(['labels','replication']).source_draw_sha256
    status=json.loads((a.root/'phase/run_status.json').read_text());assert status['complete'] and len(frame)==plan['expected_method_records']==5000
    assert status['source_sha256']==hashlib.sha256((Path(__file__).parent/'budgeted_attainment_phase.py').read_bytes()).hexdigest()
    assert status['plan_sha256']==hashlib.sha256(a.plan.read_bytes()).hexdigest()
    assert status['population_weights_sha256']==hashlib.sha256((a.root/'results/weight_functions.json').read_bytes()).hexdigest()
    selected=next(row for row in json.loads((a.root/'results/weight_functions.json').read_text()) if row['cell_id']==plan['selected_cell']);d=np.array(selected['scales']);mass=np.array(selected['probabilities']);wopt=np.array(selected['optimized_row_weights']);exponent=selected['power_exponent'];lam=plan['selected_cell_details']['power']
    verified=frame.copy();verified['initial_status']=verified.status;records=[];rays=[];separators=[]
    for (m,rep),part in frame.groupby(['labels','replication'],sort=True):
        m=int(m);rep=int(rep);base,scale,index,y,g,original_digest,current_digest=reconstruct(plan,m,rep,d,mass)
        assert original_digest==reference.loc[m,rep]==cohorts.loc[(m,rep),'previous_source_sha256']
        assert current_digest==cohorts.loc[(m,rep),'current_design_sha256'];x=base*scale[:,None]
        for row_index,row in part.iterrows():
            method=row.method;power=0. if method=='human_only' else lam
            if method in ['human_only','unweighted_compensation']:w=np.ones(len(y))
            elif method=='power_at_budget':w=scale**(-exponent)
            elif method=='optimized_at_budget':w=wopt[index]
            elif method=='inverse_scale':w=1/scale
            else:raise AssertionError(method)
            loss,c=weighted_loss(x,y,g,w,m,power);result=support(loss)
            if result['status']=='interior':state='human_finite' if method=='human_only' else 'feasible'
            elif method=='human_only' and result['status']=='boundary_or_near_boundary':
                signed=(2*y[:m]-1)[:,None]*x[:m]
                ray=linprog(np.zeros(x.shape[1]),A_ub=-signed,b_ub=-np.ones(m),bounds=[(None,None)]*x.shape[1],method='highs')
                assert ray.success and min(signed@ray.x)>1-1e-7
                separators.append(dict(row_index=int(row_index),minimum_signed_margin=float(min(signed@ray.x)),direction=ray.x.tolist()));state='human_separated'
            elif method!='human_only' and result['status']=='outside_mean_support':state='outside'
            else:raise AssertionError((int(row_index),method,result))
            if row.status!='unknown':assert row.status==state,(row_index,row.status,state)
            verified.loc[row_index,'status']=state;records.append(dict(row_index=int(row_index),verified_status=state,actual_weighted_support=result['status'],interior_margin=result.get('interior_margin'),separation_gap=result.get('separation_gap'),common_normalizer=c))
            if 'direction' in result:
                direction=np.array(result['direction']);cost=float(loss.weights@np.maximum(x@direction,0)-loss.moment@direction);assert cost<-1e-8
                rays.append(dict(row_index=int(row_index),normalized_recession_cost=cost,unnormalized_recession_cost=c*cost,direction=direction.tolist()))
        if (rep+1)%100==0:print('budget original-loss verification',m,rep+1,'records',len(records),flush=True)
    verified.to_csv(a.out/'phase_trials_verified.csv',index=False);pd.DataFrame(records).to_csv(a.out/'original_loss_checks.csv',index=False);save(a.out/'negative_recession_directions.json',rays);save(a.out/'human_separators.json',separators)
    paired=pd.read_csv(a.root/'phase/paired_transitions.csv')
    for row in paired.itertuples(index=False):
        table=verified[verified.labels==row.labels].pivot(index='replication',columns='method',values='status').isin(['feasible','human_finite']).astype(int);before=table[row.first_method];after=table[row.second_method];diff=after-before
        assert row.finite_to_outside==int(((before==1)&(after==0)).sum());assert row.outside_to_finite==int(((before==0)&(after==1)).sum())
        assert abs(row.finite_frequency_change-diff.mean())<1e-14 and abs(row.paired_Monte_Carlo_SE-diff.std(ddof=1)/np.sqrt(len(diff)))<1e-14
    save(a.out/'verification.json',dict(date='2026-11-13',all_actual_objectives_checked=len(records),previous_and_extended_cohorts_checked=len(cohorts),negative_recession_directions=len(rays),human_strict_separators=len(separators),paired_transition_rows_checked=len(paired),original_unknowns=int(frame.status.eq('unknown').sum()),final_unknowns=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),priority_or_submission_certified=False))


if __name__=='__main__':main()
