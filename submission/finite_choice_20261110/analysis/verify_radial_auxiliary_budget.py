"""Independently replay nested auxiliary experiments in the original loss.

The experiment's residual LP and draw-construction functions are not called.
The established original-objective attainable-moment implementation is reused.
Every distinct criterion is checked; duplicated low-ratio arms remain labelled.
"""
from pathlib import Path
import argparse, hashlib, json, math, time
import numpy as np
import pandas as pd
from power_tuned_choice_audit import objective, support
from verify_radial_compensation import independent_normal_box


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def reconstruct(plan, m, rep):
    q=int(round(plan['noise_label_ratio']*m))
    e=plan['epsilon']; p=plan['large_scale_probability']; small=plan['small_scale']
    large=math.sqrt((1-(1-p)*small**2)/p)
    rng=np.random.default_rng(np.random.SeedSequence([plan['source_seed'],m,rep]))
    base=np.concatenate([np.ones((2*m,1)),rng.standard_normal((2*m,q+1))],axis=1)
    y=rng.integers(0,2,size=2*m).astype(float)
    z=np.not_equal(y,rng.uniform(size=2*m)<e).astype(float)
    g=e+(1-2*e)*z
    scale=np.where(rng.uniform(size=2*m)<p,large,small)
    original=(base.copy(),y.copy(),g.copy(),scale.copy())
    extra=int(round(max(plan['auxiliary_ratios'])*m))-m
    rng=np.random.default_rng(np.random.SeedSequence([plan['extension_seed'],m,rep]))
    if extra:
        added=np.concatenate([np.ones((extra,1)),rng.standard_normal((extra,q+1))],axis=1)
        ay=rng.integers(0,2,size=extra).astype(float)
        az=np.not_equal(ay,rng.uniform(size=extra)<e).astype(float)
        ag=e+(1-2*e)*az
        ad=np.where(rng.uniform(size=extra)<p,large,small)
        base=np.concatenate([base,added]); y=np.r_[y,ay]; g=np.r_[g,ag]; scale=np.r_[scale,ad]
    full=(base,y,g,scale)
    assert all(np.array_equal(a,b[:2*m]) for a,b in zip(original,full))
    digest=hashlib.sha256()
    for a in full:
        digest.update(a[:m].tobytes())
    return full,digest.hexdigest()


def integrals(population,plan):
    rows=[];e=plan['epsilon'];p=plan['large_scale_probability'];a=plan['small_scale']
    b=math.sqrt((1-(1-p)*a*a)/p)
    for row in population.itertuples(index=False):
        scales=[(1.,1.)] if row.model=='constant_scale' else [(a,1-p),(b,p)]
        v=row.scaled_minimizer; lam=row.power; tau=row.auxiliary_ratio
        total=error=0.
        for d,mass in scales:
            for residual,probability in [(e,1-e),(1-e,e)]:
                lo=d*v*residual; hi=lo+d*v*(1-lam)/lam
                value,err=independent_normal_box(lo,hi)
                total+=mass*probability*value;error+=mass*probability*err
            value,err=independent_normal_box(-d*v*(1-e)/tau,d*v*e/tau)
            total+=tau*mass*value;error+=tau*mass*err
        discrepancy=abs(total-row.critical_ratio)
        assert discrepancy<1e-8,(row,discrepancy)
        rows.append(dict(model=row.model,auxiliary_ratio=tau,rule=row.rule,power=lam,
                         direct_normal_integral=total,formula_discrepancy=discrepancy,
                         quadrature_error_estimate=error))
    for _,part in population[population.rule=='held_power'].groupby('model'):
        assert (np.diff(part.sort_values('auxiliary_ratio').critical_ratio)>=-1e-10).all()
    return rows


def paired(frame):
    rows=[]
    for (m,model,rule),part in frame.groupby(['labels','model','rule'],sort=False):
        table=part.pivot(index='replication',columns='auxiliary_ratio',values='status')
        ratios=sorted(table.columns)
        for low,high in [(ratios[0],ratios[1]),(ratios[1],ratios[2]),(ratios[0],ratios[2])]:
            before=table[low].eq('feasible').astype(int)
            after=table[high].eq('feasible').astype(int)
            diff=after-before
            rows.append(dict(labels=int(m),model=model,rule=rule,lower_ratio=low,higher_ratio=high,
                paired_draws=len(diff),finite_to_outside=int(((before==1)&(after==0)).sum()),
                outside_to_finite=int(((before==0)&(after==1)).sum()),
                finite_frequency_change=float(diff.mean()),
                paired_Monte_Carlo_SE=float(diff.std(ddof=1)/np.sqrt(len(diff)))))
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    assert not (args.out/'verification.json').exists(),'Use a fresh output directory.'
    plan=json.loads(args.plan.read_text())
    trials=pd.read_csv(args.root/'results/nested_auxiliary_trials.csv')
    assert len(trials)==plan['expected_reported_criterion_records']
    assert not trials.duplicated(['labels','replication','model','auxiliary_ratio','rule']).any()
    population=pd.read_csv(args.root/'results/population_budget_boundaries.csv')
    independent_integrals=integrals(population,plan)
    pd.DataFrame(independent_integrals).to_csv(args.out/'independent_population_integrals.csv',index=False)
    reference=pd.read_csv(args.reference)
    reference=reference[(reference.power==.5)&(reference.criterion=='compensated_residual_support')]
    reference=reference.set_index(['labels','replication','model']).status.to_dict()
    checks=[];rays=[];hashes=[];programs=matched=0;start=time.monotonic()
    for (m,rep),part in trials.groupby(['labels','replication'],sort=True):
        m=int(m);rep=int(rep)
        (base,y,g,scale),human_hash=reconstruct(plan,m,rep)
        hashes.append(dict(labels=m,replication=rep,human_arrays_sha256=human_hash))
        cache={}
        for index,row in part.iterrows():
            N=int(row.auxiliary_rows);lam=N/(m+N) if row.rule=='variance_rule' else 1/41
            assert abs(lam-row.power)<1e-15
            key=(row.model,N,lam)
            if key not in cache:
                used=m+N;x=base[:used] if row.model=='constant_scale' else base[:used]*scale[:used,None]
                loss=objective(x,y[:used],g[:used],m,lam)
                result=support(loss);programs+=1
                status='feasible' if result['status']=='interior' else 'outside' if result['status']=='outside_mean_support' else 'unknown'
                if 'direction' in result:
                    direction=np.array(result['direction'])
                    recession=float(loss.weights@np.maximum(loss.x@direction,0)-loss.moment@direction)
                    assert recession<-1e-8
                    rays.append(dict(row_index=int(index),recession_cost=recession,direction=direction.tolist()))
                cache[key]=(result,status,int(index))
            result,status,source_index=cache[key]
            checks.append(dict(row_index=int(index),distinct_program_first_row=source_index,
                independent_status=result['status'],verified_status=status,
                interior_margin=result.get('interior_margin'),separation_gap=result.get('separation_gap'),
                original_status=row.status))
            if row.auxiliary_ratio==1 and row.rule=='variance_rule':
                assert status==reference[m,rep,row.model];matched+=1
            if status!=row.status or status=='unknown':
                pd.DataFrame(checks).to_csv(args.out/'independent_support_replays.csv',index=False)
                save(args.out/'verification_failure.json',dict(row_index=int(index),record=row.to_dict(),result=result))
                raise AssertionError((index,row.status,result))
        if (rep+1)%100==0:
            pd.DataFrame(checks).to_csv(args.out/'independent_support_replays.csv',index=False)
            print('independent nested replay',m,rep+1,'records',len(checks),'programs',programs,flush=True)
    assert programs==plan['expected_distinct_compensated_programs']
    assert len(checks)==len(trials)
    pd.DataFrame(checks).to_csv(args.out/'independent_support_replays.csv',index=False)
    pd.DataFrame(hashes).to_csv(args.out/'fixed_human_draw_hashes.csv',index=False)
    save(args.out/'negative_recession_directions.json',rays)
    pd.DataFrame(paired(trials)).to_csv(args.out/'paired_auxiliary_transitions.csv',index=False)
    summaries=[]
    for key,part in trials.groupby(['labels','model','auxiliary_ratio','rule'],sort=False):
        counts=part.status.value_counts().to_dict()
        summaries.append(dict(zip(['labels','model','auxiliary_ratio','rule'],key))|dict(
            power=float(part.power.iloc[0]),replications=len(part),feasible=counts.get('feasible',0),
            outside=counts.get('outside',0),unknown=counts.get('unknown',0),
            finite_frequency=counts.get('feasible',0)/len(part)))
    summary=pd.DataFrame(summaries)
    expected=pd.read_csv(args.root/'results/nested_auxiliary_summary.csv')
    pd.testing.assert_frame_equal(summary,expected,check_exact=False,atol=1e-15,rtol=1e-15)
    summary.to_csv(args.out/'nested_auxiliary_summary_verified.csv',index=False)
    p=plan['large_scale_probability'];small=plan['small_scale'];e=plan['epsilon']
    mean_scale=(1-p)*small+p*math.sqrt((1-(1-p)*small*small)/p)
    limit=mean_scale**2/(2*math.pi*e*(1-e))
    save(args.out/'verification.json',dict(date='2026-11-12',reported_records=len(checks),
        distinct_original_objective_programs=programs,duplicated_low_ratio_records=2000,
        independently_checked_negative_recession_directions=len(rays),
        preserved_human_cohorts=len(hashes),matched_previous_ratio_one_states=matched,
        independent_population_integrals=len(independent_integrals),
        maximum_integral_discrepancy=max(x['formula_discrepancy'] for x in independent_integrals),
        maximum_quadrature_error_estimate=max(x['quadrature_error_estimate'] for x in independent_integrals),
        model_state_mismatches=0,unknowns=0,
        scale_mixture_infinite_auxiliary_variance_rule_boundary=limit,
        fixed_power_population_ordering_verified=True,
        elapsed_seconds=time.monotonic()-start,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        original_trials_sha256=hashlib.sha256((args.root/'results/nested_auxiliary_trials.csv').read_bytes()).hexdigest(),
        priority_or_submission_certified=False))


if __name__=='__main__':main()
