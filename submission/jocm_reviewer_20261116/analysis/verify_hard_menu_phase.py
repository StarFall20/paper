"""Reconcile hard-menu records and directly verify saved separation directions."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from menu_corrected_logit import menu_features,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());status=json.loads((a.out/'hard_menu_matched_phase_status.json').read_text())
    assert status['state']=='complete'
    raw=pd.read_csv(a.out/'hard_menu_matched_phase_trials.csv');summary=pd.read_csv(a.out/'hard_menu_matched_phase_summary.csv')
    certificates=json.loads((a.out/'hard_menu_ordinary_separation_certificates.json').read_text())
    assert len(raw)==9000 and len(summary)==18 and not raw.duplicated(['label_menus','trial','orthogonal_slopes','method']).any()
    assert raw.groupby(['label_menus','trial']).size().eq(9).all()
    assert raw.independent_residual_status.notna().sum()==120
    assert not raw.independent_residual_status.eq('unresolved').any()
    interrupted=pd.read_csv(a.out/'hard_menu_matched_phase_interrupted_trials.csv')
    pd.testing.assert_frame_equal(raw.iloc[:len(interrupted)][interrupted.columns].reset_index(drop=True),interrupted,check_exact=True)
    frequency_error=MCSE_error=0.;dimension_cases=0;power_cases=0
    for _,row in summary.iterrows():
        part=raw[(raw.label_menus==row.label_menus)&(raw.orthogonal_slopes==row.orthogonal_slopes)&raw.method.eq(row.method)]
        assert len(part)==row.replications==500
        for state,col in [('interior','finite'),('outside','exterior'),('boundary_or_near_boundary','boundary'),('rank_deficient','rank_deficient')]:assert part.status.eq(state).sum()==row[col]
        assert part.status.isin(['unresolved','unresolved_boundary']).sum()==row.unresolved
        frequency=float(part.status.eq('interior').mean());se=float(np.sqrt(frequency*(1-frequency)/500))
        frequency_error=max(frequency_error,abs(frequency-row.finite_frequency));MCSE_error=max(MCSE_error,abs(se-row.binomial_MCSE))
    for key,part in raw.groupby(['label_menus','trial','method']):
        events=part.sort_values('orthogonal_slopes').status.eq('interior').to_numpy(int)
        assert np.all(np.diff(events)<=0),(key,events);dimension_cases+=1
    for key,part in raw[raw.power>0].groupby(['label_menus','trial','orthogonal_slopes']):
        events=part.sort_values('power',ascending=False).status.eq('interior').to_numpy(int)
        assert np.all(np.diff(events)>=0),(key,events);power_cases+=1
    assert max(frequency_error,MCSE_error)<1e-12
    pairs=[]
    for key,part in raw.groupby(['label_menus','orthogonal_slopes','dimension_ratio']):
        available=part.assign(finite=part.status.eq('interior')).pivot(index='trial',columns='method',values='finite')
        for other in ['ordinary','unit']:
            ref=available.oracle_null_slope_variance.to_numpy(int);base=available[other].to_numpy(int);difference=ref-base
            pairs.append(dict(label_menus=key[0],orthogonal_slopes=key[1],dimension_ratio=key[2],comparator=other,
                replications=len(ref),both_finite=int(np.sum((ref==1)&(base==1))),reference_only=int(np.sum((ref==1)&(base==0))),
                comparator_only=int(np.sum((ref==0)&(base==1))),neither_finite=int(np.sum((ref==0)&(base==0))),
                finite_frequency_difference=float(np.mean(difference)),paired_MCSE=float(np.std(difference,ddof=1)/np.sqrt(len(difference)))))
    pd.DataFrame(pairs).to_csv(a.out/'hard_menu_paired_availability.csv',index=False)
    lookup={(r['label_menus'],r['trial'],r['orthogonal_slopes']):r for r in certificates}
    assert len(lookup)==len(certificates)==status['ordinary_boundary_certificates']
    rng=np.random.default_rng(plan['seed']);checked=0;minimum_margin=1.;minimum_mean=1.
    for m in plan['label_menus']:
      N=int(m*plan['tau']);n=m+N;J=plan['alternatives'];dimensions=sorted({max(1,round(m*k)) for k in plan['dimension_ratios']})
      for trial in range(500):
        S=rng.normal(size=(n,J,1));W=rng.normal(size=(n,J,max(dimensions)))
        mu=softmax(plan['gamma']*S[...,0],axis=1)
        y=np.sum(rng.random(n)[:,None]>np.cumsum(mu,axis=1),axis=1)
        # Consume the exact proxy draw too,even though the human certificate uses no proxy.
        z=(y+(rng.random(n)<plan['epsilon'])*rng.integers(1,J,size=n))%J
        for q in dimensions:
            key=(m,trial,q)
            if key not in lookup:continue
            cert=lookup[key];f=menu_features(np.concatenate([S,W[...,:q]],axis=2))[:m];d=np.asarray(cert['direction'])
            margins=np.concatenate([f[i,y[i]]@d-f[i,np.arange(J)!=y[i]]@d for i in range(m)])
            assert np.sum(abs(d))<=1+1e-8 and margins.min()>-1e-8 and margins.mean()>1e-8,(key,margins.min(),margins.mean())
            assert abs(margins.mean()-cert['mean_margin'])<1e-10
            minimum_margin=min(minimum_margin,float(margins.min()));minimum_mean=min(minimum_mean,float(margins.mean()));checked+=1
    assert checked==len(certificates)
    # The three finite diagnostic boundaries have no labelled proxy errors.
    designs=json.loads((a.out/'hard_menu_retained_boundary_designs.json').read_text())
    bounds=json.loads((a.out/'hard_menu_boundary_certificates.json').read_text())
    for design,cert in zip(designs,bounds):
        f=np.asarray(design['features']);g=np.asarray(design['g']);y=np.asarray(design['y']);m=design['m'];n,J,p=f.shape
        assert design['labelled_prediction_errors']==0 and np.all(g[:m].argmax(1)==y[:m])
        targets=np.r_[y[:m],g[m:].argmax(1)];keep=np.r_[np.full(m,design['power']<1),np.ones(n-m,bool)];d=np.asarray(cert['direction'])
        margins=np.concatenate([f[i,targets[i]]@d-f[i,np.arange(J)!=targets[i]]@d for i in range(n) if keep[i]])
        assert margins.min()>1e-8 and abs(margins.min()-cert['strict_separation_margin'])<1e-10
    population=pd.read_csv(a.out/'hard_menu_population_grid.csv');variance=pd.read_csv(a.out/'hard_menu_variance_reference.csv')
    keys=['alternatives','gamma','tau','epsilon','event','power']
    ranges=population.groupby(keys,dropna=False).critical_ratio.agg(['min','max'])
    fine=population[population.bits==14].groupby(keys,dropna=False).critical_ratio.agg(['min','max'])
    assert len(population)==320 and len(ranges)==80 and len(variance)==32
    interrupted_case=json.loads((a.out/'hard_menu_interrupted_crosscheck_case.json').read_text())
    attempt=json.loads((a.out/'hard_menu_initial_phase_attempt.json').read_text())
    f=np.asarray(interrupted_case['features']);g=np.asarray(interrupted_case['g']);y=np.asarray(interrupted_case['y']);m=interrupted_case['m'];lam=interrupted_case['power']
    d=np.asarray(attempt['original_moment_state']['direction']);utility=f@d
    human=float(np.mean(np.max(utility[:m],axis=1)-utility[np.arange(m),y[:m]]))
    predL=float(np.mean(np.max(utility[:m],axis=1)-np.sum(g[:m]*utility[:m],axis=1)))
    predU=float(np.mean(np.max(utility[m:],axis=1)-np.sum(g[m:]*utility[m:],axis=1)))
    recession=human+lam*(predU-predL)
    assert recession<-1e-8 and abs(recession+attempt['original_moment_state']['separation_gap'])<1e-10
    wrong=population[population.epsilon==1].pivot_table(index=['alternatives','gamma','tau','bits','seed'],columns='event',values='critical_ratio',aggfunc='first')
    gaps=wrong.positive_limit-wrong.ordinary
    assert np.max(abs(gaps.loc[2]))<1e-8 and np.min(gaps.loc[3])>0
    result=dict(state='complete',phase_rows_reconciled=len(raw),summary_cells=len(summary),maximum_frequency_error=frequency_error,
        maximum_MCSE_error=MCSE_error,dimension_nesting_groups=dimension_cases,positive_power_nesting_groups=power_cases,
        ordinary_boundary_directions_checked=checked,minimum_ordinary_separator_margin=minimum_margin,minimum_ordinary_separator_mean=minimum_mean,
        additional_exact_diagnostic_boundaries_checked=len(bounds),hard_population_evaluations=len(population),hard_population_event_conditions=len(ranges),
        maximum_resolution_and_seed_range=float((ranges['max']-ranges['min']).max()),maximum_fine_seed_range=float((fine['max']-fine['min']).max()),
        binary_always_wrong_maximum_limit_gap=float(np.max(abs(gaps.loc[2]))),three_alternative_always_wrong_minimum_limit_gap=float(np.min(gaps.loc[3])),
        variance_reference_conditions=len(variance),unresolved=int(summary.unresolved.sum()),
        interrupted_checkpoint_rows_preserved_exactly=len(interrupted),interrupted_case_recession_cost=recession,
        residual_fallbacks=status['residual_fallbacks'],unresolved_independent_crosschecks=status['unresolved_independent_crosschecks'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Records,directions and numerical integration reconciliation; not an independent expert proof or originality assessment')
    save(a.out/'hard_menu_complete_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
