"""Reconcile matched records and independently check their saved directions."""
from pathlib import Path
from fractions import Fraction
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.special import expit
from threadpoolctl import threadpool_limits


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--population',type=Path,required=True);p.add_argument('--certificate',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());state=json.loads((a.out/'variance_reference_phase_status.json').read_text());assert state['state']=='complete'
    assert state['plan_sha256']==hashlib.sha256(a.plan.read_bytes()).hexdigest()
    assert state['source_sha256']==hashlib.sha256(Path(__file__).with_name('variance_reference_matched_phase.py').read_bytes()).hexdigest()
    raw=pd.read_csv(a.out/'variance_reference_phase_trials.csv');summary=pd.read_csv(a.out/'variance_reference_phase_summary.csv');pairs=pd.read_csv(a.out/'variance_reference_phase_pairs.csv')
    assert len(raw)==16000 and len(summary)==32 and len(pairs)==24 and raw.groupby(['label_rows','trial']).size().eq(16).all()
    assert not raw.duplicated(['label_rows','trial','orthogonal_slopes','method']).any()
    assert raw.status.isin(['interior','outside_mean_support','boundary_or_near_boundary']).all()
    assert state['exterior_directions_checked']==raw.status.eq('outside_mean_support').sum()
    assert raw.independent_residual_status.notna().sum()==state['independent_residual_checks']==240
    assert not raw.independent_residual_status.eq('unresolved').any()
    for r in summary.itertuples():
        part=raw[(raw.label_rows==r.label_rows)&(raw.orthogonal_slopes==r.orthogonal_slopes)&raw.method.eq(r.method)]
        assert len(part)==r.replications==500 and part.status.eq('interior').sum()==r.finite
        assert part.status.eq('outside_mean_support').sum()==r.outside and part.status.eq('boundary_or_near_boundary').sum()==r.boundary
        f=float(part.status.eq('interior').mean());assert abs(f-r.finite_frequency)<1e-12 and abs(np.sqrt(f*(1-f)/500)-r.binomial_MCSE)<1e-12
    for r in pairs.itertuples():
        part=raw[(raw.label_rows==r.label_rows)&(raw.orthogonal_slopes==r.orthogonal_slopes)]
        available=part.assign(finite=part.status.eq('interior')).pivot(index='trial',columns='method',values='finite')
        d=available[r.reference].to_numpy(int)-available[r.comparator].to_numpy(int)
        assert abs(d.mean()-r.finite_frequency_difference)<1e-12 and abs(d.std(ddof=1)/np.sqrt(500)-r.paired_MCSE)<1e-12
        assert ((d==1).sum(),(d==-1).sum())==(r.reference_only,r.comparator_only)
    for _,part in raw.groupby(['label_rows','trial','method']):assert np.all(np.diff(part.sort_values('orthogonal_slopes').status.eq('interior').to_numpy(int))<=0)
    for _,part in raw[raw.method.str.startswith('shrunk')].groupby(['label_rows','trial','orthogonal_slopes']):assert np.all(np.diff(part.sort_values('power',ascending=False).status.eq('interior').to_numpy(int))>=0)
    directions=json.loads((a.out/'ordinary_separation_directions.json').read_text());examples=json.loads((a.out/'discordant_examples.json').read_text())
    lookup={(r['label_rows'],r['trial'],r['orthogonal_slopes']):r for r in directions};sample={(r['label_rows'],r['trial'],r['orthogonal_slopes']):r for r in examples}
    assert len(lookup)==state['ordinary_directions']==raw[(raw.method=='ordinary')&raw.status.eq('boundary_or_near_boundary')].shape[0]
    rng=np.random.default_rng(plan['seed']);checked=0;examples_checked=0;score_error=0.;minimum_recession_gap=np.inf;score_groups=0
    for m in plan['label_rows']:
      N=round(m*plan['tau']);n=m+N;dimensions=sorted({round(m*k) for k in plan['dimension_ratios']})
      for trial in range(plan['replications']):
        s=rng.normal(size=n);orth=rng.normal(size=(n,max(dimensions)));mu=expit(plan['gamma']*s)
        y=(rng.random(n)<mu).astype(float);z=np.logical_xor(y,rng.random(n)<plan['epsilon']).astype(float)
        g0=expit(plan['gamma']*s+(2*z-1)*np.log((1-plan['epsilon'])/plan['epsilon']));gs=.5*(mu+g0)
        for q in dimensions:
            x=np.column_stack([np.ones(n),s,orth[:,:q]]);key=(m,trial,q)
            truth1=x[:m].T@(.5*mu[:m]-y[:m]+.5*g0[:m])/m+.5*x[m:].T@(mu[m:]-g0[m:])/N
            truth2=x[:m].T@(gs[:m]-y[:m])/m+x[m:].T@(mu[m:]-gs[m:])/N
            score_error=max(score_error,float(np.max(abs(truth1-truth2))));score_groups+=1
            if key in lookup:
                cert=lookup[key];d=np.array(cert['direction']);margins=(2*y[:m]-1)*(x[:m]@d)
                assert np.sum(abs(d))<=1+1e-8 and margins.min()> -1e-8 and margins.mean()>1e-8
                assert abs(margins.mean()-cert['mean_margin'])<1e-10;checked+=1
            if key not in sample:continue
            cert=sample[key];d=np.array(cert['shrunk_reference_direction'])
            moment=x[:m].T@(y[:m]-gs[:m])/m+x[m:].T@gs[m:]/N
            gap=float(moment@d-np.maximum(x[m:]@d,0).mean());assert gap>1e-8 and abs(gap-cert['shrunk_reference_recession_gap'])<1e-10
            minimum_recession_gap=min(minimum_recession_gap,gap)
            raw_moment=x[:m].T@(y[:m]-.5*g0[:m])/m+.5*x[m:].T@g0[m:]/N
            w=np.r_[np.full(m,.5/m),np.full(N,.5/N)];margin=cert['oracle_reference_interior_margin']/2
            fit=linprog(np.zeros(n),A_eq=(w[:,None]*x).T,b_eq=raw_moment,bounds=(margin,1-margin),method='highs-ipm')
            assert fit.success and np.max(abs((w[:,None]*x).T@fit.x-raw_moment))<1e-7
            beta0=np.r_[0.,plan['gamma'],np.zeros(q)]
            score1=x.T@(w*expit(x@beta0))-raw_moment;score2=x[m:].T@mu[m:]/N-moment
            score_error=max(score_error,float(np.max(abs(score1-score2))));examples_checked+=1
    assert checked==len(directions) and examples_checked==len(examples)==24 and score_error<1e-10 and score_groups==4000
    pop=pd.read_csv(a.population/'variance_reference_population_primary.csv');null=pop[(pop.gamma==0)&(pop.epsilon==.1)&(pop.tau==1)]
    cert=json.loads(a.certificate.read_text());upper=Fraction(cert['shrunk_boundary_upper_rational']);lower=Fraction(cert['oracle_boundary_lower_rational'])
    assert upper<Fraction(47,100)<Fraction(1,2)<lower
    assert float(null[null.prediction_strength==.5].critical_ratio.iloc[0])<float(upper)
    assert float(null[null.prediction_strength==1].critical_ratio.iloc[0])>float(lower)
    result=dict(state='complete',records_reconciled=len(raw),summary_cells=len(summary),paired_comparisons=len(pairs),
        dimension_nesting_groups=4000,shrunk_positive_power_nesting_groups=4000,independent_residual_checks=240,
        ordinary_separator_directions_replayed=checked,discordant_designs_independently_certified=examples_checked,
        truth_score_groups_replayed=score_groups,truth_score_maximum_error=score_error,minimum_checked_exterior_gap=minimum_recession_gap,
        rational_counterexample_bounds_reconciled=True,remaining_unknowns=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Stored result reconciliation,direct ray checks and an independent interior feasibility program. Does not replay all16000original classification LPs or assess originality.')
    (a.out/'variance_reference_complete_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
