"""Independent original-row checks for the zero-preserving control."""
from pathlib import Path
import argparse,gzip,hashlib,json
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from existing_panel_model import load
from verify_frontier import original
BASE=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main(path,frontiers,results,out):
    assert not out.exists()
    status=json.loads((results/'status.json').read_text());assert status['state']=='complete'
    assert status['amendment_sha256']==sha(BASE/'cap_only_amendment.json')
    assert status['primary_frontiers_sha256']==sha(frontiers/'frontiers.csv')
    assert status['analysis_sha256']==sha(BASE/'analysis/run_cap_only.py') and status['source_sha256']==sha(path)
    plan=json.loads((BASE/'inputs/model_plan.json').read_text());data=load(path,plan)
    pre=json.loads((BASE/'inputs/preflight.json').read_text());tu=data['teacher']@np.array(pre['teacher_coefficients'])
    g=np.exp(tu-logsumexp(tu,axis=2,keepdims=True))
    fr=pd.read_csv(frontiers/'frontiers.csv');rows=pd.read_csv(results/'records.csv')
    old=pd.read_csv(BASE/'inputs/baseline_records.csv')
    coef={r['record']:np.array(r['beta']) for r in json.loads((results/'coefficients.json').read_text())}
    saved={r['record']:np.array(r['beta']) for r in json.loads((BASE/'inputs/cluster_coefficients.json').read_text())}
    witness=json.loads(gzip.decompress((results/'interior_witnesses.json.gz').read_bytes()))
    assert len(rows)==len(witness)==2000 and len(coef)==status['available_coefficients']
    assert rows.power.between(0,1).all()
    rng=np.random.default_rng(plan['allocation_seed']);orders=[rng.permutation(data['pool']) for _ in range(500)]
    err=dict(moment=0.,score=0.,metric=0.);reuse=new=0
    for r in rows.itertuples(index=False):
        cap=fr.iloc[r.frontier_record].frontier
        expected=min(r.original_power,.9*cap) if cap>1e-10 else r.original_power
        assert abs(r.power-expected)<1e-14
        if r.original_power==0:assert r.power==0
        order=orders[r.allocation];L,U=order[:r.labelled_people],order[r.labelled_people:]
        if r.support=='interior':
            w=witness[r.record];p=np.full((34,3),1/3);p[np.array(w['kept_designs'])]=np.array(w['probabilities'])
            assert np.min(p)>1e-8-1e-9 and np.max(abs(p.sum(1)-1))<1e-7
            moment=original(data,L,U,g,r.power,probability=p)['moment_error'];assert moment<1e-7;err['moment']=max(err['moment'],moment)
            people=L if r.power==0 else np.r_[L,U];f=data['f'][people].reshape(-1,3,11)
            assert np.linalg.matrix_rank((f[:,:-1]-f[:,-1:]).reshape(-1,11))==11
        if r.record not in coef:continue
        beta=coef[r.record]
        if r.coefficient_source=='verified_original_objective':
            assert not r.power_changed and np.array_equal(beta,saved[r.original_record]);reuse+=1
        else:assert r.power_changed;new+=1
        score=original(data,L,U,g,r.power,beta=beta)['score'];assert score<1.01e-7;err['score']=max(err['score'],score)
        z=data['f'][data['test']]@beta;lp=z-logsumexp(z,axis=2,keepdims=True);p=np.exp(lp);y=data['y'][data['test']]
        loss=float(-np.take_along_axis(lp,y[:,:,None],axis=2).mean())
        brier=float((1+(p*p).sum(2)-2*np.take_along_axis(p,y[:,:,None],axis=2)[:,:,0]).mean())
        err['metric']=max(err['metric'],abs(loss-r.heldout_logloss),abs(brier-r.heldout_Brier));assert err['metric']<1e-10
    assert reuse==status['reused_verified_coefficients'] and new==status['new_model_fits']
    for s in json.loads((results/'summary.json').read_text()):
        group=rows[rows.labelled_people.eq(s['labelled_people'])];assert len(group)==s['denominator']==500
        assert group.fit.eq('passed_score').sum()==s['available'] and group.support.value_counts().to_dict()==s['support_counts']
        pivot=old[old.labelled_people.eq(s['labelled_people'])].pivot(index='allocation',columns='method',values='heldout_logloss')
        vals=group.set_index('allocation').heldout_logloss.reindex(pivot.index).fillna(pivot.labelled_ridge)
        assert abs(vals.mean()-s['mean_fallback_loss'])<1e-12
        for b,c in s['comparisons'].items():
            d=vals-pivot[b].fillna(pivot.labelled_ridge)
            assert abs(d.mean()-c['mean'])<1e-12 and abs(d.std(ddof=1)/np.sqrt(500)-c['allocation_MCSE'])<1e-12
    out.mkdir(parents=True)
    record=dict(state='passed',policy_records=2000,finite_coefficients=len(coef),original_coefficients_reused=reuse,new_fits=new,
                maximum_errors=err,zero_powers_preserved=True,shared_original_row_verification_helper=True,
                primary_cap_only_rule_fitter_and_metrics_not_imported=True,verifier_sha256=sha(Path(__file__)),
                result_hashes={f.name:sha(f) for f in results.iterdir() if f.is_file()})
    (out/'verification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--frontiers',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):main(a.input,a.frontiers,a.results,a.out)
