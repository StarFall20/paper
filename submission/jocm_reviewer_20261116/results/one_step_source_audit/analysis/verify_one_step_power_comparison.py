"""Independent rowwise and original-objective checks for the one-step audit."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

def rel(a,b):return abs(a-b)/(1+max(abs(a),abs(b)))

def independent_weight(t,p,j,m,N):
    if np.linalg.matrix_rank(j)<j.shape[0]:return None
    try:wt=np.linalg.solve(j,t.T).T;wp=np.linalg.solve(j,p.T).T
    except np.linalg.LinAlgError:return None
    numerator=2*np.sum((wt-wt.mean(0))*(wp[:m]-wp[:m].mean(0)))/m
    denominator=2*(1+m/N)*np.var(wp,axis=0,ddof=1).sum()
    if not np.isfinite(denominator) or denominator<=1e-14:return None
    out=float(numerator/denominator)
    return out if np.isfinite(out) else None

def main():
    p=argparse.ArgumentParser()
    for n in ['input-dir','canonical','reference','plan','results','out']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    state=json.loads((a.results/'checkpoint.json').read_text());assert state['state']=='complete' and state['records']==60000
    assert hashlib.sha256(a.plan.read_bytes()).hexdigest()==state['signatures']['plan']
    primary=a.plan.with_name('one_step_power_comparison.py')
    assert hashlib.sha256(primary.read_bytes()).hexdigest()==state['signatures']['script']
    sys.path.insert(0,str(a.canonical/'analysis'))
    from public_conjoint_optimizer_audit import load_data
    base=load_data(a.input_dir)[0][:1200];d=base.shape[1]
    assert hashlib.sha256(np.ascontiguousarray(base).tobytes()).hexdigest()==state['signatures']['support_array']
    frame=pd.read_csv(a.results/'trial_records.csv.gz');pilots=pd.read_csv(a.results/'pilot_records.csv.gz')
    keys=['study','replication','label_rows','auxiliary_rows','prediction','epsilon','method']
    assert len(frame)==60000 and not frame.duplicated(keys).any() and len(pilots)==3500
    index=frame.set_index(keys);pilot_index=pilots.set_index(['study','replication','label_rows']);beta_cols=[f'beta{j}' for j in range(d)]
    plan=json.loads(a.plan.read_text());worst=dict(pilot_score=0.,raw_weight_relative_error=0.,update_backward_error=0.,MSE_relative_error=0.,loss_relative_error=0.)
    checked=0;pilot_checked=0;negative=0;outside=0;unavailable=0
    for study in plan['studies']:
        name=study['name'];truth=np.array(json.loads((a.reference/(name+'_DGP.json')).read_text())['truth']);risk_truth=np.mean(np.logaddexp(0,base@truth)-expit(base@truth)*(base@truth));prob=expit(base@truth)
        rng=np.random.default_rng(study['seed'])
        for rep in range(500):
            lp=base[rng.integers(0,len(base),150)];up=base[rng.integers(0,len(base),600)]
            yl=(rng.random(150)<expit(lp@truth)).astype(float);yu=(rng.random(600)<expit(up@truth)).astype(float)
            fl=rng.random(150);fu=rng.random(600)
            for m in study['labels']:
                pilot=pilot_index.loc[(name,rep,m),beta_cols].to_numpy(float)
                pg=lp[:m].T@(expit(lp[:m]@pilot)-yl[:m])/m+pilot/m
                worst['pilot_score']=max(worst['pilot_score'],float(abs(pg).max()));assert abs(pg).max()<1e-7
                pilot_checked+=1
                for N in study['auxiliary']:
                    xl=lp[:m];xu=up[:N];x=np.vstack([xl,xu]);mu=expit(x@pilot);vl=mu[:m]*(1-mu[:m]);vu=mu[m:]*(1-mu[m:])
                    al=xl*np.sqrt(vl)[:,None];au=xu*np.sqrt(vu)[:,None];hl=al.T@al/m;hu=au.T@au/N
                    hp=(al.T@al+au.T@au)/(m+N)
                    settings=[('oracle',e,1.,0.) for e in study['proxy_errors']] if name=='finite_pool' else [('oracle',.25,1.,0.),('overconfident',.25,2.,0.),('intercept_bias',.25,1.,1.)]
                    for prediction,e,slope,bias in settings:
                        z=np.r_[np.logical_xor(yl[:m],fl[:m]<e),np.logical_xor(yu[:N],fu[:N]<e)].astype(float)
                        g=expit(slope*(x@truth+(2*z-1)*np.log((1-e)/e))+bias)
                        t=xl*(mu[:m]-yl[:m])[:,None];sur=x*(mu-g)[:,None]
                        powers={'pooled':independent_weight(t,sur,hp,m,N),'labelled':independent_weight(t,sur,hl,m,N)}
                        for method in plan['methods']:
                            row=index.loc[(name,rep,m,N,prediction,e,method)];checked+=1
                            if method=='human_ridge':
                                assert row.available
                                assert np.max(abs(row[beta_cols].to_numpy(float)-pilot))<1e-12
                            else:
                                raw=powers[method.split('_')[0]]
                                if raw is None:
                                    assert not row.available and pd.isna(row.raw_weight);unavailable+=1;continue
                                weight=raw if method.endswith('_unclipped') else float(np.clip(raw,0,1))
                                worst['raw_weight_relative_error']=max(worst['raw_weight_relative_error'],rel(raw,float(row.raw_weight)))
                                assert rel(weight,float(row.weight))<2e-7
                                if not row.available:unavailable+=1;continue
                                beta=row[beta_cols].to_numpy(float);delta=pilot-beta
                                h=(1-weight)*hl+weight*hu
                                gradient=xl.T@(mu[:m]-yl[:m])/m+weight*(xu.T@(mu[m:]-g[m:])/N-xl.T@(mu[:m]-g[:m])/m)
                                residual=np.linalg.norm(h@delta-gradient,np.inf)
                                scale=np.linalg.norm(h,np.inf)*np.linalg.norm(delta,np.inf)+np.linalg.norm(gradient,np.inf)
                                backward=float(residual/max(scale,np.finfo(float).tiny))
                                worst['update_backward_error']=max(worst['update_backward_error'],backward);assert backward<2e-8
                                assert bool(row.outside_unit_interval)==bool(weight<0 or weight>1)
                                negative+=int(np.linalg.eigvalsh(h)[0]<0);outside+=int(weight<0 or weight>1)
                            beta=row[beta_cols].to_numpy(float)
                            mse=float(np.dot(beta-truth,beta-truth)/d)
                            loss=float(np.mean(np.logaddexp(0,base@beta)-prob*(base@beta))-risk_truth)
                            worst['MSE_relative_error']=max(worst['MSE_relative_error'],rel(mse,float(row.MSE)))
                            worst['loss_relative_error']=max(worst['loss_relative_error'],rel(loss,float(row.excess_logloss)))
            if (rep+1)%100==0:print(json.dumps({'study':name,'replications_checked':rep+1,'records_checked':checked}),flush=True)
    assert checked==60000 and pilot_checked==3500
    assert worst['raw_weight_relative_error']<2e-7 and worst['MSE_relative_error']<1e-9 and worst['loss_relative_error']<1e-9
    comparisons=[]
    for study in plan['studies']:
        name=study['name'];original=pd.read_csv(a.reference/(name+'_summary.csv'))
        for _,r in original[original.method.eq('PPI_one_step_ridge')].iterrows():
            s=frame[frame.study.eq(name)&frame.method.eq('pooled_clipped')&frame.label_rows.eq(r.label_rows)&frame.auxiliary_rows.eq(r.auxiliary_rows)]
            s=s[s.epsilon.eq(r.epsilon)] if name=='finite_pool' else s[s.prediction.eq(r.prediction)]
            assert len(s)==500 and int(s.available.sum())==int(r.estimates)
            rm=rel(float(s.MSE.mean()),float(r.MSE));rl=rel(float(s.excess_logloss.mean()),float(r.excess_logloss))
            comparisons.append(dict(study=name,label_rows=int(r.label_rows),auxiliary_rows=int(r.auxiliary_rows),prediction='oracle' if name=='finite_pool' else r.prediction,epsilon=float(r.epsilon) if name=='finite_pool' else .25,replications=len(s),estimates=int(s.available.sum()),original_MSE=float(r.MSE),reconstructed_MSE=float(s.MSE.mean()),MSE_relative_error=rm,original_excess_logloss=float(r.excess_logloss),reconstructed_excess_logloss=float(s.excess_logloss.mean()),loss_relative_error=rl))
    pd.DataFrame(comparisons).to_csv(a.out/'all_24_legacy_cells.csv',index=False)
    assert len(comparisons)==24 and max(max(r['MSE_relative_error'],r['loss_relative_error']) for r in comparisons)<1e-7
    report=dict(state='passed',records_checked=checked,pilot_roots_checked=pilot_checked,legacy_cells_reconciled=len(comparisons),legacy_counts_exact=True,worst=worst,outside_domain_updates=outside,negative_curvature_updates=negative,unavailable_records=unavailable,new_independent_cohorts=0,new_human_or_AI_answers=0,finite_global_roots_claimed=False,formal_PPI_publication_reconciliation_complete=False,verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.out/'verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
