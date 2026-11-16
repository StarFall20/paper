"""Check every reused all-power witness against independently rebuilt cohorts."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from verify_learned_budget import read_records,interval_status


def run(base,audit,out):
    out.mkdir(parents=True,exist_ok=True);plan=json.loads((base/'plan.json').read_text())
    assert plan['phase_auxiliary_rows_per_total_label']==1
    df=pd.read_csv(audit/'trials.csv');assert len(df)==4000
    rays=read_records(audit/'directions.json');rays={(r['condition'],r['trial']):np.array(r['direction']) for r in rays}
    checked=negative=0;maxh=maxa=maxidentity=0.
    for r in df.to_dict('records'):
        B=int(r['total_labels']);d=int(r['slopes']);t=int(r['training_labels']);trial=int(r['trial']);m=B-t;N=B
        rng=np.random.default_rng(np.random.SeedSequence([plan['seed'],B,d,t,trial,1]))
        features=rng.normal(size=(B+N,d))
        latent=(rng.uniform(size=B+N)<expit(plan['utility_signal']*features[:,0])).astype(float)
        z=np.where(rng.uniform(size=B+N)<plan['proxy_error'],1-latent,latent)
        vx=np.mean((2*latent[:t,None]-1)*features[:t],axis=0)
        vz=np.mean((2*latent[:t]-1)*(2*z[:t]-1))
        x=np.c_[np.ones(m+N),features[t:]];y=np.r_[latent[t:B],np.zeros(N)]
        g=expit(np.einsum('ij,j->i',features[t:],vx)+vz*(2*z[t:]-1))
        if r['status']=='interior':
            status=interval_status(x,y,g,m,r['explicit_positive_power'])
            assert status=='interior',(r,status);checked+=1
        else:
            assert r['status']=='outside_mean_support'
            direction=rays[(r['condition'],trial)];u=x@direction
            margin=(2*y[:m]-1)*u[:m]
            h=float(np.mean(np.maximum(u[:m],0)-y[:m]*u[:m]))
            a=float(np.mean(np.maximum(u[m:],0)-g[m:]*u[m:])+np.mean((g[:m]-y[:m])*u[:m]))
            assert min(margin)>-1e-8 and abs(h)<1e-8 and a<-1e-8
            if np.isfinite(r.get('strict_human_margin',np.nan)):
                assert min(margin)>1e-10 and abs(min(margin)-r['strict_human_margin'])<1e-8
            maxh=max(maxh,abs(h),abs(h-r['human_recession']))
            maxa=max(maxa,abs(a-r['unit_recession']))
            for power in [.001,.37,1.]:
                original=float((1-power)*np.mean(np.maximum(u[:m],0))+power*np.mean(np.maximum(u[m:],0))-np.mean((y[:m]-power*g[:m])*u[:m])-power*np.mean(g[m:]*u[m:]))
                maxidentity=max(maxidentity,abs(original-((1-power)*h+power*a)))
                assert original<0
            negative+=1
    assert checked+negative==4000 and negative==len(rays)
    result=dict(existing_cohorts_reconstructed=4000,explicit_positive_power_witnesses_checked=checked,
        all_positive_power_obstruction_directions_checked=negative,maximum_human_recession_error=maxh,
        maximum_unit_recession_error=maxa,maximum_original_objective_identity_error=maxidentity,
        new_model_draws=0,new_human_or_AI_answers=0,unresolved=0)
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):run(a.base,a.audit,a.out)
