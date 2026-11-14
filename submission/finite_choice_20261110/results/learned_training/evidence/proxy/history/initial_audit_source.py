"""Secondary all-positive-power audit of the 4000 existing proxy cohorts.

It reuses classical support LPs. A negative corrected recession on a human weak
separator obstructs every positive power. A strict interval witness supplies an
explicit finite power, checked against the original objective.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
from learned_proxy_budget_boundary import generate
from vanishing_power_choice import rescaled_support
from power_tuned_choice_audit import support,objective


def run(base,out):
    out.mkdir(parents=True,exist_ok=True);assert not (out/'trials.csv').exists()
    plan=json.loads((base/'plan.json').read_text())
    frame=pd.read_csv(base/'phase/phase_trials.csv');rows=[];directions=[]
    for (condition,trial),part in frame.groupby(['condition','trial'],sort=False):
        rec=part.iloc[0];spec=tuple(int(rec[n]) for n in ['total_labels','slopes','training_labels'])
        X,Y,v,x,y,g=generate(plan,spec,int(trial));m=spec[0]-spec[2];n,p=x.shape;N=n-m
        limit=rescaled_support(x,y,g,m,None)
        row=dict(condition=int(condition),trial=int(trial),total_labels=spec[0],slopes=spec[1],training_labels=spec[2],status=limit['status'])
        if limit['status']=='interior':
            power=limit['certified_positive_power'];check=support(objective(x,y,g,m,power))
            assert check['status']=='interior',check
            row.update(explicit_positive_power=power,witness_margin=check['interior_margin'],witness_equality_error=check['equality_residual'])
        elif limit['status']=='outside_mean_support':
            signed=(2*y[:m]-1)[:,None]*x[:m]
            linear=x[:m].T@(g[:m]-y[:m])/m-x[m:].T@g[m:]/N
            constraints=sparse.vstack([sparse.hstack([-sparse.csr_matrix(signed),sparse.csr_matrix((m,N))]),sparse.hstack([sparse.csr_matrix(x[m:]),-sparse.eye(N)])],format='csr')
            fit=linprog(np.r_[linear,np.full(N,1/N)],A_ub=constraints,b_ub=np.zeros(n),bounds=[(-1.,1.)]*p+[(0.,None)]*N,method='highs')
            assert fit.success,fit.message
            d=fit.x[:p];u=x@d
            h=float(np.mean(np.maximum(u[:m],0)-y[:m]*u[:m]))
            a=float(np.mean(np.maximum(u[m:],0)-g[m:]*u[m:])+np.mean((g[:m]-y[:m])*u[:m]))
            assert h<1e-8 and a<-1e-8,(h,a)
            row.update(human_weak_margin=float(min(signed@d)),human_recession=h,unit_recession=a)
            directions.append(dict(condition=int(condition),trial=int(trial),direction=d.tolist()))
        else:raise RuntimeError(('Unresolved limit classification',condition,trial,limit))
        assert not ((part.status=='interior').any() and row['status']!='interior')
        rows.append(row)
        if len(rows)%100==0:
            pd.DataFrame(rows).to_csv(out/'trials.csv',index=False)
            print(json.dumps({'all_power_cohorts':len(rows)}),flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'trials.csv',index=False)
    df.assign(finite=df.status.eq('interior')).groupby(['total_labels','slopes','training_labels'],as_index=False).agg(draws=('trial','size'),some_positive_power_finite=('finite','sum')).to_csv(out/'summary.csv',index=False)
    (out/'directions.json').write_text(json.dumps(directions)+'\n')
    status=dict(existing_cohorts=len(df),new_model_draws=0,positive_power_witnesses=int(df.status.eq('interior').sum()),all_positive_power_negative_directions=len(directions),unresolved=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),new_estimator_claimed=False)
    (out/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):run(a.base,a.out)
