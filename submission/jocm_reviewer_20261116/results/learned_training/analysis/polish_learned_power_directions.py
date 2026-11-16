"""Replace weak numerical directions by strict, robust-margin obstructions.

Continuous Gaussian separated designs admit strict separators. The feasible LP
requires human margins at least one and negative unit recession at most minus
one, then rescales. It changes neither cohorts nor existence classifications.
"""
from pathlib import Path
import argparse,json,shutil,hashlib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
from learned_proxy_budget_boundary import generate


def run(base,audit,out):
    out.mkdir(parents=True,exist_ok=True);assert not (out/'trials.csv').exists()
    plan=json.loads((base/'plan.json').read_text());df=pd.read_csv(audit/'trials.csv')
    rays=[];minimum_margin=np.inf;minimum_gap=np.inf
    for i,r in df.iterrows():
        if r.status=='interior':continue
        spec=tuple(int(r[n]) for n in ['total_labels','slopes','training_labels'])
        _,_,_,x,y,g=generate(plan,spec,int(r.trial));m=spec[0]-spec[2];n,p=x.shape;N=n-m
        signed=(2*y[:m]-1)[:,None]*x[:m]
        linear=x[:m].T@(g[:m]-y[:m])/m-x[m:].T@g[m:]/N
        cost=np.r_[linear,np.full(N,1/N)]
        constraints=sparse.vstack([
            sparse.hstack([-sparse.csr_matrix(signed),sparse.csr_matrix((m,N))]),
            sparse.hstack([sparse.csr_matrix(x[m:]),-sparse.eye(N)]),
            sparse.csr_matrix(cost[None,:])],format='csr')
        fit=linprog(np.zeros(p+N),A_ub=constraints,b_ub=np.r_[-np.ones(m),np.zeros(N),-1.],
            bounds=[(None,None)]*p+[(0.,None)]*N,method='highs')
        assert fit.success,(r.to_dict(),fit.message)
        d=fit.x[:p];d=d/max(1.,float(max(abs(d))));u=x@d
        margin=float(min((2*y[:m]-1)*u[:m]))
        h=float(np.mean(np.maximum(u[:m],0)-y[:m]*u[:m]))
        a=float(np.mean(np.maximum(u[m:],0)-g[m:]*u[m:])+np.mean((g[:m]-y[:m])*u[:m]))
        assert margin>1e-10 and h==0. and a<-1e-10,(margin,h,a)
        minimum_margin=min(minimum_margin,margin);minimum_gap=min(minimum_gap,-a)
        df.loc[i,'human_weak_margin']=margin;df.loc[i,'human_recession']=h;df.loc[i,'unit_recession']=a
        df.loc[i,'strict_human_margin']=margin
        rays.append(dict(condition=int(r.condition),trial=int(r.trial),direction=d.tolist()))
    df.to_csv(out/'trials.csv',index=False)
    shutil.copy2(audit/'summary.csv',out/'summary.csv')
    (out/'directions.json').write_text(json.dumps(rays)+'\n')
    status=json.loads((audit/'status.json').read_text())
    status.update(strict_obstruction_directions=len(rays),minimum_strict_human_margin=minimum_margin,
        minimum_negative_recession_gap=minimum_gap,maximum_human_recession=0.,
        refinement_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_classifications_changed=False)
    (out/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):run(a.base,a.audit,a.out)
