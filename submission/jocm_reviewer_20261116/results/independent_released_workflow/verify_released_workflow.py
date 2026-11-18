"""Verify released-workflow coefficients with the original three-term loss.

Shares only source/design/split/probability loading with the main audit; does not
import its target construction, fitting or support classification functions for
the 48 primary/sensitivity checks. Extra ordinary-human LP controls are labelled.
"""
from pathlib import Path
import argparse
import io
import json
import zipfile

import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits

from audit_released_workflow import CORE, OUTPUT, read_table, split_target, subset, save, digest


def independent_loss(beta, qx, qp, vx, vy, vp, lam):
    def part(x, target):
        u = x@beta
        p = softmax(u,axis=1)
        val = np.mean(logsumexp(u,axis=1)-np.sum(target*u,axis=1))
        grad = np.einsum("nj,njp->p", p-target,x)/len(x)
        return float(val),grad
    human, gh = part(vx,vy)
    aux, gu = part(qx,qp)
    labelled_prediction, gl = part(vx,vp)
    return human+lam*(aux-labelled_prediction), gh+lam*(gu-gl)


def ordinary_control(vx,vy):
    n,j,p = vx.shape
    # Residual probabilities r_ij = p_ij-y_ij with per-menu mass zero.
    # Maximize a lower bound on y+r, using unweighted original menus.
    eq = np.r_[np.c_[np.kron(np.eye(n),np.ones((1,j))),np.zeros(n)],
                np.c_[vx.reshape(-1,p).T,np.zeros(p)]]
    ub = np.c_[-np.eye(n*j),np.ones(n*j)]
    sol = linprog(np.r_[np.zeros(n*j),-1.],A_eq=eq,b_eq=np.zeros(n+p),A_ub=ub,b_ub=vy.ravel(),bounds=[(-1,1)]*(n*j)+[(0,1/j)],method="highs")
    rank = int(np.linalg.matrix_rank((vx[:,:-1]-vx[:,-1:]).reshape(-1,p)))
    assert sol.success and sol.x[-1]>1e-8 and rank==p
    return dict(rank=rank,parameters=p,interior_margin=float(sol.x[-1]),residual=float(np.max(abs(eq@sol.x))))


def main(inputs, results):
    details = json.loads((results/'support_details.json').read_text())
    coefs = {(v['context'],v['rep'],v['coverage'],v['rule']):v for v in json.loads((results/'coefficients.json').read_text())}
    checks,ordinary,ray_checks = [],[],[]
    with zipfile.ZipFile(inputs/CORE) as core,zipfile.ZipFile(inputs/OUTPUT) as output:
        raw = output.read(next(n for n in output.namelist() if n.endswith('/previsoes_llm.csv')))
        prev = pd.read_csv(io.BytesIO(raw))
        contexts = {c:read_table(core,c) for c in sorted(prev.contexto.unique())}
        cache = {}
        for row in details:
            key = (row['context'],row['rep'],row['coverage'])
            if key not in cache:
                people,tasks,names,idx = contexts[key[0]]
                h,v = split_target(people,key[0],key[1])
                predictions = prev[prev.contexto.eq(key[0])&prev.rep.eq(key[1])&prev.cond.eq(key[2])].set_index(['pid','card'])
                qx,_,qp,_,_ = subset(tasks,predictions,h)
                vx,vy,vp,_,_ = subset(tasks,predictions,v)
                cache[key]=(qx,qp,vx,vy,vp)
            qx,qp,vx,vy,vp = cache[key]
            lam = row['power']
            if row['status']=='interior':
                beta=np.array(coefs[(*key,row['rule'])]['beta'])
                val,g=independent_loss(beta,qx,qp,vx,vy,vp,lam)
                residual=float(max(abs(g)))
                assert residual<1e-7
                allx=np.concatenate([vx,qx]);w=np.r_[np.full(len(vx),(1-lam)/len(vx)),np.full(len(qx),lam/len(qx))]
                prob=softmax(allx@beta,axis=1);mean=np.einsum('nj,njp->np',prob,allx)
                hess=np.einsum('n,nj,nja,njb->ab',w,prob,allx,allx)-np.einsum('n,na,nb->ab',w,mean,mean)
                eig=float(np.linalg.eigvalsh(hess)[0]);assert eig>0
                checks.append(dict(context=key[0],rep=key[1],coverage=key[2],rule=row['rule'],gradient=residual,hessian_min_eigenvalue=eig,loss=val))
            elif row['status']=='outside':
                d=np.array(row['direction']);assert np.linalg.norm(d,1)<=1+1e-8
                vproj=vx@d;qproj=qx@d
                slope=float(np.mean(np.max(vproj,axis=1)-np.sum(vy*vproj,axis=1)) + lam*(np.mean(np.max(qproj,axis=1)-np.sum(qp*qproj,axis=1))-np.mean(np.max(vproj,axis=1)-np.sum(vp*vproj,axis=1))))
                assert slope<-1e-8 and abs(slope+row['recession_gap'])<1e-10
                path=[dict(t=t,value=independent_loss(t*d,qx,qp,vx,vy,vp,lam)[0]) for t in [0,10,100,1000,10000]]
                # For weights summing to one, logsumexp differs from its support
                # function by a quantity between zero and log(number of options).
                assert 0<=path[-1]['value']-10000*slope<=np.log(3)+1e-9
                ray_checks.append(dict(context=key[0],rep=key[1],coverage=key[2],rule=row['rule'],slope=slope,path=path))
            else:
                raise AssertionError('Unresolved class retained: '+row['status'])
        # Secondary controls added after observing the unit-power exterior cell:
        # all eight context/split human designs, with no cell selection.
        for context in sorted(contexts):
            for rep in [0,1]:
                _,_,vx,vy,_=cache[(context,rep,'alta')]
                ordinary.append(dict(context=context,rep=rep,**ordinary_control(vx,vy)))
    assert len(checks)==47 and len(ray_checks)==1 and len(ordinary)==8
    save(results/'independent_verification.json',dict(verified_finite_coefficients=len(checks),verified_exterior_directions=len(ray_checks),ordinary_controls_added_after_primary_results=True,all_context_split_ordinary_controls=len(ordinary),ordinary_interior=8,maximum_original_three_term_score=max(v['gradient'] for v in checks),minimum_finite_hessian_eigenvalue=min(v['hessian_min_eigenvalue'] for v in checks),raw_loss_and_gradient_checks=checks,exterior_checks=ray_checks,ordinary_controls=ordinary,source_data_loading_shared=True,main_support_fit_objective_imported=False,source_code_executed=False,verifier_sha256=digest(Path(__file__).read_bytes())))
    print(json.dumps(dict(finite_coefficients=47,exterior_rays=1,ordinary_controls=8,all_passed=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--results',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):
        main(a.inputs,a.results)
