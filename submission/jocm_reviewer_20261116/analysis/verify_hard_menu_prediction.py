"""Independent finite-support and derivative checks for hard menu predictions."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from scipy.optimize import nnls,brentq
from threadpoolctl import threadpool_limits
from menu_corrected_logit import MenuLoss,menu_features,support,residual_support,save


def cone_project(w,y):
    J=len(w);generators=(np.eye(J)-np.eye(J)[y][:,None])[:,np.arange(J)!=y]
    coefficients,_=nnls(generators,w)
    return generators@coefficients


def simplex_project(w,mass):
    if mass==0:return np.zeros_like(w)
    theta=brentq(lambda x:np.maximum(w-x,0).sum()-mass,float(w.min()-mass),float(w.max()),xtol=1e-14)
    return np.maximum(w-theta,0)


def value(w,y,g,tau,v):
    c=g-np.eye(len(w))[y]
    labelled=v*c+cone_project(w-v*c,y)
    auxiliary=simplex_project(w+v*g/tau,v/tau)-v*g/tau
    norm=w@w
    return norm-np.sum((w-labelled)**2)+tau*(norm-np.sum((w-auxiliary)**2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(2026111177);finite=[];derivatives=[];perfect_error=0.
    for trial in range(80):
        J=2+trial%2;m=12+trial%7;N=m*(1+trial%3)
        d=1+trial%(N*(J-1)-(J-1))
        features=menu_features(rng.normal(size=(m+N,J,d)))
        y=rng.integers(J,size=m+N)
        error=1. if trial%4<2 else .25
        z=(y+(rng.random(m+N)<error)*rng.integers(1,J,size=m+N))%J
        g=np.eye(J)[z];states=[]
        for power in [1.,.5,.1]:
            loss=MenuLoss(features,y,g,m,power);original=support(loss);independent=residual_support(features,y,g,m,power)
            assert original['status'] not in ['rank_deficient','unresolved'] and independent!='unresolved',(trial,power,original,independent)
            expected='feasible' if original['status'] in ['interior','boundary_or_near_boundary'] else original['status']
            assert independent==expected,(trial,power,original,independent)
            finite.append(dict(trial=trial,alternatives=J,labels=m,auxiliary=N,profile_slopes=d,error_probability=error,power=power,original=original['status'],residual=independent))
            states.append(original['status']=='interior')
            # Perfect hard labels reduce exactly to positive weighted observed-choice loss.
            perfect=MenuLoss(features,y,np.eye(J)[y],m,power);beta=rng.normal(size=features.shape[-1])/5
            from scipy.special import logsumexp,softmax
            utilities=perfect.features@beta;weights=perfect.weights
            reference=float(weights@(logsumexp(utilities,axis=1)-utilities[np.arange(m+N),y]))
            score=np.einsum('n,nj,njp->p',weights,softmax(utilities,axis=1)-np.eye(J)[y],perfect.features)
            perfect_error=max(perfect_error,abs(reference-perfect.value(beta)[0]),float(np.max(abs(score-perfect.value(beta)[1]))))
        assert all(not states[i] or states[i+1] for i in range(2)),(trial,states)
    for J in [2,3,4]:
      for trial in range(80):
        w=rng.normal(size=J);w-=w.mean();y=int(rng.integers(J));z=int((y+rng.integers(1,J))%J)
        g=np.eye(J)[z] if trial%2==0 else rng.dirichlet(np.ones(J));tau=.5+2*rng.random()
        projection=cone_project(w,y);expected=2*(projection.max()-g@projection)
        baseline=value(w,y,g,tau,0.)
        errors=[]
        for step in [1e-3,1e-4,1e-5,1e-6]:errors.append(abs((value(w,y,g,tau,step)-baseline)/step-expected))
        assert errors[-1]<2e-5,(J,trial,errors)
        if J==2 and trial%2==0:assert abs(expected)<1e-12
        derivatives.append(dict(alternatives=J,trial=trial,hard_always_wrong=trial%2==0,expected_derivative=expected,FD_error_1e3=errors[0],FD_error_1e4=errors[1],FD_error_1e5=errors[2],FD_error_1e6=errors[3]))
    assert perfect_error<1e-10
    save(a.out/'hard_menu_finite_support_records.json',finite);save(a.out/'hard_menu_independent_derivative_records.json',derivatives)
    result=dict(finite_designs=80,support_classifications=len(finite),original_vs_residual_matches=len(finite),
                support_states={s:sum(r['original']==s for r in finite) for s in sorted({r['original'] for r in finite})},
                independent_NNLS_derivative_cases=len(derivatives),maximum_smallest_step_derivative_error=max(r['FD_error_1e6'] for r in derivatives),
                perfect_hard_label_objective_score_identity_error=perfect_error,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),new_human_or_LLM_responses=0,
                scope='Finite support and independent projection identities; no proof of originality or publication readiness')
    save(a.out/'hard_menu_independent_checks.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
