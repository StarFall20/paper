"""Independent trained-logistic rotation and corrected-objective identities.

Finite simulated training fits check algebra only; no convergence of trained
overlaps,phase law,inference performance or arbitrary cross-fitting is inferred.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from scipy.linalg import null_space
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from power_tuned_choice_audit import objective,support
from public_conjoint_optimizer_audit import save


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261203);prediction=orthogonal=value=score=hessian=0.;matches=0;counts={}
    for trial in range(60):
        d=4+trial%11;train=80+trial%17;m=40+trial%13;N=2*m
        Xtrain=rng.normal(size=(train,d));Ytrain=(rng.random(train)<expit(1.5*Xtrain[:,0])).astype(float)
        fit=LogisticRegression(C=.1,max_iter=3000,random_state=0).fit(Xtrain,Ytrain)
        theta=fit.coef_[0];bias=float(fit.intercept_[0]);aligned=float(theta[0]);noise=float(np.linalg.norm(theta[1:]));assert noise>1e-8
        e=np.eye(d)[:,0];direction=np.r_[0.,theta[1:]/noise]
        Q=np.c_[e,direction,null_space(np.vstack([e,direction]))]
        G=np.eye(d+1);G[1:,1:]=Q
        orthogonal=max(orthogonal,float(abs(G.T@G-np.eye(d+1)).max()))
        features=rng.normal(size=(m+N,d));y=(rng.random(m+N)<expit(1.5*features[:,0])).astype(float)
        x=np.c_[np.ones(m+N),features];rotated=x@G
        g=fit.predict_proba(features)[:,1];reconstructed=expit(bias+aligned*rotated[:,1]+noise*rotated[:,2])
        prediction=max(prediction,float(max(abs(g-reconstructed))))
        beta=rng.normal(size=d+1)/5;new_beta=G.T@beta
        for power in [1.,.5,.1]:
            before=objective(x,y,g,m,power);after=objective(rotated,y,reconstructed,m,power)
            old,grad=before.value(beta);new,new_grad=after.value(new_beta)
            value=max(value,abs(old-new));score=max(score,float(max(abs(G.T@grad-new_grad))))
            hessian=max(hessian,float(abs(G.T@before.hessian(beta)@G-after.hessian(new_beta)).max()))
            one=support(before)['status'];two=support(after)['status'];assert one==two,(trial,power,one,two)
            matches+=1;counts[one]=counts.get(one,0)+1
    assert prediction<1e-12 and max(orthogonal,value,score,hessian)<1e-9
    result={'independent_simulated_training_fits':60,'training_rule':'fixed C0.1 sklearn logistic,with independent Gaussian audit rows',
            'corrected_objective_support_rotation_matches':matches,'support_states':counts,
            'maximum_prediction_identity_error':prediction,'maximum_orthogonality_error':orthogonal,
            'maximum_objective_identity_error':value,'maximum_rotated_score_error':score,'maximum_rotated_Hessian_error':hessian,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'training_overlap_convergence_tested':False,'cross_fitting_claimed':False,'new_human_or_AI_responses':0}
    save(a.out/'trained_prediction_rotation_checks.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
