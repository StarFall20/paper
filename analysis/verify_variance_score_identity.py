"""Independent full covariance and finite-sample objective-identity checks."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from scipy.special import expit
from variance_reference_existence_audit import ShrunkPopulation
from power_tuned_choice_audit import objective


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    cov_error=0.;cov_equality=0.;minimum_improvement=np.inf
    cases=0
    for gamma in [0.,1.5]:
      for epsilon in [.1,.25]:
       for tau in [1.,4.]:
        alpha=tau/(1+tau);references=[]
        for strength in [alpha,.5*(1+alpha),1.]:
            lam=alpha/strength;pop=ShrunkPopulation(gamma,epsilon,tau,lam,strength,200)
            # These two nodes integrate every degree<=2 noise term exactly.
            s=np.repeat(pop.s,2);noise=np.tile([-1.,1.],len(pop.s));w=np.repeat(pop.weights/2,2)
            x=np.column_stack([np.ones(len(s)),s,noise]);mu=expit(gamma*s)
            y=np.repeat(pop.y,2);g=np.repeat(pop.g,2);g0=np.repeat(pop.oracle,2)
            A=x*(mu-y)[:,None];B=x*(mu-g)[:,None];B0=x*(mu-g0)[:,None]
            H=x.T@((w*mu*(1-mu))[:,None]*x);M=B0.T@(w[:,None]*B0)
            C=A.T@(w[:,None]*B);V=B.T@(w[:,None]*B)
            cov_error=max(cov_error,float(np.max(abs(C-strength*M))),float(np.max(abs(V-strength*strength*M))),float(np.max(abs(w@A))),float(np.max(abs(w@B))))
            inverse=np.linalg.inv(H);actual=inverse@(H-lam*(C+C.T)+(1+1/tau)*lam*lam*V)@inverse
            expected=inverse@(H-alpha*M)@inverse
            cov_error=max(cov_error,float(np.max(abs(actual-expected))))
            minimum_improvement=min(minimum_improvement,float(np.linalg.eigvalsh(inverse-actual).min()))
            references.append(actual);cases+=1
        cov_equality=max(cov_equality,float(np.max(abs(references[0]-references[-1]))))
    assert cov_error<1e-10 and cov_equality<1e-10 and minimum_improvement>0
    rng=np.random.default_rng(2026111209);truth_error=value_error=gradient_error=hessian_error=0.;finite_cases=0
    for trial in range(60):
        m=16+trial%11;N=m*(1+trial%3);n=m+N;q=2+trial%8;gamma=1.5;epsilon=.1;alpha=N/n
        x=np.column_stack([np.ones(n),rng.normal(size=(n,q+1))]);beta0=np.r_[0.,gamma,np.zeros(q)]
        mu=expit(x@beta0);y=(rng.random(n)<mu).astype(float);z=np.logical_xor(y,rng.random(n)<epsilon)
        g0=expit(x@beta0+(2*z.astype(float)-1)*np.log((1-epsilon)/epsilon));strength=alpha;lam=alpha/strength
        g=mu+strength*(g0-mu);base=objective(x,y,g0,m,alpha);changed=objective(x,y,g,m,lam)
        truth_error=max(truth_error,float(np.max(abs(base.value(beta0)[1]-changed.value(beta0)[1]))))
        for beta in [beta0,rng.normal(size=q+2)]:
            u=x@beta;pchoice=expit(u);ld=np.logaddexp(0,u)-mu*u
            correction=(lam-alpha)*(ld[m:].mean()-ld[:m].mean())
            dc=(lam-alpha)*(x[m:].T@(pchoice[m:]-mu[m:])/N-x[:m].T@(pchoice[:m]-mu[:m])/m)
            hL=x[:m].T@((pchoice[:m]*(1-pchoice[:m])/m)[:,None]*x[:m])
            hU=x[m:].T@((pchoice[m:]*(1-pchoice[m:])/N)[:,None]*x[m:])
            value_error=max(value_error,abs(changed.value(beta)[0]-base.value(beta)[0]-correction))
            gradient_error=max(gradient_error,float(np.max(abs(changed.value(beta)[1]-base.value(beta)[1]-dc))))
            hessian_error=max(hessian_error,float(np.max(abs(changed.hessian(beta)-base.hessian(beta)-(lam-alpha)*(hU-hL)))))
            finite_cases+=1
    assert max(truth_error,value_error,gradient_error,hessian_error)<1e-10
    result=dict(full_matrix_covariance_cases=cases,finite_objective_identity_cases=finite_cases,
        full_matrix_covariance_maximum_error=cov_error,paired_covariance_maximum_difference=cov_equality,
        minimum_covariance_improvement_eigenvalue=minimum_improvement,
        exact_truth_score_maximum_error=truth_error,objective_identity_maximum_error=value_error,
        gradient_identity_maximum_error=gradient_error,hessian_identity_maximum_error=hessian_error,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Analytic identities and exact moment quadrature. No Monte Carlo phase frequencies or independent priority claim.')
    (a.out/'score_identity_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
