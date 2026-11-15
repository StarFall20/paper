"""Read the published case and independently check its exact recession signs."""
from pathlib import Path
from fractions import Fraction
import argparse,json
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
from public_conjoint_optimizer_audit import load_data,save
from public_conjoint_matched_audit import nuisance


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--case',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    cards=json.loads((a.case/'exact_case.json').read_text())['conditions']
    preds={ (r['trial'],r['label_rows'],r['prediction']):r['g'] for r in json.loads((a.case/'frozen_case_predictions.json').read_text()) }
    x,y,z,_,_=load_data(a.input_dir);maxg=maxscore=0.;radii=[]
    for c in cards:
        rng=np.random.RandomState(20261109)
        for trial in range(c['trial']+1):order=(rng.permutation(240)[:,None]*5+np.arange(5)[None]).reshape(-1)
        xx=x[order];m=c['label_rows'];N=1200-m;yy=np.r_[y[order[:m]],np.full(N,np.nan)]
        assert np.array_equal(xx,np.round(xx));d=c['integer_direction']
        # Integer arithmetic for all attribute-direction products and signed margins.
        u=[sum(int(xx[i,j])*int(d[j]) for j in range(len(d))) for i in range(1200)]
        signed=[(2*int(yy[i])-1)*u[i] for i in range(m)]
        assert min(signed)>=0 and max(signed)>0
        g=preds[(c['trial'],m,c['prediction'])];q=[Fraction.from_float(float(v)) for v in g]
        lh=sum(Fraction(max(u[i],0)-int(yy[i])*u[i],m) for i in range(m))
        lg=sum((max(u[i],0)-q[i]*u[i])/m for i in range(m))
        ug=sum((max(u[i],0)-q[i]*u[i])/N for i in range(m,1200))
        ainf=lh+ug-lg
        assert lh==0 and ainf<0
        assert ainf==Fraction(int(c['unit_recession_numerator']),int(c['unit_recession_denominator']))
        sensitivity=Fraction(sum(abs(v) for v in u[:m]),m)+Fraction(sum(abs(v) for v in u[m:]),N)
        radius=-ainf/sensitivity
        assert radius==Fraction(int(c['perturbation_radius_numerator']),int(c['perturbation_radius_denominator']))
        regenerated,_=nuisance(xx,yy,z[order],m,c['prediction']=='AI')
        err=float(max(abs(regenerated-np.array(g))));maxg=max(maxg,err)
        assert err<min(1e-10,float(radius)/100)
        # A directly expressed score/Hessian checks the before-pool root.
        n=50;xb=xx[:m+n];gb=np.array(g[:m+n]);b=np.array(c['before_root']);lam=c['before_power'];mu=expit(xb@b)
        score=xb[:m].T@(mu[:m]-yy[:m])/m+lam*(xb[m:].T@(mu[m:]-gb[m:])/n-xb[:m].T@(mu[:m]-gb[:m])/m)
        H=(1-lam)*xb[:m].T@((mu[:m]*(1-mu[:m]))[:,None]*xb[:m])/m+lam*xb[m:].T@((mu[m:]*(1-mu[m:]))[:,None]*xb[m:])/n
        maxscore=max(maxscore,float(max(abs(score))))
        assert max(abs(score))<1e-7 and np.linalg.eigvalsh(H).min()>1e-11
        radii.append(float(radius))
    save(a.out/'exact_case_verification.json',dict(state='passed',existing_allocations=1,prediction_conditions=len(cards),
        numerical_before_roots_verified=len(cards),exact_after_all_positive_power_negative_recessions=len(cards),
        exact_zero_power_nonattainment_directions=len(cards),minimum_uniform_prediction_perturbation_radius=min(radii),
        maximum_regenerated_prediction_difference=maxg,maximum_direct_before_root_score=maxscore,
        exact_before_root_certificate=False,arithmetic_scope='Exact integer/dyadic signs for the frozen after-pool criterion; numerical before-pool interior and roots.',new_human_or_AI_answers=0))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
