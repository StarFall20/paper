"""Check the actual weighted objective, independently of effective-design LPs.

Draw reconstruction, scalar integration and weighted moment enumeration are
separate from the generator. Generic attainable-moment code is established.
"""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from power_tuned_choice_audit import WeightedLogit,support
from verify_radial_compensation import independent_normal_box


def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def actual_loss(base,d,y,g,m,lam,r):
    x=base*d[:,None];w=d**(-r);n=len(y)-m
    curvature=w*np.r_[np.full(m,(1-lam)/m),np.full(n,lam/n)]
    moment=x[:m].T@(w[:m]*(y[:m]-lam*g[:m]))/m+lam*x[m:].T@(w[m:]*g[m:])/n
    normalizer=float(curvature.sum())
    return WeightedLogit(x,curvature/normalizer,moment/normalizer),normalizer


def reconstruct(plan,m,rep):
    q=int(round(m*plan['noise_label_ratio']));rng=np.random.default_rng(np.random.SeedSequence([plan['source_seed'],m,rep]))
    base=np.concatenate([np.ones((2*m,1)),rng.standard_normal((2*m,q+1))],axis=1)
    y=rng.integers(0,2,2*m).astype(float);z=np.not_equal(y,rng.uniform(size=2*m)<plan['epsilon']).astype(float)
    g=plan['epsilon']+(1-2*plan['epsilon'])*z
    p=plan['simulation_large_probability'];low=plan['simulation_small_scale'];high=math.sqrt((1-(1-p)*low**2)/p)
    d=np.where(rng.uniform(size=2*m)<p,high,low)
    digest=hashlib.sha256()
    for arr in [base,y,g,d]:digest.update(arr.tobytes())
    return base,d,y,g,digest.hexdigest()


def direct_integral(scales,mass,v,e,lam,tau):
    total=error=0.
    for d,p in zip(scales,mass):
        for residual,probability in [(e,1-e),(1-e,e)]:
            lo=d*v*residual;hi=lo+d*v*(1-lam)/lam
            value,err=independent_normal_box(lo,hi);total+=p*probability*value;error+=p*probability*err
        value,err=independent_normal_box(-d*v*(1-e)/tau,d*v*e/tau)
        total+=tau*p*value;error+=tau*p*err
    return total,error


def population_checks(plan,frame):
    rows=[]
    for row in frame.itertuples(index=False):
        if row.model=='constant':d=np.ones(1);mass=np.ones(1)
        else:
            d=np.array([row.small_scale,math.sqrt((1-(1-row.large_probability)*row.small_scale**2)/row.large_probability)])
            mass=np.array([1-row.large_probability,row.large_probability])
        effective=d**(1-row.row_exponent);e=plan['epsilon'];lam=row.power;tau=row.auxiliary_ratio;v=row.profile_scale
        value,error=direct_integral(effective,mass,v,e,lam,tau);discrepancy=abs(value-row.critical_ratio);assert discrepancy<1e-8
        h=1e-4;vp=direct_integral(effective,mass,v+h,e,lam,tau)[0];vm=direct_integral(effective,mass,v-h,e,lam,tau)[0]
        stationary_error=abs((vp-vm)/(2*h));assert stationary_error<2e-6
        assert value>=max(vp,vm)-1e-9
        # A separate weight derivative: vary the row scale inside the integral,
        # holding the optimizer fixed, as required by the envelope theorem.
        h=1e-5;r=row.row_exponent
        plus=direct_integral(d**(1-r-h),mass,v,e,lam,tau)[0]
        minus=direct_integral(d**(1-r+h),mass,v,e,lam,tau)[0]
        derivative_error=abs((plus-minus)/(2*h)-row.capacity_derivative);assert derivative_error<2e-6
        w=d**(-r);vh=vp_score=cross=0.
        for dd,ww,prob in zip(d,w,mass):
            for y in [0,1]:
                for flip,flip_prob in [(0,1-e),(1,e)]:
                    z=y if not flip else 1-y;g=e+(1-2*e)*z;probability=prob*.5*flip_prob
                    human=ww*dd*(.5-y);pred=ww*dd*(.5-g)
                    vh+=probability*human**2;vp_score+=probability*pred**2;cross+=probability*human*pred
        hessian=float(mass@(w*d*d))/4
        covariance=(vh-2*lam*cross+(1+1/tau)*lam*lam*vp_score)/(hessian*hessian)
        covariance_error=abs(covariance-row.regular_covariance_factor);assert covariance_error<1e-10
        opt=minimize_scalar(lambda l:(vh-2*l*cross+(1+1/tau)*l*l*vp_score)/(hessian*hessian),bounds=(0,1),method='bounded')
        assert opt.success and abs(opt.x-tau/(1+tau))<1e-6
        rows.append(dict(model=row.model,auxiliary_ratio=tau,row_exponent=r,direct_integral=value,integral_discrepancy=discrepancy,quadrature_error_estimate=error,
            stationary_finite_difference_error=stationary_error,envelope_derivative_error=derivative_error,enumerated_covariance_error=covariance_error,independent_variance_power=float(opt.x)))
    for (model,tau),part in frame.groupby(['model','auxiliary_ratio']):
        part=part.sort_values('row_exponent');strict=model!='constant'
        assert (np.diff(part.critical_ratio)>(0 if strict else -1e-10)).all()
        assert (np.diff(part.regular_covariance_multiplier)>(0 if strict else -1e-10)).all()
        if strict:assert part.iloc[:-1].capacity_derivative.min()>0
        assert abs(part.iloc[-1].capacity_derivative)<1e-10
    return rows


def finite_design_checks():
    rng=np.random.default_rng(2026111355);rows=[]
    for case in range(12):
        m=8;n=12;base=np.column_stack([np.ones(m+n),rng.normal(size=(m+n,2))]);d=rng.uniform(.2,2,m+n)
        y=rng.integers(0,2,m+n).astype(float);g=rng.uniform(.05,.95,m+n)
        for r in [0.,.5,1.]:
            for lam in [.1,.5,1.]:
                loss,c=actual_loss(base,d,y,g,m,lam,r);effective=d**(1-r);vbase=base*effective[:,None]
                ordinary,cv=actual_loss(vbase,np.ones(m+n),y,g,m,lam,0.)
                cs=support(loss);vs=support(ordinary);assert cs['status']==vs['status']
                err=0.
                for direction in rng.normal(size=(10,base.shape[1])):
                    a=c*(loss.weights@np.maximum(loss.x@direction,0)-loss.moment@direction)
                    b=cv*(ordinary.weights@np.maximum(ordinary.x@direction,0)-ordinary.moment@direction)
                    err=max(err,abs(a-b))
                assert err<1e-12
                beta=np.array([.2,-.1,.3]);step=1e-5;eye=np.eye(3)
                fd=np.array([(loss.value(beta+v*step)[0]-loss.value(beta-v*step)[0])/(2*step) for v in eye])
                grad_error=float(abs(fd-loss.value(beta)[1]).max());assert grad_error<1e-8
                # The virtual likelihood is not generally the actual one.
                value_difference=abs(c*loss.value(beta)[0]-cv*ordinary.value(beta)[0])
                if r==1:assert value_difference>1e-8
                rows.append(dict(case=case,row_exponent=r,power=lam,status=cs['status'],maximum_recession_identity_error=err,gradient_difference_error=grad_error,objective_value_difference=value_difference))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'verification.json').exists(),'Use a fresh output directory.'
    plan=json.loads(a.plan.read_text());trials=pd.read_csv(a.root/'results/weight_path_trials.csv');assert len(trials)==5000
    draw_hashes=pd.read_csv(a.root/'results/source_draw_hashes.csv').set_index(['labels','replication']).source_draw_sha256
    populations=population_checks(plan,pd.read_csv(a.root/'results/population_weight_path.csv'))
    pd.DataFrame(populations).to_csv(a.out/'independent_population_checks.csv',index=False)
    finite=finite_design_checks();pd.DataFrame(finite).to_csv(a.out/'finite_design_identity_checks.csv',index=False)
    records=[];rays=[];verified=trials.copy();verified['initial_status']=verified.status
    for (m,rep),part in trials.groupby(['labels','replication'],sort=True):
        m=int(m);rep=int(rep);base,d,y,g,digest=reconstruct(plan,m,rep);assert digest==draw_hashes.loc[m,rep]
        for index,row in part.iterrows():
            loss,c=actual_loss(base,d,y,g,m,float(row.power),float(row.row_exponent));result=support(loss)
            if result['status']=='interior':status='feasible'
            elif result['status']=='outside_mean_support':status='outside'
            else:raise AssertionError((int(index),result))
            if row.status!='unknown':assert status==row.status,(int(index),row.to_dict(),result)
            verified.loc[index,'status']=status
            records.append(dict(row_index=int(index),verified_status=status,independent_status=result['status'],interior_margin=result.get('interior_margin'),separation_gap=result.get('separation_gap'),common_normalizer=c))
            if 'direction' in result:
                direction=np.array(result['direction']);cost=float(loss.weights@np.maximum(loss.x@direction,0)-loss.moment@direction)
                assert cost<-1e-8
                rays.append(dict(row_index=int(index),normalized_recession_cost=cost,unnormalized_recession_cost=c*cost,direction=direction.tolist()))
        if (rep+1)%100==0:print('original weighted objective',m,rep+1,'records',len(records),flush=True)
    pd.DataFrame(records).to_csv(a.out/'independent_support_replays.csv',index=False);save(a.out/'negative_recession_directions.json',rays)
    verified.to_csv(a.out/'weight_path_trials_verified.csv',index=False)
    expected=trials.groupby(['labels','row_exponent']).status.agg(lambda s:int(s.eq('feasible').sum())).to_dict()
    actual=verified.groupby(['labels','row_exponent']).status.agg(lambda s:int(s.eq('feasible').sum())).to_dict()
    assert expected==actual
    save(a.out/'verification.json',dict(date='2026-11-13',all_weighted_criteria_independently_checked=len(records),reused_cohort_hashes_checked=len(draw_hashes),negative_recession_directions=len(rays),independent_population_rows=len(populations),finite_design_identity_cases=len(finite),maximum_integral_discrepancy=max(x['integral_discrepancy'] for x in populations),maximum_reported_quadrature_error=max(x['quadrature_error_estimate'] for x in populations),maximum_envelope_derivative_error=max(x['envelope_derivative_error'] for x in populations),maximum_covariance_enumeration_error=max(x['enumerated_covariance_error'] for x in populations),initial_unknowns=int(trials.status.eq('unknown').sum()),final_unknowns=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),trials_sha256=hashlib.sha256((a.root/'results/weight_path_trials.csv').read_bytes()).hexdigest(),priority_or_submission_certified=False))


if __name__=='__main__':main()
