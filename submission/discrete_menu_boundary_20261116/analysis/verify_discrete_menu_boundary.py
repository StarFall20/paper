"""Reconstruct all paired profiles; check rays and an independent interior LP."""
from pathlib import Path
import argparse,gzip,hashlib,itertools,json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from scipy.special import ndtri,softmax
from threadpoolctl import threadpool_limits
from menu_corrected_logit import menu_features

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
KEY=['law','label_menus','trial','orthogonal_slopes','method']

def interior_check(f,y,g,m,power,margin):
    n,J,p=f.shape;weights=np.r_[np.full(m,(1-power)/m),np.full(n-m,power/(n-m))]
    moment=(np.einsum('nj,njp->p',np.eye(J)[y[:m]]-power*g[:m],f[:m])/m
            +power*np.einsum('nj,njp->p',g[m:],f[m:])/(n-m))
    keep=weights>0;f=f[keep];weights=weights[keep];N=len(f)
    A=sparse.vstack([sparse.kron(sparse.eye(N),np.ones((1,J))),sparse.csr_matrix((weights[:,None,None]*f).reshape(-1,p).T)])
    target=np.r_[np.ones(N),moment]
    delta=float(margin)/4
    fit=linprog(np.zeros(N*J),A_eq=A,b_eq=target,bounds=(delta,1.),method='highs-ipm',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    residual=float(np.max(abs(A@fit.x-target))) if fit.x is not None else None
    minimum=float(fit.x.min()) if fit.x is not None else None
    assert fit.success and residual<1e-8 and minimum>=delta-1e-10,(fit.message,residual,minimum,delta)
    contrast=(f[:,:-1]-f[:,-1:]).reshape(-1,p);rank=int(np.linalg.matrix_rank(contrast))
    assert rank==p
    return dict(state='passed',fixed_probability_lower_bound=delta,minimum_probability=minimum,maximum_equality_error=residual,contrast_rank=rank,coefficients=p)

def software_checks():
    from discrete_menu_boundary import transform
    raw=np.array([-3.,-.5,0.,.5,3.]);binary=transform(raw,'symmetric_binary');ternary=transform(raw,'symmetric_ternary')
    assert np.array_equal(binary,[-1,-1,1,1,1])
    assert np.array_equal(ternary,np.array([-1,-1,0,1,1])*np.sqrt(1.5))
    assert abs(ndtri(2/3)-.43072729929545744)<1e-14
    # Elementary pivot-coordinate bound, including centering preimages.
    rank_checks=0;largest_ratio=0.
    for levels in [[-1.,1.],[-1.,0.,1.]]:
        vectors=np.asarray(list(itertools.product(levels,repeat=6)));a=1/len(levels)
        centered=vectors.reshape(-1,2,3);centered=centered-centered.mean(2,keepdims=True);centered=centered.reshape(-1,6)
        bases=[np.zeros((6,0)),np.array([1,-1,0,1,-1,0.])[:,None],np.array([[1,-1,0,1,-1,0.],[0,1,-1,0,1,-1.]]).T]
        for basis in bases:
            rank=np.linalg.matrix_rank(basis) if basis.size else 0
            Q=np.linalg.qr(basis)[0][:,:rank] if rank else np.zeros((6,0))
            residual=centered-centered@Q@Q.T
            probability=float(np.mean(np.max(abs(residual),axis=1)<1e-12));bound=a**(4-rank)
            assert probability<=bound+1e-14,(levels,rank,probability,bound)
            largest_ratio=max(largest_ratio,probability/bound);rank_checks+=1
    w=np.array([[[-3.],[-2.],[-1.]]]);before=transform(w,'symmetric_binary');before-=before.mean(axis=1,keepdims=True)
    after=transform(w-w.mean(axis=1,keepdims=True),'symmetric_binary');after-=after.mean(axis=1,keepdims=True)
    assert np.max(abs(before-after))>1.
    return dict(discrete_transform_fixtures_passed=True,pivot_rank_enumeration_checks=rank_checks,maximum_enumerated_probability_to_bound=largest_ratio,nonlinear_transform_after_centering_rejected=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--out',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--verification-out',type=Path,required=True);a=p.parse_args();a.verification_out.mkdir(parents=True,exist_ok=True)
    status=json.loads((a.out/'status.json').read_text());assert status['state']=='complete'
    plan=json.loads(a.plan.read_text());old=json.loads((a.repo/plan['original_plan']).read_text());ident=status['identity']
    assert ident['plan_sha256']==sha(a.plan)
    assert ident['original_plan_sha256']==sha(a.repo/plan['original_plan']) and ident['original_trials_sha256']==sha(a.repo/plan['original_trials'])
    for name,h in ident['source_hashes'].items():assert sha(Path(__file__).resolve().parent/name)==h,name
    with gzip.open(a.out/'checkpoint.json.gz','rt') as f:checkpoint=json.load(f)
    raw=pd.read_csv(a.out/'trials.csv',float_precision='round_trip');summary=pd.read_csv(a.out/'summary.csv',float_precision='round_trip')
    pd.testing.assert_frame_equal(raw,pd.DataFrame(checkpoint['records']),check_exact=True)
    assert len(raw)==27000 and len(summary)==54 and not raw.duplicated(KEY).any()
    assert raw.groupby(['law','label_menus','trial']).size().eq(9).all()
    gaussian=raw[raw.law.eq('gaussian')];original=pd.read_csv(a.repo/plan['original_trials'])
    merge=gaussian.merge(original,on=['label_menus','trial','orthogonal_slopes','method'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(merge)==9000 and merge.status_new.eq(merge.status_old).all()
    for row in summary.itertuples():
        part=raw[raw.law.eq(row.law)&raw.label_menus.eq(row.label_menus)&raw.orthogonal_slopes.eq(row.orthogonal_slopes)&raw.method.eq(row.method)]
        assert len(part)==row.replications==500
        for state,field in [('interior','finite'),('outside','exterior'),('boundary_or_near_boundary','boundary'),('rank_deficient','rank_deficient')]:assert part.status.eq(state).sum()==getattr(row,field)
        assert abs(part.status.eq('interior').mean()-row.finite_frequency)<1e-14
    assert not raw.status.str.startswith('unresolved').any()
    selected=raw[(raw.trial%50==0)&raw.status.eq('interior')]
    smallest=raw[raw.law.ne('gaussian')&raw.status.eq('interior')].nsmallest(20,'interior_margin')
    selected=pd.concat([selected,smallest]).drop_duplicates(KEY)
    # Any nonstandard positive-power state must be explicitly resolved.
    unusual=raw[(raw.power>0)&~raw.status.isin(['interior','outside'])]
    assert len(unusual)==0,unusual.to_dict('records')
    selected_keys={tuple(r) for r in selected[KEY].itertuples(index=False,name=None)}
    certificates={tuple(r[k] for k in KEY):r for r in checkpoint['certificates']}
    assert len(certificates)==len(checkpoint['certificates'])
    indexed={tuple(r[k] for k in KEY):r for r in raw.to_dict('records')}
    rng=np.random.default_rng(old['seed']);J=old['alternatives'];rays=separators=0;maximum_gap_error=maximum_shift_error=0.;checks=[]
    minimum_exterior_gap=1.;minimum_human_mean=1.;maximum_raw_direction_l1=0.;normalized_directions=[]
    for m in old['label_menus']:
        n=m+int(m*old['tau']);dims=sorted({max(1,round(m*k)) for k in old['dimension_ratios']})
        for trial in range(500):
            signal=rng.normal(size=(n,J,1));driver=rng.normal(size=(n,J,max(dims)))
            mu=softmax(old['gamma']*signal[...,0],axis=1)
            y=np.sum(rng.random(n)[:,None]>np.cumsum(mu,axis=1),axis=1)
            z=(y+(rng.random(n)<old['epsilon'])*rng.integers(1,J,size=n))%J;g=np.eye(J)[z]
            for law in plan['profile_laws']:
                if law=='gaussian':noise=driver
                elif law=='symmetric_binary':noise=np.where(driver<0.,-1.,1.)
                else:
                    c=ndtri(2/3);noise=np.where(driver < -c,-np.sqrt(1.5),np.where(driver > c,np.sqrt(1.5),0.))
                for q in dims:
                    attrs=np.concatenate([signal,noise[...,:q]],axis=2);features=menu_features(attrs)
                    raw_features=np.concatenate([np.broadcast_to(np.eye(J)[:,:-1],(n,J,J-1)),attrs],axis=2)
                    zero_sum=g-np.eye(J)[y]
                    maximum_shift_error=max(maximum_shift_error,float(np.max(abs(np.einsum('nj,njp->p',zero_sum,features-raw_features)))))
                    for method,power in old['powers'].items():
                        key=(law,m,trial,q,method);row=indexed[key]
                        if key in selected_keys:checks.append(dict(zip(KEY,key),**interior_check(features,y,g,m,power,row['interior_margin'])))
                        if key not in certificates:continue
                        cert=certificates[key];d=np.asarray(cert['direction'])
                        raw_norm=float(np.sum(abs(d)));assert raw_norm>0
                        maximum_raw_direction_l1=max(maximum_raw_direction_l1,raw_norm)
                        direction_scale=max(1.,raw_norm);d=d/direction_scale
                        assert np.sum(abs(d))<=1+1e-14
                        if raw_norm>1+1e-8:normalized_directions.append(dict(zip(KEY,key),raw_l1=raw_norm,positive_scale_divisor=direction_scale))
                        utility=features@d
                        if cert['kind']=='exterior':
                            human=float(np.mean(utility[:m].max(1)-utility[np.arange(m),y[:m]]))
                            predictionL=float(np.mean(utility[:m].max(1)-np.sum(g[:m]*utility[:m],axis=1)))
                            predictionU=float(np.mean(utility[m:].max(1)-np.sum(g[m:]*utility[m:],axis=1)))
                            gap=-(human+power*(predictionU-predictionL));err=abs(gap-cert['gap']/direction_scale)
                            assert gap>1e-8 and err<1e-9
                            maximum_gap_error=max(maximum_gap_error,err);minimum_exterior_gap=min(minimum_exterior_gap,gap);rays+=1
                        else:
                            margins=np.concatenate([utility[i,y[i]]-utility[i,np.arange(J)!=y[i]] for i in range(m)])
                            assert margins.min()>-1e-8 and margins.mean()>1e-8
                            assert abs(margins.mean()-cert['mean_margin']/direction_scale)<1e-10
                            minimum_human_mean=min(minimum_human_mean,float(margins.mean()));separators+=1
    assert rays==raw.status.eq('outside').sum()
    assert separators==len(raw[raw.method.eq('ordinary')&raw.status.eq('boundary_or_near_boundary')])
    assert len(checks)==len(selected_keys)
    assert maximum_shift_error<1e-11
    for _,p in raw.groupby(['law','label_menus','trial','method']):assert np.all(np.diff(p.sort_values('orthogonal_slopes').status.eq('interior').astype(int))<=0)
    for _,p in raw[raw.power>0].groupby(['law','label_menus','trial','orthogonal_slopes']):assert np.all(np.diff(p.sort_values('power',ascending=False).status.eq('interior').astype(int))>=0)
    pd.DataFrame(checks).to_csv(a.verification_out/'independent_interior_LPs.csv',index=False)
    result=dict(state='passed',records_reconciled=27000,summary_cells=54,original_gaussian_states_matched=9000,all_exterior_directions_checked=int(rays),all_ordinary_separators_checked=int(separators),independent_fixed_margin_LPs=len(checks),maximum_recession_gap_error=maximum_gap_error,minimum_verified_exterior_gap=minimum_exterior_gap,minimum_human_separator_mean=minimum_human_mean,maximum_raw_centered_moment_difference=maximum_shift_error,maximum_raw_direction_l1=maximum_raw_direction_l1,directions_outside_nominal_L1_tolerance=normalized_directions,direction_check_convention='Normalize every stored direction by max(1,L1 norm); check homogeneous gaps at the same scale. Primary states and directions remain unchanged.',rank_deficient=int(raw.status.eq('rank_deficient').sum()),positive_power_boundary_cases=int(len(unusual)),unresolved=0,software_checks=software_checks(),source_sha256=sha(Path(__file__).resolve()),primary_fits_rerun=False,independent_proof_or_priority_review=False,new_independent_cohorts=0,new_human_or_AI_answers=0)
    save(a.verification_out/'verification.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
