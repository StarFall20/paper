"""Independent Gaussian-integral, training and original-loss checks.

This verifier does not import either primary learned-score implementation.
It reconstructs every generated dataset and saved direction. Oriented interval
LP checks use the 20 preplanned draws per condition, not all primary programs.
"""
from pathlib import Path
import argparse, hashlib, json, gzip
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import root, linprog
from scipy.special import expit, ndtr, roots_legendre
from scipy.integrate import quad
from scipy.stats import norm
from threadpoolctl import threadpool_limits


def read_records(path):
    if path.exists():
        return json.loads(path.read_text())
    index=json.loads(path.with_suffix('.index.json').read_text())
    records=[]
    for row in index['files']:
        source=path.parent/row['path'];raw=source.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==row['sha256']
        records.extend(json.loads(gzip.decompress(raw)))
    assert len(records)==index['record_count']
    return records


def clipped(h,t,lo,hi):
    l=t*lo;u=t*hi;A=h+l;B=h+u
    PA=ndtr(A);PB=ndtr(B);SB=ndtr(-B)
    fa=norm.pdf(A);fb=norm.pdf(B)
    eleft=-fa-h*PA;eright=fb-h*SB
    middle=(1+h*h)*(PB-PA)+(A-2*h)*fa-(B-2*h)*fb
    value=2*l*eleft-l*l*PA+middle+2*u*eright-u*u*SB
    meanclip=l*PA+fa-fb-h*(PB-PA)+u*SB
    dh=-2*meanclip
    dt=2*lo*(eleft-l*PA)+2*hi*(eright-u*SB)
    return value,dh,dt


def independent_population(plan,frame,out):
    # Legendre integration against normal densities, distinct from primary GH.
    z,w=roots_legendre(100);z=11*z;w=11*w*norm.pdf(z)
    s,t=np.meshgrid(z,z,indexing='ij');s=s.ravel();t=t.ravel()
    weight=np.outer(w,w).ravel();p=expit(plan['utility_signal']*s)
    proxy='proxy_error' in plan
    labels=[(0,0),(0,1),(1,0),(1,1)] if proxy else [(0,1),(1,1)]
    S=np.tile(s,len(labels));T=np.tile(t,len(labels))
    Y=np.repeat([y for y,z in labels],len(s));Z=np.repeat([z for y,z in labels],len(s))
    epsilon=plan.get('proxy_error',0.)
    weights=np.concatenate([weight*(p if y else 1-p)*((1-epsilon if y==z else epsilon) if proxy else 1.) for y,z in labels])
    v=np.c_[S,T];sl=1-2*Y;su=2*Z-1
    a=quad(lambda x:x*np.tanh(plan['utility_signal']*x/2)*norm.pdf(x),-12,12,epsabs=1e-12)[0]
    rows=[];maxvalue=maxstationarity=maxcoef=0.
    for rec in frame.to_dict('records'):
        alpha=rec['training_fraction'];rho=rec['total_slope_ratio'];power=rec['power'];tau=rec['auxiliary_per_total_label']/(1-alpha)
        g=expit(a*S+np.sqrt(rho/alpha)*T+((1-2*epsilon)*(2*Z-1) if proxy else 0.))
        r=np.where(Y==0,g,1-g);lo=power*r;hi=1-power+power*r
        ulo=power/tau*np.where(Z==1,-g,g-1);uhi=power/tau*np.where(Z==1,1-g,g)
        def value(e):
            u=v@e[:2];scale=e[2]
            left,dl,tl=clipped(sl*u,scale,lo,hi)
            right,dr,tr=clipped(su*u,scale,ulo,uhi)
            grad=v.T@(weights*(sl*dl+tau*su*dr))
            return float(weights@(left+tau*right)),np.r_[grad,weights@(tl+tau*tr)]
        start=np.array([rec['signal_multiplier'],rec['training_noise_multiplier'],rec['scale']])
        fit=root(lambda e:value(e)[1],start,method='hybr',options={'xtol':1e-9})
        val,grad=value(fit.x);err=abs(val-rec['critical_ratio'])
        assert max(abs(grad))<1e-7 and fit.x[2]>0,(fit.message,grad)
        maxvalue=max(maxvalue,err);maxstationarity=max(maxstationarity,float(max(abs(grad))))
        maxcoef=max(maxcoef,float(max(abs(fit.x-start))))
        rows.append(dict(total_slope_ratio=rho,training_fraction=alpha,
            auxiliary_per_total_label=rec['auxiliary_per_total_label'],power=power,
            independent_critical_ratio=val,difference=err,stationarity=float(max(abs(grad)))))
    pd.DataFrame(rows).to_csv(out/'independent_population.csv',index=False)
    assert maxvalue<2e-6,maxvalue
    return dict(population_conditions=len(rows),maximum_independent_boundary_difference=maxvalue,
                maximum_joint_saddle_residual=maxstationarity,maximum_saddle_coordinate_difference=maxcoef,
                independent_signal_mean=a)


def interval_status(x,y,g,m,power):
    n=len(y);tau=(n-m)/m
    sign=np.r_[1-2*y[:m],np.ones(n-m)]
    r=np.where(y[:m]==0,g[:m],1-g[:m])
    lower=np.r_[power*r,-power*g[m:]/tau]
    upper=np.r_[1-power+power*r,power*(1-g[m:])/tau]
    width=upper-lower;A=x.T*sign
    equality=np.c_[A*width,np.zeros(x.shape[1])]
    eye=sparse.eye(n,format='csr');one=sparse.csr_matrix(np.ones((n,1)))
    inequalities=sparse.vstack([sparse.hstack([-eye,one]),sparse.hstack([eye,one])],format='csr')
    fit=linprog(np.r_[np.zeros(n),-1.],A_ub=inequalities,b_ub=np.r_[np.zeros(n),np.ones(n)],
        A_eq=equality,b_eq=-A@lower,bounds=[(0.,1.)]*n+[(0.,.5)],method='highs')
    if fit.status==2:return 'outside_mean_support'
    assert fit.success,fit.message
    return 'interior' if fit.x[-1]>1e-8 else 'boundary_or_near_boundary'


def verify(base,out):
    out.mkdir(parents=True,exist_ok=True);plan=json.loads((base/'plan.json').read_text())
    phase=base/'phase';raw=pd.read_csv(phase/'phase_trials.csv');overlap=pd.read_csv(phase/'training_overlaps.csv')
    population=pd.read_csv(base/'population/population.csv')
    state=json.loads((phase/'run_status.json').read_text());assert state['state']=='complete'
    assert len(raw)==16000 and len(overlap)==4000
    assert not raw.duplicated(['condition','trial','method']).any()
    assert not state['nonstandard_states'],state['nonstandard_states']
    rays=read_records(phase/'directions.json');byray={(r['condition'],r['trial'],r['method']):r for r in rays}
    byoverlap=overlap.set_index(['condition','trial']);maxover=maxgap=maxfd=maxvalue=0.;LPchecks=0;negative=separators=0
    for (ci,trial),part in raw.groupby(['condition','trial'],sort=False):
        rec=part.iloc[0];B=int(rec.total_labels);d=int(rec.slopes);train=int(rec.training_labels);m=B-train;N=int(rec.auxiliary_rows)
        seed=[plan['seed'],B,d,train,int(trial)]+([1] if 'proxy_error' in plan else [])
        rng=np.random.default_rng(np.random.SeedSequence(seed));X=rng.standard_normal((B+N,d))
        if 'proxy_error' in plan:
            latent=(rng.uniform(size=B+N)<expit(plan['utility_signal']*X[:,0])).astype(float)
            Z=np.where(rng.uniform(size=B+N)<plan['proxy_error'],1-latent,latent);Y=latent[:B]
            vz=float(np.mean((2*Y[:train]-1)*(2*Z[:train]-1)))
        else:Y=(rng.uniform(size=B)<expit(plan['utility_signal']*X[:B,0])).astype(float);vz=0.
        # Sum explicitly rather than call the primary matrix multiplication.
        v=np.sum(X[:train]*(2*Y[:train,None]-1),axis=0)/train
        xx=np.c_[np.ones(m+N),X[train:]];yy=np.r_[Y[train:],np.zeros(N)]
        g=expit(np.einsum('ij,j->i',X[train:],v)+(vz*(2*Z[train:]-1) if 'proxy_error' in plan else 0.))
        ov=byoverlap.loc[(ci,trial)]
        maxover=max(maxover,abs(v[0]-ov.signal_overlap),abs(v[1:]@v[1:]-ov.noise_norm_squared))
        if 'proxy_error' in plan:maxover=max(maxover,abs(vz-ov.proxy_coefficient))
        for row in part.to_dict('records'):
            method=row['method'];key=(ci,trial,method)
            if key in byray:
                ray=byray[key];direction=np.array(ray['direction'])
                if method=='all_labels_human':
                    margin=float(min((2*Y-1)*(np.c_[np.ones(B),X[:B]]@direction)))
                    assert margin>1-1e-7;maxgap=max(maxgap,abs(margin-ray['minimum_margin']));separators+=1
                else:
                    u=xx@direction;power=row['power']
                    basecost=float(np.mean(np.maximum(u[:m],0)-yy[:m]*u[:m]))
                    correction=float(np.mean(np.maximum(u[m:],0)-g[m:]*u[m:])-np.mean(np.maximum(u[:m],0)-g[:m]*u[:m]))
                    cost=basecost+power*correction
                    assert cost<-1e-8;maxgap=max(maxgap,abs(cost+ray['gap']));negative+=1
            else:assert row['status']=='interior'
            if method!='all_labels_human' and trial<plan['independent_oriented_LP_trials_per_condition']:
                independent=interval_status(xx,yy,g,m,row['power']);assert independent==row['status'],(ci,trial,method,independent,row['status']);LPchecks+=1
        if trial==0:
            beta=np.linspace(-.13,.17,d+1);u=xx@beta
            def direct(beta,power):
                u=xx@beta
                return float(np.mean(np.logaddexp(0,u[:m])-yy[:m]*u[:m])+power*(np.mean(np.logaddexp(0,u[m:])-g[m:]*u[m:])-np.mean(np.logaddexp(0,u[:m])-g[:m]*u[:m])))
            for power in plan['correction_powers']:
                w=np.r_[np.full(m,(1-power)/m),np.full(N,power/N)]
                moment=xx[:m].T@(yy[:m]-power*g[:m])/m+power*xx[m:].T@g[m:]/N
                maxvalue=max(maxvalue,abs(direct(beta,power)-(w@np.logaddexp(0,u)-moment@beta)))
                grad=xx.T@(w*expit(u))-moment
                step=1e-5
                # Three nonadjacent coordinates independently check the gradient.
                for j in [0,1,d]:
                    e=np.eye(d+1)[j];fd=(direct(beta+step*e,power)-direct(beta-step*e,power))/(2*step)
                    maxfd=max(maxfd,abs(fd-grad[j]))
    summary=raw.assign(finite=raw.status.eq('interior')).groupby(['total_labels','slopes','training_labels','method','power'],as_index=False).agg(draws=('trial','size'),finite_events=('finite','sum'))
    summary['finite_frequency']=summary.finite_events/summary.draws
    summary['binomial_MCSE']=np.sqrt(summary.finite_frequency*(1-summary.finite_frequency)/summary.draws)
    original=pd.read_csv(phase/'phase_summary.csv')
    keys=['total_labels','slopes','training_labels','method','power']
    joint=summary.merge(original,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    delta=max(float(max(abs(joint[c+'_new']-joint[c+'_old']))) for c in ['draws','finite_events','finite_frequency','binomial_MCSE'])
    result=dict(model_draws_reconstructed=4000,method_records_reconciled=len(raw),
        original_negative_recession_directions=negative,ordinary_strict_separators=separators,
        independent_oriented_LP_matches=LPchecks,maximum_training_overlap_discrepancy=maxover,
        maximum_original_direction_identity_error=maxgap,maximum_direct_objective_identity_error=maxvalue,
        maximum_direct_gradient_difference=maxfd,maximum_summary_difference=delta,
        all_primary_programs_rerun=False,new_human_or_AI_answers=0,
        original_proofs_independently_peer_reviewed=False)
    assert max(maxover,maxgap,maxvalue,maxfd,delta)<1e-7,result
    result.update(independent_population(plan,population,out))
    a=result['independent_signal_mean'];distribution=[]
    for key,part in overlap.groupby(['total_labels','slopes','training_labels']):
        B,d,train=key;expected=(d-1)/train
        distribution.append(dict(total_labels=int(B),slopes=int(d),training_labels=int(train),
            draws=len(part),mean_signal=float(part.signal_overlap.mean()),signal_target=a,
            signal_mean_z=(float(part.signal_overlap.mean())-a)/np.sqrt((1-a*a)/(train*len(part))),
            mean_noise_squared=float(part.noise_norm_squared.mean()),noise_squared_target=expected,
            noise_squared_mean_z=(float(part.noise_norm_squared.mean())-expected)/np.sqrt(2*(d-1)/(train*train*len(part)))))
    pd.DataFrame(distribution).to_csv(out/'training_moment_diagnostics.csv',index=False)
    save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
    save(out/'verification.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=1):verify(a.base,a.out)
