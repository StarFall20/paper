"""Worked independent-training case of the existing compensated boundary.

The empirical signed-label feature mean is an established simple classifier.
Its LLN/chi-square training limit is not a new learning algorithm. All labels
generated here are statistical simulations, not human or LLM responses.
"""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize, linprog
from scipy.special import expit, roots_hermitenorm
from scipy.stats import norm
from threadpoolctl import threadpool_limits
from finite_ratio_signal_boundary import B_terms, tail, tail_derivative
from power_tuned_choice_audit import objective, support, WeightedLogit


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def signal_mean(gamma, nodes=160):
    s,w=roots_hermitenorm(nodes)
    return float((w/np.sqrt(2*np.pi))@(s*np.tanh(gamma*s/2)))


def ordinary_boundary(gamma):
    s,w=roots_hermitenorm(200);w=w/np.sqrt(2*np.pi);p=expit(gamma*s)
    grad=lambda h:float(w@((1-p)*tail_derivative(h*s)*s-p*tail_derivative(-h*s)*s))
    h=brentq(grad,-20.,20.)
    return float(w@((1-p)*tail(h*s)+p*tail(-h*s)))


class LearnedPopulation:
    def __init__(self,gamma,a,b,tau,power,nodes=60):
        z,w=roots_hermitenorm(nodes);w=w/np.sqrt(2*np.pi)
        S,T=np.meshgrid(z,z,indexing='ij');W=np.outer(w,w).ravel()
        S=S.ravel();T=T.ravel();p=expit(gamma*S)
        self.S=np.tile(S,2);self.T=np.tile(T,2)
        self.y=np.repeat([0.,1.],len(S));self.weights=np.r_[W*(1-p),W*p]
        self.V=np.c_[np.ones(len(self.S)),self.S,self.T]
        self.g=expit(a*self.S+b*self.T);self.sign=1-2*self.y
        r=np.where(self.y==0,self.g,1-self.g)
        self.lo=power*r;self.hi=1-power+power*r
        self.ulo=-power*self.g/tau;self.uhi=power*(1-self.g)/tau
        self.tau=tau;self.power=power
        assert abs(self.weights.sum()-1)<1e-12

    def value(self,eta,t):
        u=self.V@eta
        l,dl,tl=B_terms(self.sign*u,t,self.lo,self.hi)
        r,dr,tr=B_terms(u,t,self.ulo,self.uhi)
        value=float(self.weights@(l+self.tau*r))
        grad=self.V.T@(self.weights*(self.sign*dl+self.tau*dr))
        dt=float(self.weights@(tl+self.tau*tr))
        return value,grad,dt

    def maximize(self,eta):
        first=self.value(eta,0.)
        if first[2]<=0:return 0.,np.zeros(3),0.
        upper=1.
        while self.value(eta,upper)[2]>0:
            upper*=2
            if upper>1e7:raise RuntimeError('Unbracketed cone scale')
        t=brentq(lambda t:self.value(eta,t)[2],0.,upper,xtol=1e-11)
        v,g,_=self.value(eta,t)
        return v,g,t

    def critical(self):
        # Joint sign symmetry fixes only the intercept multiplier to zero.
        def fn(e):
            value,gradient,_=self.maximize(np.r_[0.,e])
            return value,gradient[1:]
        fit=minimize(fn,np.zeros(2),method='BFGS',jac=True,
                     options={'gtol':1e-9,'maxiter':300})
        eta=np.r_[0.,fit.x];v,g,t=self.maximize(eta)
        if max(abs(g))>1e-7:raise RuntimeError(('Population stationary residual',g,fit.message))
        return dict(critical_ratio=v,signal_multiplier=float(eta[1]),
                    training_noise_multiplier=float(eta[2]),scale=t,
                    gradient_max=float(max(abs(g))),scale_derivative=self.value(eta,t)[2])


def population(plan,out):
    out.mkdir(parents=True,exist_ok=True);rows=[]
    gamma=plan['utility_signal'];a=signal_mean(gamma);ordinary=ordinary_boundary(gamma)
    quad=gradient=symmetry=0.
    for rho in plan['population_dimension_ratios']:
      for alpha in plan['training_fractions']:
       for nu in plan['population_auxiliary_rows_per_total_label']:
        b=np.sqrt(rho/alpha);tau=nu/(1-alpha)
        for power in plan['correction_powers']:
            low=LearnedPopulation(gamma,a,b,tau,power,40)
            fine=LearnedPopulation(gamma,a,b,tau,power,80)
            l=low.critical();f=fine.critical()
            quad=max(quad,abs(l['critical_ratio']-f['critical_ratio']))
            eta=np.array([0.,f['signal_multiplier'],f['training_noise_multiplier']])
            delta=1e-5;eye=np.eye(3)
            fd=np.array([(fine.maximize(eta+delta*e)[0]-fine.maximize(eta-delta*e)[0])/(2*delta) for e in eye])
            gradient=max(gradient,float(max(abs(fd-fine.maximize(eta)[1]))))
            symmetry=max(symmetry,abs(fine.maximize(eta+np.array([.2,0.,0.]))[0]-fine.maximize(eta-np.array([.2,0.,0.]))[0]))
            kappa=rho/(1-alpha)
            rows.append(dict(total_slope_ratio=rho,training_fraction=alpha,
                auxiliary_per_total_label=nu,power=power,training_signal_limit=a,
                training_noise_norm_limit=b,auxiliary_per_correction_label=tau,
                correction_slope_ratio=kappa,**f,
                strict_gap=f['critical_ratio']-kappa,
                direct_human_boundary=ordinary,direct_human_gap=ordinary-rho))
            print(json.dumps({'population_completed':len(rows),'rho':rho,'alpha':alpha,'nu':nu,'power':power}),flush=True)
    pd.DataFrame(rows).to_csv(out/'population.csv',index=False)
    checks=dict(conditions=len(rows),signal_mean=a,direct_human_boundary=ordinary,
        quadrature_40_80_max_difference=quad,gradient_finite_difference_max_error=gradient,
        intercept_sign_symmetry_error=symmetry,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(out/'population_attempt.json',checks)
    assert quad<2e-6 and gradient<1e-7 and symmetry<1e-9,checks
    save(out/'population_checks.json',checks)


def generate(plan,condition,trial):
    budget,d,train=condition;m=budget-train;N=round(plan['phase_auxiliary_rows_per_total_label']*budget)
    rng=np.random.default_rng(np.random.SeedSequence([plan['seed'],budget,d,train,trial]))
    X=rng.normal(size=(budget+N,d))
    y=(rng.random(budget)<expit(plan['utility_signal']*X[:budget,0])).astype(float)
    v=X[:train].T@(2*y[:train]-1)/train
    audit=np.c_[np.ones(m+N),X[train:]]
    outcomes=np.r_[y[train:],np.zeros(N)]
    g=expit(X[train:]@v)
    assert min(g)>0 and max(g)<1
    return X,y,v,audit,outcomes,g


def phase(plan,planpath,out):
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'phase_trials.csv').exists(),'Use a fresh output directory'
    rows=[];rays=[];training=[];start=time.monotonic()
    specs=[(B,round(B*rho),round(B*alpha)) for B in plan['total_label_budgets']
           for rho in plan['phase_dimension_ratios'] for alpha in plan['training_fractions']]
    sourcehash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for ci,spec in enumerate(specs):
      B,d,train=spec;m=B-train;N=round(plan['phase_auxiliary_rows_per_total_label']*B)
      for trial in range(plan['repetitions_per_condition']):
        X,y,v,xx,yy,g=generate(plan,spec,trial)
        training.append(dict(condition=ci,trial=trial,total_labels=B,slopes=d,training_labels=train,
            signal_overlap=float(v[0]),noise_norm_squared=float(v[1:]@v[1:]),
            label_one_fraction=float(y[:train].mean())))
        allx=np.c_[np.ones(B),X[:B]]
        human=WeightedLogit(allx,np.full(B,1/B),allx.T@y/B)
        methods=[('all_labels_human',0.,human)]+[(f'corrected_{power:g}',power,objective(xx,yy,g,m,power)) for power in plan['correction_powers']]
        for method,power,loss in methods:
            result=support(loss)
            row=dict(condition=ci,trial=trial,total_labels=B,slopes=d,training_labels=train,
                correction_labels=m,auxiliary_rows=N,method=method,power=power,status=result['status'],
                interior_margin=result.get('interior_margin'),equality_residual=result.get('equality_residual'),
                separation_gap=result.get('separation_gap'))
            if 'direction' in result:
                rays.append(dict(condition=ci,trial=trial,method=method,direction=result['direction'],gap=result['separation_gap']))
            if result['status']=='boundary_or_near_boundary' and method=='all_labels_human':
                signed=(2*y-1)[:,None]*allx
                sep=linprog(np.zeros(d+1),A_ub=-signed,b_ub=-np.ones(B),bounds=[(None,None)]*(d+1),method='highs')
                if sep.success and min(signed@sep.x)>1-1e-7:
                    row['status']='ordinary_strict_separation'
                    rays.append(dict(condition=ci,trial=trial,method=method,direction=sep.x.tolist(),minimum_margin=float(min(signed@sep.x))))
            rows.append(row)
        if (trial+1)%25==0:
            pd.DataFrame(rows).to_csv(out/'phase_trials.csv',index=False)
            save(out/'run_status.json',dict(state='running',model_draws=len(training),method_records=len(rows),source_sha256=sourcehash,plan_sha256=hashlib.sha256(planpath.read_bytes()).hexdigest()))
            print(json.dumps({'condition':ci,'condition_draws':trial+1,'total_draws':len(training),'seconds':round(time.monotonic()-start,1)}),flush=True)
      pd.DataFrame(training).to_csv(out/'training_overlaps.csv',index=False)
      save(out/'directions.json',rays)
    frame=pd.DataFrame(rows);frame.to_csv(out/'phase_trials.csv',index=False)
    frame.groupby(['total_labels','slopes','training_labels','method','status'],as_index=False).size().to_csv(out/'status_counts.csv',index=False)
    summary=frame.assign(finite=frame.status.eq('interior')).groupby(['total_labels','slopes','training_labels','method','power'],as_index=False).agg(draws=('trial','size'),finite_events=('finite','sum'))
    summary['finite_frequency']=summary.finite_events/summary.draws
    summary['binomial_MCSE']=np.sqrt(summary.finite_frequency*(1-summary.finite_frequency)/summary.draws)
    summary.to_csv(out/'phase_summary.csv',index=False)
    paired=[]
    for keys,part in frame.groupby(['total_labels','slopes','training_labels']):
        pivot=part.assign(finite=part.status.eq('interior')).pivot(index='trial',columns='method',values='finite').astype(float)
        for method in pivot.columns:
            if method=='all_labels_human':continue
            diff=pivot[method]-pivot.all_labels_human
            paired.append(dict(zip(['total_labels','slopes','training_labels'],keys),method=method,
                mean_difference=float(diff.mean()),MCSE=float(diff.std(ddof=1)/np.sqrt(len(diff))),
                corrected_only=int((diff==1).sum()),human_only=int((diff==-1).sum()),draws=len(diff)))
    pd.DataFrame(paired).to_csv(out/'paired_events.csv',index=False)
    status=dict(state='complete',model_draws=len(training),method_records=len(rows),directions=len(rays),
        source_sha256=sourcehash,plan_sha256=hashlib.sha256(planpath.read_bytes()).hexdigest(),
        nonstandard_states=frame.loc[~frame.status.isin(['interior','outside_mean_support','ordinary_strict_separation']),'status'].value_counts().to_dict(),
        new_human_or_AI_answers=0,forecast_or_interval_performance_tested=False)
    save(out/'run_status.json',status);print(json.dumps(status),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['population','phase'],required=True)
    a=p.parse_args();plan=json.loads(a.plan.read_text())
    with threadpool_limits(limits=1):
        if a.mode=='population':population(plan,a.out)
        else:phase(plan,a.plan,a.out)
