"""Exploratory panel comparison with whole-record validation and retained sources.

The likelihood integrates a person's entire eight-answer vector. Prediction for
held-out people uses population mixing weights, never their posterior classes.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, expit
from scipy.stats import norm
from threadpoolctl import threadpool_limits
import dh36_observational_analysis as old
from application_support_models import build, save, cell_contrast


class Panel:
    def __init__(self,x,y,kind,nodes=61):
        self.x=x; self.y=y; self.kind=kind; self.n,self.T,self.J,self.K=x.shape
        # Instruments vary only by version. Grouping is an exact computation.
        unique, inv=np.unique(x.reshape(self.n,-1),axis=0,return_inverse=True)
        self.groups=[(np.where(inv==g)[0],row.reshape(self.T,self.J,self.K)) for g,row in enumerate(unique)]
        self.obs=np.eye(self.J)[y]; self.service=np.array([1.,1.,0.])
        self.nodes=nodes
        if kind=='RI':
            z,w=np.polynomial.hermite.hermgauss(nodes); self.z=np.sqrt(2)*z; self.lw=np.log(w)-.5*np.log(np.pi)

    def components(self,b,xx):
        k=self.K; s=self.service
        if self.kind=='MNL':
            u=np.einsum('tjk,k->tj',xx,b)[None]; weights=np.zeros(1)
        elif self.kind=='RI':
            u=np.einsum('tjk,k->tj',xx,b[:k])[None]+np.exp(b[k])*self.z[:,None,None]*s[None,None,:]; weights=self.lw
        elif self.kind=='participation':
            ub=np.einsum('tjk,k->tj',xx,b[:k]); u=np.stack([ub,ub+b[k]*s]); weights=np.array([-np.logaddexp(0,b[-1]),-np.logaddexp(0,-b[-1])])
        elif self.kind=='spike':
            # Classical serial-nonparticipation model; no new-method claim.
            ub=np.einsum('tjk,k->tj',xx,b[:k]); outside=np.zeros_like(ub); outside[:,:2]=-np.inf
            u=np.stack([ub,outside]); weights=np.array([-np.logaddexp(0,b[-1]),-np.logaddexp(0,-b[-1])])
        elif self.kind=='scale':
            ub=np.einsum('tjk,k->tj',xx,b[:k]); u=np.stack([ub,np.exp(b[k])*ub]); weights=np.array([-np.logaddexp(0,b[-1]),-np.logaddexp(0,-b[-1])])
        elif self.kind=='taste':
            # Two class-specific five-parameter utilities; shared version ASC.
            c0=np.r_[b[:5],b[10:12]]; c1=np.r_[b[5:10],b[10:12]]
            u=np.stack([np.einsum('tjk,k->tj',xx,c0),np.einsum('tjk,k->tj',xx,c1)])
            weights=np.array([-np.logaddexp(0,b[-1]),-np.logaddexp(0,-b[-1])])
        else: raise ValueError(self.kind)
        return u,weights

    def eval(self,b,details=False):
        score=np.zeros((self.n,len(b))); lpi=np.zeros(self.n)
        marginal=np.zeros((self.n,self.T,self.J)); refusal=np.zeros((self.n,self.T+1))
        k=self.K
        for use,xx in self.groups:
            u,lw=self.components(b,xx); lp=u-logsumexp(u,axis=-1,keepdims=True); p=np.exp(lp)
            # Index only chosen log probabilities: avoids zero times -infinity
            # for the exact all-outside component.
            lj=lp[:,np.arange(self.T)[None,:],self.y[use]].sum(axis=-1).T+lw
            lpi[use]=logsumexp(lj,axis=1); post=np.exp(lj-lpi[use,None])
            obx=np.einsum('ntj,tjk->nk',self.obs[use],xx,optimize=True)
            ex=np.einsum('rtj,tjk->rk',p,xx,optimize=True)
            ds=self.obs[use,:,:2].sum((1,2))[:,None]-p[:,:,:2].sum((1,2))[None]
            diff=obx[:,None,:]-ex[None]
            if self.kind=='MNL': score[use]=diff[:,0]
            elif self.kind=='RI':
                score[use,:k]=np.einsum('nr,nrk->nk',post,diff,optimize=True)
                score[use,k]=np.sum(post*ds*np.exp(b[k])*self.z,axis=1)
            elif self.kind=='participation':
                score[use,:k]=np.einsum('nr,nrk->nk',post,diff,optimize=True)
                score[use,k]=post[:,1]*ds[:,1]; score[use,-1]=post[:,1]-expit(b[-1])
            elif self.kind=='spike':
                score[use,:k]=post[:,0,None]*diff[:,0]; score[use,-1]=post[:,1]-expit(b[-1])
            elif self.kind=='scale':
                scales=np.array([1.,np.exp(b[k])]); score[use,:k]=np.einsum('nr,nrk,r->nk',post,diff,scales,optimize=True)
                score[use,k]=post[:,1]*np.exp(b[k])*(diff[:,1]@b[:k]); score[use,-1]=post[:,1]-expit(b[-1])
            elif self.kind=='taste':
                score[use,:5]=post[:,0,None]*diff[:,0,:5]; score[use,5:10]=post[:,1,None]*diff[:,1,:5]
                score[use,10:12]=np.einsum('nr,nrk->nk',post,diff[:,:,5:7],optimize=True); score[use,-1]=post[:,1]-expit(b[-1])
            if details:
                w=np.exp(lw); marginal[use]=np.einsum('r,rtj->tj',w,p,optimize=True)
                # Poisson-binomial distribution conditional on each latent draw.
                dist=np.zeros((len(w),self.T+1)); dist[:,0]=1
                for t in range(self.T):
                    q=p[:,t,2]; prev=dist.copy(); dist=prev*(1-q[:,None]); dist[:,1:]+=prev[:,:-1]*q[:,None]
                refusal[use]=w@dist
        result=(-float(lpi.sum()),-score.sum(0))
        return (*result,score,lpi,marginal,refusal) if details else result


def initial_and_bounds(kind,k,basic):
    bounds=[(-12.,12.)]*k
    starts=[]
    if kind=='MNL': starts=[basic.copy()]
    elif kind=='RI':
        starts=[np.r_[basic,np.log(sd)] for sd in [.3,.8,1.6]]; bounds+=[(-5,2)]
    elif kind=='participation':
        for delta,pi in [(1.,.2),(2.5,.5),(4.,.8),(6.,.5)]:
            a=basic.copy(); a[:2]-=pi*delta; starts.append(np.r_[a,delta,np.log(pi/(1-pi))])
        bounds +=[(0.,16.),(-6.,6.)]
    elif kind=='spike':
        starts=[np.r_[basic,logit] for logit in [-3.,-1.,1.]]; bounds +=[(-20.,6.)]
    elif kind=='scale':
        starts=[np.r_[basic,lg,np.log(pi/(1-pi))] for lg,pi in [(.4,.2),(.8,.5),(1.3,.8),(1.8,.5)]]; bounds +=[(0.,3.),(-6.,6.)]
    elif kind=='taste':
        bounds=[(-12.,12.)]*13; bounds[2]=(-3.,0.); bounds[7]=(-3.,0.); bounds[-1]=(-6.,6.)
        for delta,tilt,pi in [(1.,.4,.3),(2.,-.4,.5),(3.,.8,.7),(4.,-.8,.5),(1.,1.2,.5),(3.,0.,.2)]:
            a=basic[:5].copy(); c=a.copy(); a[:2]-=delta*pi; c[:2]+=delta*(1-pi); a[3:5]+=[tilt,-tilt]; c[3:5]-=[tilt,-tilt]
            a[2]=max(-2,min(-.01,a[2]*.65)); c[2]=max(-2,min(-.01,c[2]*1.35))
            starts.append(np.r_[a,c,basic[5:7],np.log(pi/(1-pi))])
    return starts,bounds


def fit(x,y,kind,full=False,nodes=61):
    obj=Panel(x,y,kind,nodes); k=x.shape[-1]
    base=np.zeros(k)
    if k in (7,8): base[2]=-.2
    starts,bounds=initial_and_bounds(kind,k,base)
    opts=[minimize(obj.eval,s,jac=True,method='L-BFGS-B',bounds=bounds,options={'ftol':1e-14,'gtol':1e-7,'maxiter':1800,'maxls':50}) for s in starts]
    opt=min(opts,key=lambda o:o.fun); val,g,score,lpi,probs,refusal=obj.eval(opt.x,True)
    projected=g.copy()
    for j,(lo,hi) in enumerate(bounds):
        if opt.x[j]<=lo+1e-5 and g[j]>0: projected[j]=0
        if opt.x[j]>=hi-1e-5 and g[j]<0: projected[j]=0
    checks={'loglik':-val,'parameters':len(opt.x),'n':len(y),'gradient_max':float(max(abs(g))),'projected_gradient_max':float(max(abs(projected))),'solver_success':bool(opt.success),'start_logliks':[-float(o.fun) for o in opts],'at_bound':[j for j,(lo,hi) in enumerate(bounds) if min(abs(opt.x[j]-lo),abs(opt.x[j]-hi))<1e-5]}
    result={'beta':opt.x,'checks':checks}
    if max(abs(projected))>2e-3: raise ValueError(f'Convergence: {kind} {checks}')
    if full:
        eps=2e-5; eye=np.eye(len(opt.x)); btest=starts[0]*.9
        fd=np.array([(obj.eval(btest+eps*e)[0]-obj.eval(btest-eps*e)[0])/(2*eps) for e in eye]); checks['gradient_check_error']=float(max(abs(fd-obj.eval(btest)[1])))
        if checks['gradient_check_error']>5e-5: raise ValueError('Analytic gradient failed')
        h=np.column_stack([(obj.eval(opt.x+eps*e)[1]-obj.eval(opt.x-eps*e)[1])/(2*eps) for e in eye]); h=(h+h.T)/2; eig=np.linalg.eigvalsh(h)
        hi=np.linalg.pinv(h); cov=len(y)/(len(y)-1)*hi@score.T@score@hi
        result['covariance']=cov; checks['hessian_min_eigenvalue']=float(min(eig))
        if kind=='RI':
            high=Panel(x,y,kind,161); checks['quadrature_61_vs_161_loglik_difference']=float(high.eval(opt.x)[0]-val)
        result['observed_refusal_counts']=np.bincount((y==2).sum(1),minlength=9); result['predicted_refusal_counts']=refusal.sum(0)
        # Aggregate diagnostics only, no answer vectors or per-person outputs.
        result['marginal_loglik']=float(np.log(probs[np.arange(len(y))[:,None],np.arange(8)[None],y]).sum())
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--input-dir',type=Path,required=True); p.add_argument('--instrument',type=Path,required=True); p.add_argument('--folds',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    spec=json.loads(a.instrument.read_text()); d,*_=old.read_exports(a.input_dir,spec); models,names,*_=build(spec,d); y=d[[f'q{i}' for i in range(1,9)]].to_numpy(int)-1
    folds=d[['participant_id']].merge(pd.read_csv(a.folds),on='participant_id',how='left',validate='one_to_one').fold.to_numpy(); fold_ids=np.sort(np.unique(folds)); assert len(folds)==360 and len(fold_ids)==5 and not pd.isna(folds).any()
    definitions=[('MNL_linear','linear','MNL'),('MNL_quadratic','quadratic','MNL'),('MNL_cells','cells','MNL'),('RI_linear','linear','RI'),('RI_cells','cells','RI'),('participation_2class','linear','participation'),('scale_2class','linear','scale'),('taste_2class','linear','taste')]
    fits={}; summary=[]; contrasts=[]; pred={}; foldchecks=[]
    for label,form,kind in definitions:
        x=models[form]; full=fit(x,y,kind,True); fits[label]=full
        print(label,full['checks'],flush=True)
        if form=='cells':
            contrasts.append({'population':'all_360','model':label,**cell_contrast(full['beta'],full['covariance'],names,('G2',89))})
        joint=np.zeros(len(y)); marginal=np.zeros(len(y)); outside=np.zeros((len(y),9))
        for fold in fold_ids:
            train=folds!=fold; test=~train; ff=fit(x[train],y[train],kind); oo=Panel(x[test],y[test],kind)
            _,_,_,ll,pr,rf=oo.eval(ff['beta'],True); joint[test]=ll; marginal[test]=np.log(pr[np.arange(test.sum())[:,None],np.arange(8)[None],y[test]]).sum(1); outside[test]=rf
            foldchecks.append({'model':label,'fold':fold,**ff['checks']})
        pred[label]=joint/8
        summary.append({'model':label,'parameters':len(full['beta']),'in_sample_joint_nll_per_choice':-full['checks']['loglik']/len(y)/8,'fivefold_joint_nll_per_choice':-joint.mean()/8,'fivefold_marginal_nll_per_choice':-marginal.mean()/8,'observed_all_outside':int(np.sum((y==2).all(1))),'fivefold_predicted_all_outside':float(outside[:,8].sum())})
        save(a.out/'panel_model_fits.json',fits); pd.DataFrame(summary).to_csv(a.out/'panel_model_comparison.csv',index=False); save(a.out/'panel_fold_fit_checks.json',foldchecks)
    # This source sensitivity changes no raw labels and certifies no humans.
    nonai=~d.ai_label.to_numpy(); sourcefits={}
    for label,form,kind in [('MNL_cells','cells','MNL'),('RI_cells','cells','RI')]:
        result=fit(models[form][nonai],y[nonai],kind,True); sourcefits[label]=result
        contrasts.append({'population':'300_records_without_AI_source_label','model':label,**cell_contrast(result['beta'],result['covariance'],names,('G2',89))})
    save(a.out/'source_sensitivity_panel_fits.json',sourcefits); pd.DataFrame(contrasts).to_csv(a.out/'panel_same_price_rule_contrasts.csv',index=False)
    paired=[]
    for label in pred:
        delta=pred[label]-pred['RI_linear']; se=delta.std(ddof=1)/np.sqrt(len(delta)); paired.append({'model':label,'reference':'RI_linear','mean_joint_logscore_gain_per_choice':delta.mean(),'paired_descriptive_ci_low':delta.mean()-1.96*se,'paired_descriptive_ci_high':delta.mean()+1.96*se,'caution':'exploratory reused folds; overlapping training sets; not external validation or model-selection-adjusted inference'})
    pd.DataFrame(paired).to_csv(a.out/'paired_prediction_diagnostics.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
