"""Reconstruct original task objectives and residual LPs, independent of compression.

The source reader and fixed split are shared with the primary program. Verification
does not use its Loss, support, fit, tuning or metric implementations.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from run_panel_audit import load
BASE=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def original(data,L,U,g,method,power):
    ridge=.01 if method in ['labelled_ridge','all_human_ridge'] else 0.
    if method.startswith('all_human'):
        people=np.r_[data['train'],L];f=data['f'][people].reshape(-1,3,11)
        targets=np.eye(3)[data['y'][people]].reshape(-1,3)/len(f);w=np.full(len(f),1/len(f))
    elif method.startswith('labelled'):
        f=data['f'][L].reshape(-1,3,11);targets=np.eye(3)[data['y'][L]].reshape(-1,3)/len(f);w=np.full(len(f),1/len(f))
    elif method=='pooling':
        people=np.r_[L,U];f=data['f'][people].reshape(-1,3,11)
        targets=np.concatenate([np.eye(3)[data['y'][L]],g[U]],axis=0).reshape(-1,3)/len(f);w=np.full(len(f),1/len(f))
    else:
        f=np.concatenate([data['f'][L],data['f'][U]],axis=0).reshape(-1,3,11)
        m=6*len(L);N=6*len(U)
        targets=np.r_[(np.eye(3)[data['y'][L]]-power*g[L]).reshape(-1,3)/m,power*g[U].reshape(-1,3)/N]
        w=np.r_[np.full(m,(1-power)/m),np.full(N,power/N)]
    assert np.max(abs(targets.sum(1)-w))<1e-14
    return f,w,targets,ridge

def direct_score(f,w,target,ridge,beta):
    u=np.einsum('njp,p->nj',f,beta);lp=u-logsumexp(u,axis=1)[:,None];prob=np.exp(lp)
    val=float(np.sum(w*logsumexp(u,axis=1))-np.sum(target*u)+ridge*np.sum(beta*beta)/2)
    grad=np.sum((w[:,None]*prob-target)[:,:,None]*f,axis=(0,1))+ridge*beta
    return val,grad

def residual_state(f,w,target):
    keep=w>0;p=f.shape[-1]
    rank=np.linalg.matrix_rank((f[keep,:-1]-f[keep,-1:]).reshape(-1,p))
    if rank<p:return 'rank_deficient',None
    n,J=f.shape[:2];size=n*J
    # r_i = w_i p_i - target_i, each r_i sums to zero.
    eq=sparse.vstack([sparse.kron(sparse.eye(n),np.ones((1,J))),sparse.csr_matrix(f.reshape(size,p).T)])
    eq=sparse.hstack([eq,sparse.csr_matrix((n+p,1))])
    lower=-target.ravel();upper=np.repeat(w,J)-target.ravel()
    ub=sparse.hstack([-sparse.eye(size),sparse.csr_matrix(np.repeat(w,J)[:,None])])
    fit=linprog(np.r_[np.zeros(size),-1.],A_ub=ub,b_ub=target.ravel(),A_eq=eq,b_eq=np.zeros(n+p),bounds=list(zip(lower,upper))+[(0.,1/3)],method='highs')
    if fit.status==2:return 'outside',None
    assert fit.success,(fit.status,fit.message)
    residual=max(float(np.max(abs(eq@fit.x))),max(0.,float(np.max(ub@fit.x-target.ravel()))))
    assert residual<1e-7
    return ('interior' if fit.x[-1]>1e-8 else 'boundary_or_near_boundary'),float(fit.x[-1])

def main(path,results,out):
    plan=json.loads((BASE/'plan.json').read_text());data=load(path,plan)
    status=json.loads((results/'status.json').read_text());assert status['state']=='complete'
    assert status['plan_sha256']==sha(BASE/'plan.json') and status['source_sha256']==sha(path)
    rec=pd.read_csv(results/'method_records.csv');assert len(rec)==16000 and not rec.duplicated(['allocation','labelled_people','method']).any()
    assert rec.groupby(['labelled_people','method']).size().eq(500).all()
    coef={r['record']:np.array(r['beta']) for r in json.loads((results/'coefficients.json').read_text())}
    rays={r['record']:np.array(r['direction']) for r in json.loads((results/'exterior_certificates.json').read_text())}
    pref=json.loads((results/'preflight.json').read_text());bt=np.array(pref['teacher_coefficients'])
    tu=data['teacher']@bt;g=np.exp(tu-logsumexp(tu,axis=2,keepdims=True))
    assert len(coef)==rec.fit.eq('passed_score').sum() and len(rays)==rec.support.eq('outside').sum()
    rng=np.random.default_rng(plan['allocation_seed']);permutations=[rng.permutation(data['pool']) for _ in range(500)]
    # Select at most five per actual state/method/budget after preserving all states.
    selected=set(rec.groupby(['labelled_people','method','support'],sort=True).head(5).record.astype(int))
    maxgrad=maxmetric=maxray=maxscore_difference=max_power_difference=0.;checks=[]
    test=data['test'];fTest=data['f'][test];yTest=data['y'][test]
    for (trial,m),part in rec.groupby(['allocation','labelled_people'],sort=True):
        order=permutations[int(trial)];L=order[:int(m)];U=order[int(m):]
        assert not (set(L)&set(U)) and len(L)+len(U)==138
        pilot_row=part[part.method.eq('labelled_ridge')].iloc[0];pilot=coef[int(pilot_row.record)]
        fp=data['f'][np.r_[L,U]];up=fp@pilot;pp=np.exp(up-logsumexp(up,axis=2,keepdims=True))
        flat=fp.reshape(-1,3,11);pf=pp.reshape(-1,3);mean=(flat*pf[:,:,None]).sum(1)
        H=((flat[:,:,:,None]*flat[:,:,None,:])*pf[:,:,None,None]).sum(axis=(0,1))/len(flat)-mean.T@mean/len(flat)
        pL=pp[:len(L)];pU=pp[len(L):]
        h=((pL-np.eye(3)[data['y'][L]])[:,:,:,None]*data['f'][L]).sum(axis=(1,2))/6
        d=((pL-g[L])[:,:,:,None]*data['f'][L]).sum(axis=(1,2))/6
        dU=((pU-g[U])[:,:,:,None]*data['f'][U]).sum(axis=(1,2))/6
        hs,ds,us=[np.linalg.solve(H,v.T).T for v in [h,d,dU]]
        def trace_at(lam):
            left=hs-lam*ds;left-=left.mean(0);right=us-us.mean(0)
            return (np.sum(left*left)/(len(L)*(len(L)-1))+lam*lam*np.sum(right*right)/(len(U)*(len(U)-1)))
        v0,vm,vp=trace_at(0),trace_at(-1),trace_at(1)
        a=(vp+vm)/2-v0;b=(vp-vm)/2;raw=-b/(2*a)
        tuned=part[part.method.eq('corrected_cluster_power')].iloc[0]
        max_power_difference=max(max_power_difference,abs(raw-tuned.raw_cluster_power))
        assert abs(raw-tuned.raw_cluster_power)<1e-8 and abs(np.clip(raw,0,1)-tuned.power)<1e-8
        for row in part.itertuples(index=False):
            power=float(row.power) if np.isfinite(row.power) else 0.
            f,w,target,ridge=original(data,L,U,g,row.method,power)
            if row.record in rays:
                d=rays[row.record];assert np.sum(abs(d))<=1+1e-8
                u=np.einsum('njp,p->nj',f,d)
                recession=float(np.sum(w*np.max(u,axis=1))-np.sum(target*u))
                assert recession<-1e-8
                maxray=max(maxray,abs(recession+row.separation_gap))
                # Direct large-ray secant additionally checks original uncompressed loss.
                v1=direct_score(f,w,target,0,1e7*d)[0];v2=direct_score(f,w,target,0,2e7*d)[0]
                assert abs((v2-v1)/1e7-recession)<2e-7
            if row.record in coef:
                beta=coef[row.record];value,grad=direct_score(f,w,target,ridge,beta)
                maxgrad=max(maxgrad,float(np.max(abs(grad))));assert max(abs(grad))<1.01e-7
                maxscore_difference=max(maxscore_difference,abs(max(abs(grad))-row.gradient_max))
                u=np.einsum('ntjp,p->ntj',fTest,beta);lp=u-logsumexp(u,axis=2)[:,:,None];prob=np.exp(lp)
                logloss=float(np.mean(-np.take_along_axis(lp,yTest[:,:,None],axis=2)))
                # Brier = 1 + sum(p^2) - 2 p_y, computed directly by person first.
                brier=float(np.mean((1+np.sum(prob*prob,axis=2)-2*np.take_along_axis(prob,yTest[:,:,None],axis=2)[:,:,0]).mean(axis=1)))
                maxmetric=max(maxmetric,abs(logloss-row.heldout_logloss),abs(brier-row.heldout_Brier))
                assert abs(np.linalg.norm(beta)-row.coefficient_norm)<1e-8 and bool(beta[-1]>0)==row.cost_positive
            if row.record in selected and not ridge:
                state,margin=residual_state(f,w,target);assert state==row.support,(row.record,state,row.support,margin)
                checks.append(dict(record=int(row.record),original_row_menus=len(f),method=row.method,support=state,residual_margin=margin))
    assert maxray<1e-9 and maxmetric<1e-10 and maxscore_difference<1e-10
    summaries=json.loads((results/'summary.json').read_text())
    for r in summaries:
        part=rec[(rec.labelled_people==r['labelled_people'])&(rec.method==r['method'])];good=part[part.fit.eq('passed_score')]
        assert len(part)==r['allocations'] and len(good)==r['available'] and part.support.value_counts().to_dict()==r['support_counts']
        if len(good):assert abs(good.heldout_logloss.mean()-r['mean_logloss'])<1e-12
    paired=pd.read_csv(results/'paired_loss.csv');fallback=pd.read_csv(results/'fallback_loss.csv')
    for r in paired.itertuples(index=False):
        pivot=rec[rec.labelled_people==r.labelled_people].pivot(index='allocation',columns='method',values='heldout_logloss')
        d=(pivot[r.method]-pivot[r.baseline]).dropna();assert len(d)==r.shared_allocations
        if len(d):assert abs(d.mean()-r.mean_difference)<1e-12 and abs(d.std(ddof=1)/np.sqrt(len(d))-r.allocation_MCSE)<1e-12
    for r in fallback.itertuples(index=False):
        pivot=rec[rec.labelled_people==r.labelled_people].pivot(index='allocation',columns='method',values='heldout_logloss')
        v=pivot[r.method].fillna(pivot.labelled_ridge);d=v-pivot.labelled_ridge
        assert pivot[r.method].isna().sum()==r.fallback_allocations and abs(v.mean()-r.mean_logloss)<1e-12 and abs(d.mean()-r.paired_excess_over_labelled_ridge)<1e-12
    out.mkdir(parents=True,exist_ok=True)
    result=dict(state='passed',method_records_reconstructed=len(rec),available_coefficients_checked=len(coef),all_exterior_rays_checked=len(rays),independent_original_row_residual_LPs=len(checks),maximum_original_gradient=maxgrad,maximum_score_reconstruction_error=maxscore_difference,maximum_metric_error=maxmetric,maximum_recession_error=maxray,cluster_weights_independently_reconstructed=2000,maximum_cluster_weight_difference=max_power_difference,all_summary_paired_and_fallback_results_reconciled=True,source_reader_and_partition_shared=True,compressed_Loss_support_fitter_tuning_and_metrics_not_imported=True,formal_PPI_gap_resolved=False,population_performance_or_novelty_certified=False,verifier_sha256=sha(Path(__file__)),result_hashes={f.name:sha(f) for f in results.iterdir() if f.is_file()})
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');(out/'residual_LP_checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):main(a.input,a.results,a.out)
