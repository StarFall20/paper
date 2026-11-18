"""Verify original menu constraints, original three-term fits and separate LPs.

Shares only the existing data reader. Does not import run_frontier, its LP,
compressed objective, numerical fitter, or metrics.
"""
from pathlib import Path
import argparse,gzip,hashlib,json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from existing_panel_model import load

BASE=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def original(data,L,U,g,lam,beta=None,direction=None,probability=None):
    fL=data['f'][L].reshape(-1,3,11);fU=data['f'][U].reshape(-1,3,11)
    y=np.eye(3)[data['y'][L]].reshape(-1,3);gL=g[L].reshape(-1,3);gU=g[U].reshape(-1,3)
    answer={}
    if beta is not None:
        grad=np.zeros(11);value=0.
        for f,target,weight in [(fL,y,1.),(fL,gL,-lam),(fU,gU,lam)]:
            z=np.einsum('njp,p->nj',f,beta);lp=z-logsumexp(z,axis=1)[:,None];p=np.exp(lp)
            value+=weight*np.mean(logsumexp(z,axis=1)-np.sum(target*z,axis=1))
            grad+=weight*np.mean(np.sum((p-target)[:,:,None]*f,axis=1),axis=0)
        answer.update(value=float(value),score=float(max(abs(grad))))
    if direction is not None:
        zL=fL@direction;zU=fU@direction
        h=np.mean(np.max(zL,axis=1)-np.sum(y*zL,axis=1))
        predL=np.mean(np.max(zL,axis=1)-np.sum(gL*zL,axis=1))
        predU=np.mean(np.max(zU,axis=1)-np.sum(gU*zU,axis=1))
        answer.update(recession=float(h+lam*(predU-predL)),slope=float(predU-predL))
    if probability is not None:
        pL=probability[data['design_index'][L].ravel()];pU=probability[data['design_index'][U].ravel()]
        moment=np.mean(np.sum(((1-lam)*pL-y+lam*gL)[:,:,None]*fL,axis=1),axis=0)
        moment+=lam*np.mean(np.sum((pU-gU)[:,:,None]*fU,axis=1),axis=0)
        answer['moment_error']=float(np.max(abs(moment)))
    return answer

def raw_frontier(data,L,U,g):
    f=np.r_[data['f'][L].reshape(-1,3,11),data['f'][U].reshape(-1,3,11)]
    nL=6*len(L);nU=6*len(U);n=nL+nU
    y=np.eye(3)[data['y'][L]].reshape(-1,3);gL=g[L].reshape(-1,3);gU=g[U].reshape(-1,3)
    w0=np.r_[np.full(nL,1/nL),np.zeros(nU)];dw=np.r_[np.full(nL,-1/nL),np.full(nU,1/nU)]
    b0=np.sum(f[:nL]*y[:,:,None],axis=(0,1))/nL
    db=-np.sum(f[:nL]*gL[:,:,None],axis=(0,1))/nL+np.sum(f[nL:]*gU[:,:,None],axis=(0,1))/nU
    eq=sparse.vstack([sparse.kron(sparse.eye(n),np.ones((1,3))),sparse.csr_matrix(f.reshape(-1,11).T)])
    eq=sparse.hstack([eq,sparse.csr_matrix(np.r_[-dw,-db][:,None])]).tocsr()
    ans=linprog(np.r_[np.zeros(n*3),-1.],A_eq=eq,b_eq=np.r_[w0,b0],
                bounds=[(0,None)]*(n*3)+[(0,1)],method='highs-ds',
                options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    assert ans.success,(ans.status,ans.message)
    return float(ans.x[-1])

def main(path,results,out):
    assert not out.exists(),'Use a fresh verification directory.'
    status=json.loads((results/'status.json').read_text());assert status['state']=='complete'
    assert status['plan_sha256']==sha(BASE/'plan.json') and status['source_sha256']==sha(path)
    assert status['analysis_sha256']==sha(BASE/'analysis/run_frontier.py')
    model_plan=json.loads((BASE/'inputs/model_plan.json').read_text());data=load(path,model_plan)
    pre=json.loads((BASE/'inputs/preflight.json').read_text());u=data['teacher']@np.array(pre['teacher_coefficients'])
    g=np.exp(u-logsumexp(u,axis=2,keepdims=True))
    fr=pd.read_csv(results/'frontiers.csv');rec=pd.read_csv(results/'policy_records.csv')
    coef={r['record']:np.array(r['beta']) for r in json.loads((results/'coefficients.json').read_text())}
    cert=json.loads(gzip.decompress((results/'certificates.json.gz').read_bytes()))
    assert len(fr)==2000 and len(rec)==4000 and len(cert)==2000
    assert fr.frontier.between(0,1).all() and rec.power.between(0,1).all()
    assert np.max(abs(fr.frontier-fr.raw_frontier))<1e-9
    assert len(coef)==rec.fit.eq('passed_score').sum()
    # First four members of every actual budget/cap/midpoint state, chosen without error or forecast filtering.
    selected=set(fr.groupby(['labelled_people','frontier_capped_at_one','midpoint_state'],sort=True).head(4).frontier_record.astype(int))
    maxima=dict(primal_moment=0.,dual_recession=0.,score=0.,metric=0.,raw_frontier=0.)
    checked=[];rng=np.random.default_rng(model_plan['allocation_seed']);orders=[rng.permutation(data['pool']) for _ in range(500)]
    for row in fr.itertuples(index=False):
        order=orders[row.allocation];m=int(row.labelled_people);L,U=order[:m],order[m:];cap=row.frontier
        witness=cert[row.frontier_record];assert witness['frontier_record']==row.frontier_record
        weighted=np.array(witness['weighted_probabilities'])
        wm=weighted.sum(axis=1);assert np.min(weighted)>-1e-8
        p=np.full_like(weighted,1/3);keep=wm>1e-12;p[keep]=weighted[keep]/wm[keep,None]
        original_weights=(1-cap)*np.bincount(data['design_index'][L].ravel(),minlength=34)/(6*m)+cap*np.bincount(data['design_index'][U].ravel(),minlength=34)/(6*len(U))
        assert max(abs(wm-original_weights))<1e-7
        error=original(data,L,U,g,cap,probability=p)['moment_error'];maxima['primal_moment']=max(maxima['primal_moment'],error);assert error<1e-7
        if cap<1-1e-9:
            d=np.array(witness['direction']);rr=original(data,L,U,g,cap,direction=d)
            maxima['dual_recession']=max(maxima['dual_recession'],abs(rr['recession']))
            assert abs(rr['recession'])<1e-7 and rr['slope']<-.999999
            assert original(data,L,U,g,cap+1e-3,direction=d)['recession']<-.0009
        for policy in rec[rec.frontier_record.eq(row.frontier_record)].itertuples(index=False):
            expected=cap/2 if policy.method=='half_frontier' else np.clip(row.original_cluster_power,.1*cap,.9*cap)
            assert abs(policy.power-expected)<1e-12
            if policy.support=='interior':
                state=witness['policies'][policy.method];p=np.full((34,3),1/3)
                p[np.array(state['kept_designs'])]=np.array(state['probabilities'])
                assert np.min(p)>1e-8-1e-9 and max(abs(p.sum(1)-1))<1e-7
                er=original(data,L,U,g,policy.power,probability=p)['moment_error'];assert er<1e-7
                maxima['primal_moment']=max(maxima['primal_moment'],er)
                f=data['f'][np.r_[L,U]].reshape(-1,3,11)
                assert np.linalg.matrix_rank((f[:,:-1]-f[:,-1:]).reshape(-1,11))==11
            if policy.record in coef:
                beta=coef[policy.record];sc=original(data,L,U,g,policy.power,beta=beta)['score'];maxima['score']=max(maxima['score'],sc);assert sc<1.01e-7
                z=data['f'][data['test']]@beta;lp=z-logsumexp(z,axis=2,keepdims=True);prob=np.exp(lp);y=data['y'][data['test']]
                picked=np.take_along_axis(lp,y[:,:,None],axis=2)[:,:,0]
                loss=float((-picked).mean(axis=1).mean())
                brier=float((1+np.sum(prob**2,axis=2)-2*np.take_along_axis(prob,y[:,:,None],axis=2)[:,:,0]).mean(axis=1).mean())
                maxima['metric']=max(maxima['metric'],abs(loss-policy.heldout_logloss),abs(brier-policy.heldout_Brier))
                assert maxima['metric']<1e-10
        if row.frontier_record in selected:
            separate=raw_frontier(data,L,U,g);delta=abs(separate-cap);assert delta<1e-7
            maxima['raw_frontier']=max(maxima['raw_frontier'],delta)
            checked.append(dict(record=int(row.frontier_record),original_menus=828,power=separate))
    old=pd.read_csv(BASE/'inputs/baseline_records.csv')
    for s in json.loads((results/'summary.json').read_text()):
        group=rec[rec.labelled_people.eq(s['labelled_people'])&rec.method.eq(s['method'])]
        assert len(group)==s['denominator']==500 and group.fit.eq('passed_score').sum()==s['available']
        assert group.support.value_counts().to_dict()==s['support_counts']
        pivot=old[old.labelled_people.eq(s['labelled_people'])].pivot(index='allocation',columns='method',values='heldout_logloss')
        vals=group.set_index('allocation').heldout_logloss.reindex(pivot.index).fillna(pivot.labelled_ridge)
        assert abs(vals.mean()-s['all_allocation_fallback_mean_logloss'])<1e-12
        for b,r in s['comparisons'].items():
            diff=vals-pivot[b].fillna(pivot.labelled_ridge)
            assert abs(diff.mean()-r['mean'])<1e-12 and abs(diff.std(ddof=1)/np.sqrt(500)-r['allocation_MCSE'])<1e-12
    out.mkdir(parents=True)
    answer=dict(state='passed',frontiers_checked=2000,policy_records_checked=4000,finite_coefficients_checked=len(coef),
                original_row_frontier_LPs=len(checked),maximum_errors=maxima,source_reader_shared=True,
                primary_frontier_support_objective_fitter_metrics_not_imported=True,new_inference_or_originality_certified=False,
                verifier_sha256=sha(Path(__file__)),result_hashes={p.name:sha(p) for p in results.iterdir() if p.is_file()})
    (out/'verification.json').write_text(json.dumps(answer,indent=2)+'\n');(out/'original_row_LPs.json').write_text(json.dumps(checked,indent=2)+'\n')
    print(json.dumps(answer))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=1):main(a.input,a.results,a.out)
