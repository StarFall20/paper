"""Independent original-loss, influence-score and interval checks for the audit."""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy import sparse
from scipy.special import expit
from threadpoolctl import threadpool_limits
from public_conjoint_optimizer_audit import load_data, save
from public_conjoint_matched_audit import nuisance
from observable_low_label_path import ridge_anchor


def interval_status(x,y,g,m,lam):
    N=len(x)-m
    lower=np.r_[-y[:m]+lam*g[:m],-m*lam*g[m:]/N]
    width=np.r_[np.full(m,1-lam),np.full(N,m*lam/N)]
    n=len(x);eye=sparse.eye(n,format='csr');widthcol=sparse.csr_matrix(width[:,None])
    aub=sparse.vstack([sparse.hstack([-eye,widthcol]),sparse.hstack([eye,widthcol])],format='csr')
    fit=linprog(np.r_[np.zeros(n),-1.],A_ub=aub,b_ub=np.r_[-lower,lower+width],
        A_eq=sparse.hstack([sparse.csr_matrix(x.T),sparse.csr_matrix((x.shape[1],1))]),
        b_eq=np.zeros(x.shape[1]),bounds=list(zip(lower,lower+width))+[(0.,.5)],method='highs-ipm')
    if fit.status==2:return 'outside_mean_support',None
    assert fit.success,fit.message
    assert max(abs(x.T@fit.x[:-1]))<1e-7
    return ('interior' if fit.x[-1]>1e-8 else 'boundary_or_near_boundary'),float(fit.x[-1])


def influence_calculation(x,y,g,m,b,power):
    mu=expit(x@b);N=len(x)-m
    info=np.einsum('ij,i,ik->jk',x,mu*(1-mu),x)/len(x)
    # Solve for row influences instead of multiplying covariance matrices.
    influence=np.linalg.solve(info,x.T).T
    h=influence[:m]*(mu[:m]-y[:m])[:,None]
    p=influence*(mu-g)[:,None]
    hc=h-h.mean(0);pc=p[:m]-p[:m].mean(0)
    a=float(np.mean(np.sum(hc*hc,axis=1)))
    cross=2*float(np.mean(np.sum(hc*pc,axis=1)))
    bpred=float(np.var(p,axis=0,ddof=1).sum())
    tuned=float(np.clip(cross/(2*(1+m/N)*bpred),0.,1.))
    corrected=h-power*p[:m]
    psd=float(np.var(corrected,axis=0,ddof=1).sum()/m+power**2*np.var(p[m:],axis=0,ddof=1).sum()/N)
    quadratic=(a-power*cross+(1+m/N)*power**2*bpred)/m
    return tuned,psd,quadratic


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();base=args.base;out=args.out;out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((base/'plan.json').read_text());status=json.loads((base/'results/status.json').read_text())
    assert status['state']=='complete' and status['method_records']==36000
    records=base/'results/records.csv'
    if not records.exists():records=records.with_suffix('.csv.gz')
    frame=pd.read_csv(records);assert len(frame)==36000
    assert not frame.duplicated(['trial','label_rows','prediction','auxiliary_rows','method']).any()
    dirs=json.loads((base/'results/exterior_directions.json').read_text())
    key=lambda r:(int(r['trial']),int(r['label_rows']),r['prediction'],int(r['auxiliary_rows']),r['method'])
    directions={key(d):d for d in dirs};assert len(directions)==len(dirs)
    humans={ (int(r['trial']),int(r['label_rows'])):r for r in json.loads((base/'results/ordinary_checks.json').read_text()) }
    assert len(humans)==1000
    x,y,z,_,_=load_data(args.input_dir);x=x[:1200];y=y[:1200]
    rng=np.random.RandomState(plan['seed']);positive_lp=human_lp=negative=0
    maxpower=maxtrace=maxquad=maxdirect=maxscore=0.;actual=[]
    for trial in range(plan['allocations']):
        order=(rng.permutation(240)[:,None]*5+np.arange(5)[None]).reshape(-1)
        xx=x[order];zz=z[order]
        for m in plan['label_budgets']:
            yy=np.r_[y[order[:m]],np.full(1200-m,np.nan)]
            pilot,err=ridge_anchor(xx[:m],yy[:m]);mu=expit(xx[:m]@pilot)
            score=xx[:m].T@(mu-yy[:m])/m+pilot/m
            maxscore=max(maxscore,float(max(abs(score))));assert max(abs(score))<1e-9
            ordinary=humans[(trial,m)]
            # A separate soft-probability moment LP checks overlap on every human design.
            hfit=linprog(np.r_[np.zeros(m),-1.],
                A_ub=np.r_[np.c_[-np.eye(m),np.ones(m)],np.c_[np.eye(m),np.ones(m)]],
                b_ub=np.r_[np.zeros(m),np.ones(m)],
                A_eq=np.c_[xx[:m].T,np.zeros(xx.shape[1])],b_eq=xx[:m].T@yy[:m],
                bounds=[(0.,1.)]*m+[(0.,.5)],method='highs-ipm')
            assert hfit.success;human_lp+=1
            assert (hfit.x[-1]>1e-8)==(ordinary['status']=='overlap')
            if ordinary['status']=='nonattained_separated':
                d=np.asarray(ordinary['direction']);margins=(2*yy[:m]-1)*(xx[:m]@d)
                assert min(margins)>-1e-8 and margins.mean()>1e-8
            for pred,use_ai in [('AI',True),('X_only',False)]:
                g,fallback=nuisance(xx,yy,zz,m,use_ai)
                group=frame[(frame.trial==trial)&(frame.label_rows==m)&(frame.prediction==pred)]
                assert len(group)==18
                initial=None
                for n in [int(v) if isinstance(v,int) else 1200-m for v in plan['auxiliary_budgets']]:
                    nx=xx[:m+n];ny=yy[:m+n];ng=g[:m+n]
                    tuned,_,_=influence_calculation(nx,ny,ng,m,pilot,0.)
                    if initial is None:initial=tuned
                    for method,power in [('source_tuned_each_N',tuned),('fixed_first_N_power',initial),('unit',1.)]:
                        row=group[(group.auxiliary_rows==n)&(group.method==method)].iloc[0]
                        assert row.ordinary_status==ordinary['status'] and int(row.nuisance_fallbacks)==fallback
                        _,trace,quadratic=influence_calculation(nx,ny,ng,m,pilot,float(row.power))
                        maxpower=max(maxpower,abs(float(row.power)-power))
                        maxtrace=max(maxtrace,abs(trace-row.psd_score_trace))
                        maxquad=max(maxquad,abs(quadratic-row.source_quadratic_trace))
                        assert abs(row.power-power)<1e-9
                        assert abs(trace-row.psd_score_trace)<1e-9*max(1.,abs(trace))
                        assert abs(quadratic-row.source_quadratic_trace)<1e-9*max(1.,abs(quadratic))
                        k=(trial,m,pred,n,method)
                        if row.status=='outside_mean_support':
                            d=np.asarray(directions[k]['direction']);u=nx@d
                            human=float(np.mean(np.maximum(u[:m],0)-ny[:m]*u[:m]))
                            aux=float(np.mean(np.maximum(u[m:],0)-ng[m:]*u[m:]))
                            predlabel=float(np.mean(np.maximum(u[:m],0)-ng[:m]*u[:m]))
                            recession=human+row.power*(aux-predlabel)
                            maxdirect=max(maxdirect,abs(recession-directions[k]['original_recession']))
                            assert recession<-1e-8 and abs(recession+row.separation_gap)<1e-9
                            negative+=1
                        if row.power==0:
                            assert row.status==('interior' if ordinary['status']=='overlap' else 'boundary_or_near_boundary')
                        elif trial%25==0 or row.status=='boundary_or_near_boundary':
                            other,margin=interval_status(nx,ny,ng,m,float(row.power));positive_lp+=1
                            assert other==row.status,(k,row.status,other,margin)
                        actual.append(dict(trial=trial,label_rows=m,prediction=pred,auxiliary_rows=n,method=method,status=row.status))
        if (trial+1)%50==0:print(json.dumps(dict(verified_allocations=trial+1,checked_records=len(actual))),flush=True)
    assert negative==len(dirs)
    # Reconcile every published pair without calling the primary summary routine.
    paired=pd.read_csv(base/'results/paired_transitions.csv');pairrows=0
    for r in paired.to_dict('records'):
        group=frame[(frame.label_rows==r['label_rows'])&(frame.prediction==r['prediction'])&(frame.method==r['method'])]
        a=group[group.auxiliary_rows==r['from_auxiliary_rows']].set_index('trial')
        b=group[group.auxiliary_rows==r['to_auxiliary_rows']].set_index('trial')
        lost=a.status.eq('interior')&b.status.eq('outside_mean_support')
        gained=a.status.eq('outside_mean_support')&b.status.eq('interior')
        assert int(lost.sum())==r['finite_to_exterior'] and int(gained.sum())==r['exterior_to_finite']
        assert int((lost&(b.psd_score_trace<a.psd_score_trace)).sum())==r['finite_to_exterior_with_lower_psd_score_trace']
        pairrows+=1
    save(out/'verification.json',dict(state='passed',method_records_reconstructed=len(actual),ordinary_design_LPs=human_lp,
        independent_positive_power_interval_LPs=positive_lp,original_negative_recession_directions=negative,
        paired_transition_rows=pairrows,maximum_power_difference=maxpower,maximum_PSD_trace_difference=maxtrace,
        maximum_source_quadratic_trace_difference=maxquad,maximum_original_recession_difference=maxdirect,
        maximum_direct_ridge_pilot_score=maxscore,all_primary_support_programs_rerun=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),new_human_or_AI_answers=0))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
