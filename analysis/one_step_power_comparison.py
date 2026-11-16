"""Reconstruct original draws and audit the one-step comparator's power domain.

Implements the stated formulas independently of downloaded software. Old model
sources and results are read only. A finite update is not a finite global root.
"""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

METHODS=('pooled_clipped','pooled_unclipped','labelled_clipped','labelled_unclipped','human_ridge')

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def write_json(p,a):
    p.write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def components(x,y,g,m,b):
    mu=expit(x@b);v=mu*(1-mu);N=len(x)-m
    hl=x[:m].T@(v[:m,None]*x[:m])/m
    hu=x[m:].T@(v[m:,None]*x[m:])/N
    true=x[:m]*(mu[:m]-y[:m])[:,None]
    pred=x*(mu-g)[:,None]
    return hl,hu,true,pred

def scalar_weight(true,pred,information,m,N):
    try:
        if np.linalg.matrix_rank(information)<information.shape[0]:return None,'information_rank_deficient'
        inv=np.linalg.inv(information)
    except np.linalg.LinAlgError:return None,'information_solve_failed'
    aa=true-true.mean(0);bb=pred[:m]-pred[:m].mean(0)
    cross=(aa.T@bb+bb.T@aa)/m
    denominator=2*(1+m/N)*np.trace(inv@np.cov(pred.T)@inv)
    if not np.isfinite(denominator) or denominator<=1e-14:return None,'weight_unresolved'
    raw=float(np.trace(inv@cross@inv)/denominator)
    return (raw,'weight_computed') if np.isfinite(raw) else (None,'weight_unresolved')

def update(pilot,hl,hu,true,pred,m,lam):
    if lam is None:return None,{'status':'weight_unresolved'}
    h=(1-lam)*hl+lam*hu
    gradient=true.mean(0)+lam*(pred[m:].mean(0)-pred[:m].mean(0))
    eig=np.linalg.eigvalsh(h)
    diagnostics={'curvature_min_eigenvalue':float(eig[0]),'curvature_max_eigenvalue':float(eig[-1]),
                 'curvature_condition':float(np.linalg.cond(h))}
    try: direction=np.linalg.solve(h,gradient)
    except np.linalg.LinAlgError:return None,{'status':'step_solve_failed',**diagnostics}
    residual=np.linalg.norm(h@direction-gradient,np.inf)
    scale=np.linalg.norm(h,np.inf)*np.linalg.norm(direction,np.inf)+np.linalg.norm(gradient,np.inf)
    backward=float(residual/max(scale,np.finfo(float).tiny))
    beta=pilot-direction
    diagnostics.update(update_backward_error=backward,step_norm=float(np.linalg.norm(direction)))
    if not np.isfinite(beta).all() or not np.isfinite(backward) or backward>1e-8:
        return None,{'status':'step_numerically_unresolved',**diagnostics}
    return beta,{'status':'finite_update',**diagnostics}

def analytic_checks():
    rng=np.random.default_rng(10579);x=rng.normal(size=(29,4));y=rng.integers(0,2,29).astype(float)
    g=expit(rng.normal(size=29));b=rng.normal(size=4)/3;m=11
    hl,hu,true,pred=components(x,y,g,m,b);eps=1e-5;errors=[]
    def value(beta,lam):
        z=x@beta;l=np.logaddexp(0,z)
        return np.mean(l[:m]-y[:m]*z[:m])+lam*(np.mean(l[m:]-g[m:]*z[m:])-np.mean(l[:m]-g[:m]*z[:m]))
    for lam in [-.75,.2,1.5]:
        grad=true.mean(0)+lam*(pred[m:].mean(0)-pred[:m].mean(0))
        fd=np.array([(value(b+eps*np.eye(4)[j],lam)-value(b-eps*np.eye(4)[j],lam))/(2*eps) for j in range(4)])
        hd=[]
        for j in range(4):
            def score(beta):
                _,_,t,p=components(x,y,g,m,beta)
                return t.mean(0)+lam*(p[m:].mean(0)-p[:m].mean(0))
            hd.append((score(b+eps*np.eye(4)[j])-score(b-eps*np.eye(4)[j]))/(2*eps))
        errors.append({'lambda':lam,'gradient_error':float(np.max(abs(fd-grad))),'Hessian_error':float(np.max(abs(np.array(hd).T-((1-lam)*hl+lam*hu))))})
    assert max(max(r['gradient_error'],r['Hessian_error']) for r in errors)<1e-8
    information=(m*hl+(len(x)-m)*hu)/len(x);lam,_=scalar_weight(true,pred,information,m,len(x)-m)
    inv=np.linalg.inv(information);at=(true-true.mean(0))@inv;bt=(pred[:m]-pred[:m].mean(0))@inv
    independent=float(np.sum(at*bt)/m/((1+m/(len(x)-m))*np.var(pred@inv,axis=0,ddof=1).sum()))
    assert abs(lam-independent)<1e-12
    return {'state':'passed','negative_and_above_one_powers_checked':errors,'trace_vs_rowwise_weight_error':abs(lam-independent)}

def summarize(rows,out):
    frame=pd.DataFrame(rows);frame.to_csv(out/'trial_records.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    keys=['study','prediction','label_rows','auxiliary_rows','epsilon','method']
    summary=frame.groupby(keys,dropna=False,as_index=False).agg(
        records=('replication','size'),available=('available','sum'),mean_MSE=('MSE','mean'),median_MSE=('MSE','median'),
        mean_excess_logloss=('excess_logloss','mean'),median_excess_logloss=('excess_logloss','median'),
        q90_excess_logloss=('excess_logloss',lambda s:s.quantile(.9)),q99_excess_logloss=('excess_logloss',lambda s:s.quantile(.99)),
        max_excess_logloss=('excess_logloss','max'),mean_weight=('weight','mean'),
        outside_unit_interval=('outside_unit_interval','sum'),negative_curvature=('negative_curvature','sum'))
    summary.to_csv(out/'summary.csv',index=False)
    frame.groupby(keys+['status'],dropna=False,as_index=False).size().to_csv(out/'status_counts.csv',index=False)

def main():
    a=argparse.ArgumentParser();a.add_argument('--input-dir',type=Path,required=True);a.add_argument('--canonical',type=Path,required=True)
    a.add_argument('--reference',type=Path,required=True);a.add_argument('--plan',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a.add_argument('--checks-only',action='store_true');a.add_argument('--resume',action='store_true');args=a.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    checks=analytic_checks();write_json(args.out/'analytic_checks.json',checks)
    if args.checks_only:print(json.dumps(checks),flush=True);return
    sys.path.insert(0,str(args.canonical/'analysis'))
    from public_conjoint_optimizer_audit import load_data
    from power_tuned_choice_audit import WeightedLogit,power_weight
    from finite_pool_choice_validation import interior_fit
    plan=json.loads(args.plan.read_text());assert plan['replications_per_original_study']==500
    source,_,_,_,_=load_data(args.input_dir);base=source[:1200]
    signatures={'script':digest(Path(__file__)),'plan':digest(args.plan),'support_array':hashlib.sha256(np.ascontiguousarray(base).tobytes()).hexdigest()}
    for name in ['power_tuned_choice_audit.py','finite_pool_choice_validation.py','public_conjoint_optimizer_audit.py']:
        signatures[name]=digest(args.canonical/'analysis'/name)
    rows=[];pilots=[];completed={s['name']:0 for s in plan['studies']};checkpoint=args.out/'checkpoint.json'
    if args.resume and checkpoint.exists():
        prev=json.loads(checkpoint.read_text());assert prev['signatures']==signatures
        completed=prev['completed'];rows=pd.read_csv(args.out/'trial_records.csv.gz').to_dict('records');pilots=pd.read_csv(args.out/'pilot_records.csv.gz').to_dict('records')
    elif checkpoint.exists():raise ValueError('Use a new output directory or --resume with unchanged sources')
    for study in plan['studies']:
        name=study['name'];truth=np.array(json.loads((args.reference/(name+'_DGP.json')).read_text())['truth']);prob_base=expit(base@truth)
        rng=np.random.default_rng(study['seed'])
        for rep in range(500):
            primary=base[rng.integers(0,len(base),150)];aux=base[rng.integers(0,len(base),600)]
            yp=(rng.random(150)<expit(primary@truth)).astype(float);yu=(rng.random(600)<expit(aux@truth)).astype(float)
            fp=rng.random(150);fu=rng.random(600)
            if rep<completed[name]:continue
            for m in study['labels']:
                human=WeightedLogit(primary[:m],np.full(m,1/m),primary[:m].T@yp[:m]/m)
                pilot,pc=interior_fit(human,np.zeros(base.shape[1]),rho=1/m);assert pilot is not None
                pilots.append(dict(study=name,replication=rep,label_rows=m,score_max=pc['score_max'],**{f'beta{j}':float(v) for j,v in enumerate(pilot)}))
                for N in study['auxiliary']:
                    x=np.vstack([primary[:m],aux[:N]]);y=np.r_[yp[:m],yu[:N]]
                    settings=[('oracle',e,1.,0.) for e in study['proxy_errors']] if name=='finite_pool' else [('oracle',.25,1.,0.),('overconfident',.25,2.,0.),('intercept_bias',.25,1.,1.)]
                    for prediction,e,slope,bias in settings:
                        z=np.r_[np.logical_xor(yp[:m],fp[:m]<e),np.logical_xor(yu[:N],fu[:N]<e)].astype(float)
                        g=expit(slope*(x@truth+(2*z-1)*np.log((1-e)/e))+bias)
                        hl,hu,true,pred=components(x,y,g,m,pilot)
                        old_lam,old_tune=power_weight(x,y,g,m,pilot);raw_pool=old_tune.get('raw_weight')
                        raw_label,label_status=scalar_weight(true,pred,hl,m,N)
                        cases=[]
                        for method,raw in [('pooled_clipped',raw_pool),('pooled_unclipped',raw_pool),('labelled_clipped',raw_label),('labelled_unclipped',raw_label)]:
                            lam=float(np.clip(raw,0,1)) if raw is not None and method.endswith('_clipped') else raw
                            beta,diagnostic=update(pilot,hl,hu,true,pred,m,lam)
                            diagnostic['tuning_status']=old_tune['tuning_status'] if method.startswith('pooled_') else label_status
                            cases.append((method,raw,lam,beta,diagnostic))
                        cases.append(('human_ridge',None,None,pilot,{'status':'finite_ridge_pilot'}))
                        for method,raw,lam,beta,diagnostic in cases:
                            row=dict(study=name,prediction=prediction,label_rows=m,auxiliary_rows=N,epsilon=e,replication=rep,method=method,
                                     raw_weight=raw,weight=lam,available=beta is not None,outside_unit_interval=bool(lam is not None and (lam<0 or lam>1)),
                                     negative_curvature=bool(diagnostic.get('curvature_min_eigenvalue',0)<0),**diagnostic)
                            if beta is not None:
                                error=beta-truth
                                row.update(MSE=float(np.mean(error**2)),excess_logloss=float(np.mean(np.logaddexp(0,base@beta)-np.logaddexp(0,base@truth)-prob_base*(base@error))),
                                           **{f'beta{j}':float(v) for j,v in enumerate(beta)})
                            rows.append(row)
            if (rep+1)%25==0 or rep==499:
                completed[name]=rep+1;summarize(rows,args.out)
                pd.DataFrame(pilots).to_csv(args.out/'pilot_records.csv.gz',index=False,compression={'method':'gzip','mtime':0})
                write_json(checkpoint,{'state':'running','signatures':signatures,'completed':completed,'records':len(rows),'new_independent_cohorts':0})
                print(json.dumps({'study':name,'completed':rep+1,'records':len(rows)}),flush=True)
    assert len(rows)==60000 and len(pilots)==3500
    write_json(checkpoint,{'state':'complete','signatures':signatures,'completed':completed,'records':len(rows),'pilot_records':len(pilots),'original_cohort_families':1500,'new_independent_cohorts':0,'new_human_or_AI_answers':0})
    print('Complete: all 24 original conditions, 500 repetitions, five comparators.',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
