"""Replay public choice allocations on an observable, anchor-score-preserving path.

The loss, pooled-information variance rule and support LP are established.
This diagnostic makes no oracle truth, inference or method-superiority claim.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.special import expit
from threadpoolctl import threadpool_limits
from public_conjoint_optimizer_audit import load_data, Logistic, converged, save
from public_conjoint_matched_audit import nuisance
from power_tuned_choice_audit import objective, support, power_weight


def path_prediction(mu, g, alpha, power):
    return g.copy() if power == 0 else mu + (alpha / power) * (g - mu)


def score_rows(x, y, g, m, power, anchor):
    mu=expit(x@anchor)
    labelled=x[:m]*((1-power)*mu[:m]-y[:m]+power*g[:m])[:,None]
    auxiliary=power*x[m:]*(mu[m:]-g[m:])[:,None]
    return labelled,auxiliary


def middle_matrix(labelled,auxiliary):
    return np.cov(labelled.T,ddof=1)/len(labelled)+np.cov(auxiliary.T,ddof=1)/len(auxiliary)


def residual_check(x,y,g,m,power):
    tau=(len(y)-m)/m
    lower=np.r_[-y[:m]+power*g[:m],-power/tau*g[m:]]
    upper=np.r_[1-power-y[:m]+power*g[:m],power/tau*(1-g[m:])]
    fit=linprog(np.zeros(len(y)),A_eq=x.T,b_eq=np.zeros(x.shape[1]),
                bounds=list(zip(lower,upper)),method='highs-ipm')
    if fit.status==2:return dict(status='infeasible')
    if not fit.success:return dict(status='unresolved',solver_status=int(fit.status))
    equality=float(np.max(np.abs(x.T@fit.x)))
    violation=float(max(0.,np.max(lower-fit.x),np.max(fit.x-upper)))
    return dict(status='feasible' if max(equality,violation)<1e-7 else 'unresolved',
                equality_residual=equality,bound_violation=violation)


def summaries(rows,out):
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'observable_score_path_records.csv',index=False)
    frame.groupby(['prediction','fraction','status'],dropna=False).size().reset_index(name='allocations').to_csv(out/'observable_score_path_status_counts.csv',index=False)
    summary=[]
    for (pred,fraction),part in frame.groupby(['prediction','fraction'],sort=False):
        summary.append(dict(prediction=pred,fraction=fraction,allocations=len(part),
            interior=int(part.status.eq('interior').sum()),outside=int(part.status.eq('outside_mean_support').sum()),
            boundary=int(part.status.eq('boundary_or_near_boundary').sum()),
            unresolved=int((~part.status.isin(['interior','outside_mean_support','boundary_or_near_boundary'])).sum()),
            alpha_zero=int(part.base_power.eq(0).sum()),alpha_one=int(part.base_power.eq(1).sum()),
            minimum_power=float(part.power.min()),maximum_power=float(part.power.max())))
    pd.DataFrame(summary).to_csv(out/'observable_score_path_summary.csv',index=False)
    pairs=[]
    for pred,part in frame.groupby('prediction'):
        pivot=part.assign(finite=part.status.eq('interior')).pivot(index='trial',columns='fraction',values='finite')
        for fraction in sorted(pivot.columns):
            a=pivot[0.0].to_numpy(int);b=pivot[fraction].to_numpy(int);d=b-a
            pairs.append(dict(prediction=pred,fraction=fraction,allocations=len(a),
                both_interior=int(((a==1)&(b==1)).sum()),base_only=int(((a==1)&(b==0)).sum()),
                path_only=int(((a==0)&(b==1)).sum()),neither=int(((a==0)&(b==0)).sum()),
                difference=float(d.mean()),paired_allocation_SE=float(d.std(ddof=1)/np.sqrt(len(d)))))
    pd.DataFrame(pairs).to_csv(out/'observable_score_path_pairs.csv',index=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();plan=json.loads(args.plan.read_text());out=args.out;out.mkdir(parents=True,exist_ok=True)
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();plan_sha=hashlib.sha256(args.plan.read_bytes()).hexdigest()
    ref=pd.read_csv(args.reference).query('label_rows == 50 and method == "power_human_init"').set_index(['trial','prediction'])
    assert len(ref)==2*plan['allocations']
    x,y,z,_,_=load_data(args.input_dir);x=x[:plan['prefix_rows']];y=y[:plan['prefix_rows']]
    m=plan['label_rows'];rng=np.random.RandomState(plan['seed']);rows=[];certificates=[];complete=set()
    status_file=out/'observable_score_path_status.json'
    if args.resume:
        old=json.loads(status_file.read_text());assert old['source_sha256']==source_sha and old['plan_sha256']==plan_sha
        rows=pd.read_csv(out/'observable_score_path_records.csv').to_dict('records')
        certificates=json.loads((out/'observable_score_path_certificates.json').read_text())
        complete={int(r['trial']) for r in rows}
        assert all(sum(int(r['trial'])==t for r in rows)==10 for t in complete)
    else:assert not status_file.exists(),'Use a new output directory or --resume'
    def checkpoint(state):
        if rows:summaries(rows,out)
        save(out/'observable_score_path_certificates.json',certificates)
        save(status_file,dict(state=state,completed_allocations=len(complete),records=len(rows),
            requested_allocations=plan['allocations'],requested_support_calls=plan['requested_positive_or_zero_power_support_calls'],
            source_sha256=source_sha,plan_sha256=plan_sha,
            reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            new_human_or_LLM_responses=0))
    checkpoint('running')
    for trial in range(plan['allocations']):
        order=(rng.permutation(240)[:,None]*5+np.arange(5)[None]).reshape(-1)
        if trial in complete:continue
        xx=x[order];yy=y[order];zz=z[order]
        anchor,anchor_check=converged(Logistic(xx[:m],yy[:m]));mu=expit(xx@anchor)
        for pred,use_ai in [('AI',True),('X_only',False)]:
            g,fallback=nuisance(xx,yy,zz,m,use_ai);assert fallback==0
            alpha,tuning=power_weight(xx,yy,g,m,anchor)
            assert alpha is not None and np.isfinite(alpha)
            reference=ref.loc[(trial,pred)]
            assert abs(alpha-reference.weight)<1e-9,(trial,pred,alpha,reference.weight)
            base_loss=objective(xx,yy,g,m,alpha);base_value,base_gradient=base_loss.value(anchor)
            base_labelled,base_aux=score_rows(xx,yy,g,m,alpha,anchor)
            base_middle=middle_matrix(base_labelled,base_aux)
            block=[]
            for fraction in plan['path_fractions']:
                power=float(alpha+fraction*(1-alpha));transformed=path_prediction(mu,g,alpha,power)
                assert np.min(transformed)>=0 and np.max(transformed)<=1
                loss=objective(xx,yy,transformed,m,power);value,gradient=loss.value(anchor)
                labelled,aux=score_rows(xx,yy,transformed,m,power,anchor)
                row_error=float(max(np.max(abs(labelled-base_labelled)),np.max(abs(aux-base_aux))))
                gradient_error=float(np.max(abs(gradient-base_gradient)))
                middle_error=float(np.max(abs(middle_matrix(labelled,aux)-base_middle)))
                assert max(row_error,gradient_error,middle_error)<1e-10
                weight,weight_meta=power_weight(xx,yy,transformed,m,anchor)
                weight_error=None if weight is None else abs(weight-power)
                if 0<alpha<1:
                    assert weight is not None and weight_error<1e-8,(trial,pred,alpha,power,weight)
                result=support(loss)
                if fraction==0:
                    assert result['status']=='interior' and reference.status=='finite_root'
                independently=None
                if trial%25==0 or result['status']!='interior':
                    independent=residual_check(xx,yy,transformed,m,power);independently=independent['status']
                    assert independently==('infeasible' if result['status']=='outside_mean_support' else 'feasible'),(trial,pred,fraction,result,independent)
                if result['status']=='outside_mean_support':
                    direction=np.asarray(result['direction']);cost=float(loss.weights@np.maximum(xx@direction,0)-loss.moment@direction)
                    assert cost< -1e-8 and abs(cost+result['separation_gap'])<1e-10
                    certificates.append(dict(trial=trial,prediction=pred,fraction=fraction,power=power,
                        base_power=alpha,direction=direction.tolist(),recession_cost=cost,
                        separation_gap=result['separation_gap'],anchor=anchor.tolist()))
                elif result['status']!='interior':
                    save(out/'unresolved_case.json',dict(trial=trial,prediction=pred,fraction=fraction,result=result))
                    checkpoint('stopped_on_unresolved');raise RuntimeError('Retained unresolved support case')
                block.append(dict(trial=trial,prediction=pred,fraction=fraction,base_power=alpha,power=power,
                    status=result['status'],interior_margin=result.get('interior_margin'),
                    separation_gap=result.get('separation_gap'),anchor_gradient_error=gradient_error,
                    full_score_row_error=row_error,full_score_middle_error=middle_error,
                    recomputed_power=weight,recomputed_power_error=weight_error,
                    recomputed_weight_status=weight_meta['tuning_status'],
                    independent_residual_status=independently,
                    anchor_score_max=anchor_check['mean_gradient_max']))
            rows.extend(block)
        complete.add(trial)
        if len(complete)%20==0:
            checkpoint('running');print(json.dumps(dict(allocations=len(complete),records=len(rows),exterior=len(certificates))),flush=True)
    assert len(rows)==plan['requested_positive_or_zero_power_support_calls']
    checkpoint('complete');print('Observable anchor path complete.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
