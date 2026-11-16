"""Independent record, residual-interior, direction and exact-example checks."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from scipy.special import expit
from threadpoolctl import threadpool_limits
from public_conjoint_optimizer_audit import load_data,Logistic,converged,save
from public_conjoint_matched_audit import nuisance
from power_tuned_choice_audit import objective,power_weight,support
from observable_low_label_path import ridge_anchor


def independent_interior(x,y,g,m,power):
    n,p=x.shape;tau=(n-m)/m
    lower=np.r_[-y[:m]+power*g[:m],-power/tau*g[m:]]
    widths=np.r_[np.full(m,1-power),np.full(n-m,power/tau)]
    keep=widths>0;matrix=x[keep].T;lo=lower[keep];width=widths[keep]
    if np.linalg.matrix_rank(matrix)<p:return dict(status='rank_deficient')
    count=len(width);eye=sparse.eye(count,format='csr');col=sparse.csr_matrix(width[:,None])
    result=linprog(np.r_[np.zeros(count),-1.],A_ub=sparse.vstack([sparse.hstack([-eye,col]),sparse.hstack([eye,col])]),
        b_ub=np.r_[np.zeros(count),width],A_eq=sparse.hstack([sparse.csr_matrix(matrix),sparse.csr_matrix((p,1))]),
        b_eq=-x.T@lower,bounds=[(0.,None)]*count+[(0.,.5)],method='highs-ipm')
    if result.status==2:return dict(status='outside_mean_support')
    if not result.success:raise RuntimeError('Independent interior LP unresolved')
    u=result.x[:-1];slack=float(result.x[-1]);equality=float(np.max(abs(matrix@u+x.T@lower)))
    inequality=float(max(0.,np.max(slack*width-u),np.max(u+slack*width-width)))
    assert max(equality,inequality)<1e-7,(equality,inequality)
    return dict(status='interior' if slack>1e-8 else 'boundary_or_near_boundary',slack=slack,equality_residual=equality,inequality_violation=inequality)


def stratum(a):return 'zero' if a==0 else ('one' if a==1 else 'interior')


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--high',type=Path,required=True);p.add_argument('--low',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    frames={50:pd.read_csv(a.high/'observable_score_path_records.csv'),20:pd.read_csv(a.low/'observable_score_path_records.csv')}
    statuses={50:json.loads((a.high/'observable_score_path_status.json').read_text()),20:json.loads((a.low/'observable_low_label_status.json').read_text())}
    summary_rows=[];pairs=[];selected={50:set(range(0,500,50)),20:set(range(0,500,50))}
    for m,frame in frames.items():
        assert statuses[m]['state']=='complete' and len(frame)==5000
        assert frame.groupby(['trial','prediction']).size().eq(5).all() and frame.trial.nunique()==500
        assert frame.status.isin(['interior','outside_mean_support','boundary_or_near_boundary']).all()
        assert frame[['anchor_gradient_error','full_score_row_error','full_score_middle_error']].max().max()<1e-10
        frame['stratum']=frame.base_power.map(stratum)
        for (pred,group,fraction),part in frame.groupby(['prediction','stratum','fraction']):
            summary_rows.append(dict(label_rows=m,prediction=pred,base_power_stratum=group,fraction=fraction,allocations=len(part),
                finite=int(part.status.eq('interior').sum()),outside=int(part.status.eq('outside_mean_support').sum()),boundary=int(part.status.eq('boundary_or_near_boundary').sum())))
        for (pred,group),part in frame.groupby(['prediction','stratum']):
            pivot=part.assign(finite=part.status.eq('interior')).pivot(index='trial',columns='fraction',values='finite')
            base=pivot[0.].to_numpy(int);end=pivot[1.].to_numpy(int)
            pairs.append(dict(label_rows=m,prediction=pred,base_power_stratum=group,allocations=len(base),both_finite=int(((base==1)&(end==1)).sum()),base_only=int(((base==1)&(end==0)).sum()),endpoint_only=int(((base==0)&(end==1)).sum()),neither_finite=int(((base==0)&(end==0)).sum())))
            if group=='interior':
                assert np.max(abs(part.recomputed_power-part.power))<1e-8
                selected[m].update(int(t) for t in pivot.index[pivot[0.]!=pivot[1.]])
        retained=pd.read_csv((a.high if m==50 else a.low)/'observable_score_path_summary.csv')
        for row in retained.itertuples():
            part=frame[(frame.prediction==row.prediction)&(frame.fraction==row.fraction)]
            assert len(part)==row.allocations and part.status.eq('interior').sum()==row.interior and part.status.eq('outside_mean_support').sum()==row.outside
    low_cert=json.loads((a.low/'path_certificates.json').read_text());high_cert=json.loads((a.high/'observable_score_path_certificates.json').read_text());assert not high_cert
    selected[20].update(int(c['trial']) for c in low_cert)
    x,y,z,_,_=load_data(a.input_dir);x=x[:1200];y=y[:1200];rng=np.random.RandomState(20261109)
    ordinary=json.loads((a.low/'ordinary_support_checks.json').read_text());assert len(ordinary)==500
    ordinary_map={r['trial']:r for r in ordinary};rays=0;separators=0;replayed=0;independent=0;max_row=max_middle=max_grad=max_ridge=max_weight=0.;primal=[]
    for trial in range(500):
        order=(rng.permutation(240)[:,None]*5+np.arange(5)[None]).reshape(-1);xx=x[order];yy=y[order];zz=z[order]
        row=ordinary_map[trial]
        if row['status']=='boundary_or_near_boundary':
            margins=(2*yy[:20]-1)*(xx[:20]@np.asarray(row['direction']))
            assert margins.min()>-1e-8 and margins.mean()>1e-8;separators+=1
        for m in [50,20]:
            if trial not in selected[m]:continue
            if m==50:anchor,_=converged(Logistic(xx[:m],yy[:m]))
            else:anchor,err=ridge_anchor(xx[:m],yy[:m]);max_ridge=max(max_ridge,err)
            mu=expit(xx@anchor)
            for pred,use_ai in [('AI',True),('X_only',False)]:
                g,fall=nuisance(xx,yy,zz,m,use_ai);alpha,_=power_weight(xx,yy,g,m,anchor)
                part=frames[m].query('trial==@trial and prediction==@pred').sort_values('fraction')
                assert len(part)==5 and np.max(abs(part.base_power-alpha))<1e-9
                # Direct separate-sample derivatives, without the generator's
                # score_rows or middle_matrix helper.
                baseline_l=xx[:m]*((1-alpha)*mu[:m]-yy[:m]+alpha*g[:m])[:,None]
                baseline_u=alpha*xx[m:]*(mu[m:]-g[m:])[:,None]
                middle0=np.cov(baseline_l.T,ddof=1)/m+np.cov(baseline_u.T,ddof=1)/(1200-m)
                for record in part.itertuples():
                    power=record.power;gs=g.copy() if power==0 else mu+alpha/power*(g-mu)
                    sl=xx[:m]*((1-power)*mu[:m]-yy[:m]+power*gs[:m])[:,None]
                    su=power*xx[m:]*(mu[m:]-gs[m:])[:,None]
                    er=float(max(np.max(abs(sl-baseline_l)),np.max(abs(su-baseline_u))))
                    mid=np.cov(sl.T,ddof=1)/m+np.cov(su.T,ddof=1)/(1200-m)
                    max_row=max(max_row,er);max_middle=max(max_middle,float(np.max(abs(mid-middle0))))
                    loss=objective(xx,yy,gs,m,power)
                    max_grad=max(max_grad,float(np.max(abs(loss.value(anchor)[1]-baseline_l.mean(0)-baseline_u.mean(0)))))
                    original=support(loss);assert original['status']==record.status,(trial,m,pred,power,record.status,original)
                    independent_result=independent_interior(xx,yy,gs,m,power)
                    assert independent_result['status']==record.status,(trial,m,pred,power,record.status,independent_result)
                    if 'slack' in independent_result:primal.append(independent_result)
                    replayed+=1;independent+=1
                    if 0<alpha<1:
                        w,_=power_weight(xx,yy,gs,m,anchor);max_weight=max(max_weight,abs(w-power))
                    if m==20:
                        for cert in low_cert:
                            if cert['trial']==trial and cert['prediction']==pred and cert['fraction']==record.fraction:
                                d=np.asarray(cert['direction']);cost=float(loss.weights@np.maximum(xx@d,0)-loss.moment@d)
                                assert cost<-1e-8 and abs(cost-cert['recession_cost'])<1e-10;rays+=1
    assert rays==len(low_cert) and max(max_row,max_middle,max_grad,max_ridge,max_weight)<1e-8
    # Numerical crosscheck of the rational example using the existing support
    # code. The exact rational proof is a separate artifact.
    ex_x=np.array([[2.],[4.],[1.],[1.]]);ex_y=np.array([1.,1.,0.,0.]);ex_g=np.array([.9,.9,.5,.5]);anchor=np.array([0.])
    # The retained 11-coordinate tuning helper uses matrix covariance products;
    # np.cov returns a scalar in this new one-coordinate illustration. Evaluate
    # the same published scalar convention directly, without editing that helper.
    true=ex_x[:2,0]*(.5-ex_y[:2]);pred=ex_x[:,0]*(.5-ex_g)
    alpha=float(np.mean((true-true.mean())*(pred[:2]-pred[:2].mean()))/(2*np.var(pred,ddof=1)))
    assert abs(alpha-15/88)<1e-14
    for power in [alpha,.2,9/44,.3,1.]:
        gs=.5+alpha/power*(ex_g-.5);status=support(objective(ex_x,ex_y,gs,2,power))['status']
        expected='interior' if power<9/44-1e-12 else ('outside_mean_support' if power>9/44+1e-12 else 'boundary_or_near_boundary')
        assert status==expected,(power,status,expected)
    pd.DataFrame(summary_rows).to_csv(a.out/'power_stratified_counts.csv',index=False);pd.DataFrame(pairs).to_csv(a.out/'power_stratified_pairs.csv',index=False)
    result=dict(date='2026-11-12',retained_path_records=10000,completed_allocations_per_budget=500,
        replayed_support_classifications=replayed,independent_residual_interior_LPs=independent,
        all_low_label_exterior_directions_verified=rays,all_ordinary_weak_separators_verified=separators,
        maximum_score_row_error=max_row,maximum_middle_matrix_error=max_middle,maximum_gradient_error=max_grad,
        maximum_existing_weight_recomputation_error=max_weight,maximum_ridge_anchor_score_error=max_ridge,
        maximum_independent_primal_residual=max(r.get('equality_residual',0) for r in primal),
        exact_scalar_example_numerically_crosschecked=True,all_unresolved=0,
        scalar_example_variance_rule_evaluated_directly=True,
        common_covariance_at_unknown_truth_claimed=False,new_method_superiority_claimed=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_state_hashes={str(root/name):hashlib.sha256((root/name).read_bytes()).hexdigest() for root,name in [(a.high,'observable_score_path_status.json'),(a.low,'observable_low_label_status.json')]})
    save(a.out/'observable_path_complete_verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
