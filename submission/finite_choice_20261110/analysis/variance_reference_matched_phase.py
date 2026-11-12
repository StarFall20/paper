"""Matched finite-solution experiment along an identical truth-score path."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.special import expit
from threadpoolctl import threadpool_limits
from power_tuned_choice_audit import objective,support


def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def weak_separator(x,y):
    d=x.shape[1];A=(2*y-1)[:,None]*x
    result=linprog(np.r_[-A.mean(0),A.mean(0)],A_ub=np.r_[np.c_[-A,A],np.ones((1,2*d))],
        b_ub=np.r_[np.zeros(len(y)),1.],bounds=(0.,None),method='highs')
    if not result.success:raise RuntimeError('Ordinary separator LP unresolved')
    direction=result.x[:d]-result.x[d:];margins=A@direction
    if margins.min()< -1e-8 or margins.mean()<=1e-8:raise RuntimeError('Ordinary boundary did not have a verified weak separator')
    return dict(direction=direction.tolist(),minimum_margin=float(margins.min()),mean_margin=float(margins.mean()))


def residual_feasibility(x,y,z,g,m,lam):
    N=len(y)-m;tau=N/m;sign=np.r_[1-2*y[:m],2*z[m:]-1]
    r=np.abs(y[:m]-g[:m]);lower=np.r_[lam*r,lam/tau*np.where(z[m:]==1,-g[m:],g[m:]-1)]
    upper=np.r_[1-lam+lam*r,lam/tau*np.where(z[m:]==1,1-g[m:],g[m:])]
    matrix=(x*sign[:,None]).T
    fit=linprog(np.zeros(len(y)),A_eq=matrix,b_eq=np.zeros(x.shape[1]),bounds=list(zip(lower,upper)),method='highs')
    if fit.status==2:return 'infeasible'
    if not fit.success or np.max(abs(matrix@fit.x))>1e-7:return 'unresolved'
    return 'feasible'


def summarize(rows,out):
    frame=pd.DataFrame(rows);frame.to_csv(out/'variance_reference_phase_trials.csv',index=False)
    summaries=[];pairs=[]
    for key,part in frame.groupby(['label_rows','orthogonal_slopes','dimension_ratio','method','strength','power'],sort=False):
        states=part.status.value_counts();count=len(part);f=float(part.status.eq('interior').mean())
        summaries.append(dict(zip(['label_rows','orthogonal_slopes','dimension_ratio','method','strength','power'],key))|
            dict(replications=count,finite=int(states.get('interior',0)),outside=int(states.get('outside_mean_support',0)),
            boundary=int(states.get('boundary_or_near_boundary',0)),unresolved=int(count-part.status.isin(['interior','outside_mean_support','boundary_or_near_boundary']).sum()),
            finite_frequency=f,binomial_MCSE=float(np.sqrt(f*(1-f)/count))))
    for key,part in frame.groupby(['label_rows','orthogonal_slopes','dimension_ratio'],sort=False):
        available=part.assign(finite=part.status.eq('interior')).pivot(index='trial',columns='method',values='finite')
        for comparator in ['ordinary','shrunk_variance_reference','shrunk_fixed_half']:
            reference=available.oracle_variance_reference.to_numpy(int);other=available[comparator].to_numpy(int);difference=reference-other
            pairs.append(dict(label_rows=key[0],orthogonal_slopes=key[1],dimension_ratio=key[2],reference='oracle_variance_reference',comparator=comparator,
                replications=len(reference),both_finite=int(np.sum((reference==1)&(other==1))),reference_only=int(np.sum((reference==1)&(other==0))),
                comparator_only=int(np.sum((reference==0)&(other==1))),neither_finite=int(np.sum((reference==0)&(other==0))),
                finite_frequency_difference=float(difference.mean()),paired_MCSE=float(difference.std(ddof=1)/np.sqrt(len(difference)))))
    pd.DataFrame(summaries).to_csv(out/'variance_reference_phase_summary.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'variance_reference_phase_pairs.csv',index=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    plan=json.loads(a.plan.read_text());a.out.mkdir(parents=True,exist_ok=True);source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();plan_sha=hashlib.sha256(a.plan.read_bytes()).hexdigest()
    rows=[];separators=[];examples=[];completed=set();maximum_identity=0.;crosschecks=0;exterior_checked=0
    status_file=a.out/'variance_reference_phase_status.json'
    if a.resume:
        state=json.loads(status_file.read_text());assert state['source_sha256']==source and state['plan_sha256']==plan_sha
        rows=pd.read_csv(a.out/'variance_reference_phase_trials.csv').to_dict('records');separators=json.loads((a.out/'ordinary_separation_directions.json').read_text());examples=json.loads((a.out/'discordant_examples.json').read_text())
        completed={(int(r['label_rows']),int(r['trial'])) for r in rows};assert all(sum(r['label_rows']==m and r['trial']==trial for r in rows)==16 for m,trial in completed)
        maximum_identity=state['maximum_truth_score_difference'];crosschecks=state['independent_residual_checks'];exterior_checked=state['exterior_directions_checked']
    else:assert not status_file.exists(),'Use a new output directory or --resume'
    def checkpoint(state='running'):
        if rows:summarize(rows,a.out)
        save(a.out/'ordinary_separation_directions.json',separators);save(a.out/'discordant_examples.json',examples)
        save(status_file,dict(state=state,model_draws=len(completed),support_classifications=len(rows),requested_model_draws=plan['requested_model_draws'],requested_support_classifications=plan['requested_support_classifications'],
            maximum_truth_score_difference=maximum_identity,independent_residual_checks=crosschecks,exterior_directions_checked=exterior_checked,
            ordinary_directions=len(separators),source_sha256=source,plan_sha256=plan_sha,new_human_or_LLM_responses=0))
    checkpoint();rng=np.random.default_rng(plan['seed'])
    for m in plan['label_rows']:
      N=round(m*plan['tau']);n=m+N;dimensions=sorted({round(m*k) for k in plan['dimension_ratios']})
      for trial in range(plan['replications']):
        s=rng.normal(size=n);orth=rng.normal(size=(n,max(dimensions)));mu=expit(plan['gamma']*s)
        y=(rng.random(n)<mu).astype(float);z=np.logical_xor(y,rng.random(n)<plan['epsilon']).astype(float)
        oracle=expit(plan['gamma']*s+(2*z-1)*np.log((1-plan['epsilon'])/plan['epsilon']))
        if (m,trial) in completed:continue
        block=[];new_separators=[];new_examples=[];block_cross=0;block_exterior=0;block_error=0.
        for q in dimensions:
            x=np.column_stack([np.ones(n),s,orth[:,:q]]);beta0=np.r_[0.,plan['gamma'],np.zeros(q)]
            results={};losses={};predictions={}
            for method in plan['methods']:
                name=method['name'];g=mu+method['strength']*(oracle-mu);loss=objective(x,y,g,m,method['power']);result=support(loss)
                results[name]=result;losses[name]=loss;predictions[name]=g
                if result['status']=='outside_mean_support':
                    d=np.array(result['direction']);cost=float(loss.weights@np.maximum(x@d,0)-loss.moment@d)
                    assert cost< -1e-8 and abs(cost+result['separation_gap'])<1e-10;block_exterior+=1
                elif name=='ordinary' and result['status']=='boundary_or_near_boundary':
                    cert=weak_separator(x[:m],y[:m]);new_separators.append(dict(label_rows=m,trial=trial,orthogonal_slopes=q,**cert))
                elif result['status']!='interior':
                    save(a.out/'unresolved_case.json',dict(label_rows=m,trial=trial,orthogonal_slopes=q,method=name,result=result,x=x.tolist(),y=y.tolist(),z=z.tolist(),g=g.tolist()))
                    checkpoint('stopped_on_unresolved');raise RuntimeError('Retained unresolved case; no draw dropped')
                independent=None
                if method['power']>0 and trial%plan['independent_residual_check_every_trials']==0:
                    independent=residual_feasibility(x,y,z,g,m,method['power']);expected='infeasible' if result['status']=='outside_mean_support' else 'feasible'
                    if independent!=expected:
                        save(a.out/'unresolved_case.json',dict(label_rows=m,trial=trial,orthogonal_slopes=q,method=name,result=result,residual_status=independent,x=x.tolist(),y=y.tolist(),z=z.tolist(),g=g.tolist()))
                        checkpoint('stopped_on_crosscheck');raise RuntimeError('Retained original/residual discrepancy')
                    block_cross+=1
                block.append(dict(label_rows=m,auxiliary_rows=N,trial=trial,orthogonal_slopes=q,dimension_ratio=q/m,method=name,strength=method['strength'],power=method['power'],status=result['status'],
                    interior_margin=result.get('interior_margin'),separation_gap=result.get('separation_gap'),independent_residual_status=independent))
            error=float(np.max(abs(losses['oracle_variance_reference'].value(beta0)[1]-losses['shrunk_variance_reference'].value(beta0)[1])));block_error=max(block_error,error);assert error<1e-10
            if results['oracle_variance_reference']['status']=='interior' and results['shrunk_variance_reference']['status']=='outside_mean_support' and len(examples)+len(new_examples)<24:
                new_examples.append(dict(label_rows=m,trial=trial,orthogonal_slopes=q,ordinary_status=results['ordinary']['status'],
                    truth_score_error=error,oracle_reference_interior_margin=results['oracle_variance_reference']['interior_margin'],
                    shrunk_reference_recession_gap=results['shrunk_variance_reference']['separation_gap'],shrunk_reference_direction=results['shrunk_variance_reference']['direction']))
        rows.extend(block);separators.extend(new_separators);examples.extend(new_examples);maximum_identity=max(maximum_identity,block_error);crosschecks+=block_cross;exterior_checked+=block_exterior;completed.add((m,trial))
        if len(completed)%25==0:
            checkpoint();print(json.dumps(dict(model_draws=len(completed),support_classifications=len(rows),ordinary_directions=len(separators))),flush=True)
    assert len(rows)==plan['requested_support_classifications'] and len(completed)==plan['requested_model_draws'];checkpoint('complete');print('Matched truth-score phase complete.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
