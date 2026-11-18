"""Post-hoc stronger upper-cap control: preserve zero and document reuse."""
from pathlib import Path
import argparse,gzip,hashlib,json
import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from existing_panel_model import load,corrected,fit_loss,metrics
from run_frontier import interior
BASE=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main(path,frontiers,out):
    assert not out.exists();out.mkdir(parents=True)
    plan=json.loads((BASE/'inputs/model_plan.json').read_text());data=load(path,plan)
    pre=json.loads((BASE/'inputs/preflight.json').read_text());g=softmax(data['teacher']@np.array(pre['teacher_coefficients']),axis=2)
    fr=pd.read_csv(frontiers/'frontiers.csv');old=pd.read_csv(BASE/'inputs/baseline_records.csv')
    source={r['record']:np.array(r['beta']) for r in json.loads((BASE/'inputs/cluster_coefficients.json').read_text())}
    pilots={r['record']:np.array(r['beta']) for r in json.loads((BASE/'inputs/labelled_ridge_coefficients.json').read_text())}
    rows=[];co=[];witness=[];rng=np.random.default_rng(plan['allocation_seed']);fitted=reused=0
    for trial in range(500):
        order=rng.permutation(data['pool'])
        for m in plan['labelled_respondent_budgets']:
            L,U=order[:m],order[m:];part=old[old.allocation.eq(trial)&old.labelled_people.eq(m)]
            previous=part[part.method.eq('corrected_cluster_power')].iloc[0]
            frontier=fr[fr.allocation.eq(trial)&fr.labelled_people.eq(m)].iloc[0]
            cap=frontier.frontier;lam=float(min(previous.power,.9*cap)) if cap>1e-10 else float(previous.power)
            row=dict(record=len(rows),allocation=trial,labelled_people=m,original_record=int(previous.record),
                     original_power=float(previous.power),power=lam,frontier_record=int(frontier.frontier_record),
                     power_changed=bool(lam!=previous.power),cap_near_zero=bool(cap<=1e-10))
            loss=corrected(data,L,U,g,lam);state=interior(loss);row['support']=state['support']
            if state['support']=='interior':
                if lam==previous.power:
                    beta=source[int(previous.record)];reused+=1
                    check=dict(fit='passed_score',gradient_max=float(np.max(abs(loss.value(beta)[1]))),coefficient_source='verified_original_objective')
                else:
                    pilot=pilots[int(part[part.method.eq('labelled_ridge')].iloc[0].record)]
                    beta,check=fit_loss(loss,pilot);fitted+=1;check['coefficient_source']='new_cap_fit'
                row.update(check)
                if check['fit']=='passed_score':
                    row.update(metrics(data['f'][data['test']],data['y'][data['test']],beta));co.append(dict(record=len(rows),beta=beta.tolist()))
            else:row['fit']='unavailable_by_support'
            witness.append(dict(record=len(rows),**state));rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(out/'records.csv',index=False)
    (out/'coefficients.json').write_text(json.dumps(co,separators=(',',':'))+'\n')
    (out/'interior_witnesses.json.gz').write_bytes(gzip.compress(json.dumps(witness,separators=(',',':')).encode(),mtime=0))
    summaries=[]
    for m,part in frame.groupby('labelled_people'):
        pivot=old[old.labelled_people.eq(m)].pivot(index='allocation',columns='method',values='heldout_logloss')
        vals=part.set_index('allocation').heldout_logloss.reindex(pivot.index).fillna(pivot.labelled_ridge)
        comparisons={}
        for b in pivot.columns:
            d=vals-pivot[b].fillna(pivot.labelled_ridge)
            comparisons[b]=dict(mean=float(d.mean()),allocation_MCSE=float(d.std(ddof=1)/np.sqrt(500)))
        summaries.append(dict(labelled_people=int(m),available=int(part.fit.eq('passed_score').sum()),denominator=500,
                              support_counts=part.support.value_counts().to_dict(),power_changes=int(part.power_changed.sum()),
                              mean_fallback_loss=float(vals.mean()),comparisons=comparisons))
    (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    status=dict(state='complete',records=len(rows),new_model_fits=fitted,reused_verified_coefficients=reused,
                available_coefficients=len(co),amendment_sha256=sha(BASE/'cap_only_amendment.json'),
                analysis_sha256=sha(Path(__file__)),primary_frontiers_sha256=sha(frontiers/'frontiers.csv'),source_sha256=sha(path))
    (out/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--frontiers',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):main(a.input,a.frontiers,a.out)
