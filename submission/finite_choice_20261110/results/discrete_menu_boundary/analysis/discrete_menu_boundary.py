"""Coupled independent discrete-profile variants of the existing menu cohorts."""
from pathlib import Path
import argparse,ast,gzip,hashlib,json,time
import numpy as np
import pandas as pd
from scipy.special import ndtri,softmax
from threadpoolctl import threadpool_limits
from menu_corrected_logit import MenuLoss,menu_features,support
from hard_menu_matched_phase import weak_separator

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def transform(w,law):
    if law=='gaussian':return w
    if law=='symmetric_binary':return np.where(w<0.,-1.,1.)
    if law=='symmetric_ternary':
        c=ndtri(2/3);return np.where(w < -c,-np.sqrt(1.5),np.where(w > c,np.sqrt(1.5),0.))
    raise ValueError(law)

def cohorts(plan):
    rng=np.random.default_rng(plan['seed']);J=plan['alternatives']
    for m in plan['label_menus']:
        n=m+int(m*plan['tau']);dims=sorted({max(1,round(m*k)) for k in plan['dimension_ratios']})
        for trial in range(plan['replications_per_budget']):
            s=rng.normal(size=(n,J,1));w=rng.normal(size=(n,J,max(dims)))
            mu=softmax(plan['gamma']*s[...,0],axis=1)
            y=np.sum(rng.random(n)[:,None]>np.cumsum(mu,axis=1),axis=1)
            z=(y+(rng.random(n)<plan['epsilon'])*rng.integers(1,J,size=n))%J
            yield m,trial,dims,s,w,y,np.eye(J)[z]

def closure():
    base=Path(__file__).resolve().parent;found={}
    def visit(f):
        if f.name in found:return
        found[f.name]=sha(f)
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node,ast.ImportFrom) and node.module:
                p=base/(node.module.split('.')[0]+'.py')
                if p.is_file():visit(p)
    visit(Path(__file__).resolve());return found

def aggregate(frame,out):
    keys=['law','label_menus','orthogonal_slopes','method','power','critical_ratio']
    summary=frame.groupby(keys,as_index=False).agg(replications=('status','size'),finite=('status',lambda x:int(x.eq('interior').sum())),exterior=('status',lambda x:int(x.eq('outside').sum())),boundary=('status',lambda x:int(x.eq('boundary_or_near_boundary').sum())),rank_deficient=('status',lambda x:int(x.eq('rank_deficient').sum())),unresolved=('status',lambda x:int(x.str.startswith('unresolved').sum())))
    summary['finite_frequency']=summary.finite/summary.replications
    summary['binomial_MCSE']=np.sqrt(summary.finite_frequency*(1-summary.finite_frequency)/summary.replications)
    summary.to_csv(out/'summary.csv',index=False)
    paired=[]
    for key,p in frame.groupby(['label_menus','orthogonal_slopes','method']):
        x=p.assign(finite=p.status.eq('interior').astype(int)).pivot(index='trial',columns='law',values='finite')
        for law in ['symmetric_binary','symmetric_ternary']:
            diff=(x[law]-x.gaussian).to_numpy()
            paired.append(dict(label_menus=key[0],orthogonal_slopes=key[1],method=key[2],law=law,paired_cohorts=len(diff),both_finite=int(((x[law]==1)&(x.gaussian==1)).sum()),discrete_only=int(((x[law]==1)&(x.gaussian==0)).sum()),gaussian_only=int(((x[law]==0)&(x.gaussian==1)).sum()),neither_finite=int(((x[law]==0)&(x.gaussian==0)).sum()),finite_frequency_difference=float(diff.mean()),paired_MCSE=float(diff.std(ddof=1)/np.sqrt(len(diff)))))
    pd.DataFrame(paired).to_csv(out/'paired_with_gaussian.csv',index=False)
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--out',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    plan=json.loads(a.plan.read_text());old=json.loads((a.repo/plan['original_plan']).read_text());a.out.mkdir(parents=True,exist_ok=True)
    old_trials=pd.read_csv(a.repo/plan['original_trials']);lookup={(int(r.label_menus),int(r.trial),int(r.orthogonal_slopes),r.method):r.status for r in old_trials.itertuples()}
    src=closure();ident=dict(source_hashes=src,plan_sha256=sha(a.plan),original_plan_sha256=sha(a.repo/plan['original_plan']),original_trials_sha256=sha(a.repo/plan['original_trials']))
    records=[];certs=[];done=0;start=time.monotonic()
    if a.resume:
        status=json.loads((a.out/'status.json').read_text());assert status['state']=='running' and status['identity']==ident
        with gzip.open(a.out/'checkpoint.json.gz','rt') as f:checkpoint=json.load(f)
        records=checkpoint['records'];certs=checkpoint['certificates'];done=checkpoint['completed_cohorts'];assert len(records)==done*27
    elif (a.out/'status.json').exists():raise RuntimeError('Use a fresh output directory or explicit matching resume')
    def checkpoint(completed):
        data=json.dumps(dict(records=records,certificates=certs,completed_cohorts=completed),allow_nan=False).encode()
        (a.out/'checkpoint.json.gz').write_bytes(gzip.compress(data,mtime=0))
        save(a.out/'status.json',dict(state='running',identity=ident,completed_original_cohorts=completed,planned_original_cohorts=1000,completed_classifications=len(records),planned_classifications=27000))
        print(json.dumps(dict(completed_cohorts=completed,classifications=len(records))),flush=True)
    for index,(m,trial,dims,s,w,y,g) in enumerate(cohorts(old),1):
        if index<=done:continue
        for law in plan['profile_laws']:
            new_w=transform(w,law)
            for q in dims:
                f=menu_features(np.concatenate([s,new_w[...,:q]],axis=2))
                for method,power in old['powers'].items():
                    loss=MenuLoss(f,y,g,m,power);state=support(loss);key=dict(law=law,label_menus=m,trial=trial,orthogonal_slopes=q,method=method)
                    if law=='gaussian':assert state['status']==lookup[(m,trial,q,method)],(key,state,lookup[(m,trial,q,method)])
                    if state['status']=='outside':certs.append(dict(**key,kind='exterior',direction=state['direction'],gap=state['separation_gap']))
                    if method=='ordinary' and state['status']=='boundary_or_near_boundary':
                        cert=weak_separator(f[:m],y[:m]);certs.append(dict(**key,kind='human_separation',**cert))
                        if cert['state']!='verified':state['status']='unresolved_boundary'
                    records.append(dict(**key,auxiliary_menus=len(y)-m,dimension_ratio=q/m,power=power,critical_ratio=old['critical_ratios'][method],status=state['status'],interior_margin=state.get('margin'),exterior_gap=state.get('separation_gap'),equality_residual=state.get('equality_residual')))
        if index%20==0 or index==1000:checkpoint(index)
    frame=pd.DataFrame(records);assert len(frame)==27000
    frame.to_csv(a.out/'trials.csv',index=False);summary=aggregate(frame,a.out)
    assert len(summary)==54 and summary.replications.eq(500).all()
    save(a.out/'status.json',dict(state='complete',identity=ident,existing_independent_cohorts=1000,coupled_profile_variants=3000,new_independent_cohorts=0,classifications=len(frame),cells=len(summary),gaussian_original_states_matched=9000,finite=int(summary.finite.sum()),boundary=int(summary.boundary.sum()),positive_power_boundary=int(summary.loc[summary.power>0,'boundary'].sum()),rank_deficient=int(summary.rank_deficient.sum()),unresolved=int(summary.unresolved.sum()),certificate_records=len(certs),new_human_or_AI_answers=0,elapsed_seconds=time.monotonic()-start,scope='Finite-budget availability under independent symmetric discrete noise profiles; no new estimator, risk ranking, universality proof or originality certification'))
    print((a.out/'status.json').read_text(),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
