"""Rebuild the fixed diagnostic inputs with the preceding nuisance environment.

Raw public arrays remain local. Exact frozen hashes are checked before official
package execution in its separate environment.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
from scipy.optimize import linprog
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(a):
    root=Path(__file__).resolve().parents[1];a.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'frozen_input_manifest.json').read_text());plan=json.loads((root/'plan.json').read_text());assert sha(root/'plan.json')==manifest['plan_sha256']
    ref=root/'reference/analysis';assert sha(ref/'public_conjoint_optimizer_audit.py')==manifest['canonical_reader_sha256'] and sha(ref/'public_conjoint_matched_audit.py')==manifest['canonical_nuisance_sha256']
    assert sha(a.data_dir/'source_manifest.json')==manifest['source_manifest_sha256']
    sys.path.insert(0,str(ref));from public_conjoint_optimizer_audit import load_data
    from public_conjoint_matched_audit import nuisance
    x,y,z,_,_=load_data(a.data_dir);x=x[:1200];y=y[:1200]
    rng=np.random.RandomState(20261109);orders=[(rng.permutation(240)[:,None]*5+np.arange(5)[None]).reshape(-1) for _ in range(500)]
    cases=[]
    for c in plan['public_cases']:
        t,pr=c['trial'],c['prediction'];order=orders[t];xx,yy,zz=x[order],y[order],z[order];g,_=nuisance(xx,yy,zz,50,pr=='AI');b=xx[:50].T@(yy[:50]-g[:50])/50+xx[50:].T@g[50:]/1150
        f=linprog(np.zeros(1150),A_eq=xx[50:].T/1150,b_eq=b,bounds=(0,1),method='highs');assert (f.status==2)==(c['existing_unit_support']=='outside_mean_support')
        name=f'public_{t}_{pr}';np.savez(a.out/(name+'.npz'),X=xx[:50],Y=yy[:50],Yhat=g[:50],X_unlabeled=xx[50:],Yhat_unlabeled=g[50:]);cases.append(dict(name=name,input_file=name+'.npz',powers=plan['API_powers'],kind='public',trial=t,prediction=pr))
    for l in [.5,.55,.6,1.]:
        name='matched_reference_'+str(l);X=np.array([[1.,-1.],[1.,1.]]);Y=np.array([0.,1.]);U=np.column_stack([np.ones(8),[-2.,0.,0.,0.,0.,0.,0.,2.]])
        gl=.5+(.5/l)*(np.array([.1,.9])-.5);gu=.5+(.5/l)*(np.array([.25,.5,.5,.5,.5,.5,.5,.75])-.5)
        np.savez(a.out/(name+'.npz'),X=X,Y=Y,Yhat=gl,X_unlabeled=U,Yhat_unlabeled=gu);cases.append(dict(name=name,input_file=name+'.npz',powers=[l],kind='analytic',critical_power='11/20'))
    (a.out/'cases.json').write_text(json.dumps(cases,indent=2,ensure_ascii=False)+'\n');assert sha(a.out/'cases.json')==manifest['case_list_sha256']
    for c in manifest['cases']:assert sha(a.out/(c['name']+'.npz'))==c['sha256'],c['name']
    print(json.dumps(dict(state='passed',frozen_case_files_reproduced=len(cases),all_frozen_input_hashes_identical=True,new_respondents=0)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
