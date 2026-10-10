"""External implementation check on the public mlogit Electricity DCE."""
from __future__ import annotations
import argparse,csv,hashlib,os
from pathlib import Path
import numpy as np
from scipy.stats import chi2
from observational_equivalence_test import fit_mnl,predict

ATTRS=('pf','cl','loc','wk','tod','seas')

def read_electricity(path):
    with open(path, newline="") as handle:
        d = list(csv.DictReader(handle))
    if not d:
        raise ValueError('electricity CSV is empty')
    columns = set(d[0])
    required={'choice','id'}|{f'{a}{j}' for a in ATTRS for j in range(1,5)}
    missing=required-columns
    if missing: raise ValueError(f'missing columns: {sorted(missing)}')
    ids=np.asarray([int(float(row['id'])) for row in d], dtype=int)
    y=np.asarray([int(float(row['choice'])) for row in d], dtype=int)-1
    if np.any((y<0)|(y>3)): raise ValueError('choice must be 1..4')
    attrs=np.stack([np.asarray([[float(row[f'{a}{j}']) for j in range(1,5)] for row in d], dtype=float)
                    for a in ATTRS], axis=2)
    avail=np.ones((len(d),4),float)
    return d,ids,y,attrs,avail

def design(attrs,name):
    pf=-attrs[:,:,0]/10.0; cl=-attrs[:,:,1]/5.0
    loc,wk,tod,seas=[attrs[:,:,j] for j in range(2,6)]
    base=[pf,cl,loc,wk,tod,seas]
    if name=='linear': cols=base
    elif name=='quadratic': cols=base+[pf*pf,cl*cl,pf*cl,pf*tod,pf*seas,cl*seas]
    elif name=='interactions': cols=base+[loc*wk,tod*seas,loc*seas,wk*tod]
    elif name=='full': cols=base+[pf*pf,cl*cl,pf*cl,pf*tod,pf*seas,cl*seas,loc*wk,tod*seas,loc*seas,wk*tod]
    else: raise ValueError(name)
    return np.stack(cols,axis=2)

def effective_rank(X):
    """Rank of alternative utility differences after removing a reference arm."""
    D=(X[:,:-1,:]-X[:,-1:,:]).reshape(-1,X.shape[2])
    return int(np.linalg.matrix_rank(D,tol=1e-9))

def scores(X,y,avail,beta):
    p=predict(X,avail,beta); return float(np.mean(np.log(p[np.arange(len(y)),y]+1e-300))),float(np.mean(np.argmax(p,axis=1)==y))

def run(path,out,provenance_out,seed=20261008):
    d,ids,y,attrs,avail=read_electricity(path); people=np.unique(ids); rng=np.random.default_rng(seed); sh=people.copy();rng.shuffle(sh);folds=np.array_split(sh,5)
    names=('linear','quadratic','interactions','full'); records=[]; fits={}
    for name in names:
        X=design(attrs,name); beta=fit_mnl(X,y,avail); rank=effective_rank(X); p=predict(X,avail,beta); ll=float(np.log(p[np.arange(len(y)),y]+1e-300).sum()); fits[name]=(X,beta,ll)
        oof=[]; acc=[]
        for test_ids in folds:
            test=np.isin(ids,test_ids); b=fit_mnl(X[~test],y[~test],avail[~test]); ls,ac=scores(X[test],y[test],avail[test],b);oof.append(ls);acc.append(ac)
        records.append({'model':name,'raw_parameters':int(X.shape[2]),'effective_rank':rank,'n_parameters':rank,'n_tasks':len(y),'respondents':len(people),'loglik':ll,'aic':-2*ll+2*rank,'bic':-2*ll+np.log(len(y))*rank,'in_sample_log_score':scores(X,y,avail,beta)[0],'in_sample_accuracy':scores(X,y,avail,beta)[1],'five_fold_oof_log_score':float(np.mean(oof)),'five_fold_oof_accuracy':float(np.mean(acc)),'beta':';'.join(f'{v:.9g}' for v in beta)})
    base=fits['linear'][2]
    for r in records:
        r['lr_vs_linear']=2*(r['loglik']-base);r['lr_df_vs_linear']=r['effective_rank']-records[0]['effective_rank'];r['lr_p_vs_linear']=1.0 if r['lr_df_vs_linear']<=0 else float(chi2.sf(r['lr_vs_linear'],r['lr_df_vs_linear']))
    os.makedirs(os.path.dirname(out) or '.',exist_ok=True)
    with open(out,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    signature_columns = sorted(set(d[0]) - {'choice', 'id'})
    sig=['|'.join(row[k] for k in signature_columns) for row in d]
    counts=np.bincount(ids-min(ids)); sha=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    prov=[f'parsed_csv={path}',f'parsed_csv_sha256={sha}','source_package=mlogit','source_documentation=https://search.r-project.org/CRAN/refmans/mlogit/html/Electricity.html','source_rda=https://raw.githubusercontent.com/cran/mlogit/master/data/Electricity.rda',f'rows={len(d)}',f'tasks={len(d)}',f'respondents={len(people)}','alternatives_per_task=4',f'unique_complete_menu_signatures={len(set(sig))}',f'duplicate_complete_menu_tasks={len(d)-len(set(sig))}',f'min_tasks_per_respondent={int(counts[counts>0].min())}',f'max_tasks_per_respondent={int(counts.max())}','assignment_note=public observational stated-choice archive; no candidate-preserving randomization',f'holdout_note=5-fold respondent-grouped cross-fitting with fixed seed {seed}','raw_data_note=RDA and converted CSV are local inputs and are not redistributed']
    Path(provenance_out).write_text('\n'.join(prov)+'\n')
    for r in records: print(r['model'],'oof_log',round(r['five_fold_oof_log_score'],6),'oof_acc',round(r['five_fold_oof_accuracy'],4),'LR_p',f"{r['lr_p_vs_linear']:.4g}")
    print('tasks',len(d),'respondents',len(people),'unique_menus',len(set(sig)))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('path');ap.add_argument('--out',default='results/electricity_external_validation.csv');ap.add_argument('--provenance-out',default='data/provenance_electricity_2026-10-08.md');ap.add_argument('--seed',type=int,default=20261008);run(**vars(ap.parse_args()))
