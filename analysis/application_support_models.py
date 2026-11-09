"""Exploratory price-support models; no source rewriting or human certification."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp,expit
from scipy.stats import norm
from threadpoolctl import threadpool_limits
import dh36_observational_analysis as old
from questionnaire_reuse_audit import fit as fit_full


def save(path,obj):
    def cv(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=cv,allow_nan=False)+'\n')


def build(spec,people):
    base=old.menu_arrays(spec); flat,names=old.design(people,base,'mnl_cohort_intercepts'); x=flat.reshape(len(people),8,3,7)
    cells=sorted({(task['sharing'][j],task['prices'][j]) for task in spec['tasks'] for j in range(2)})
    ref=('G1',89); active=[c for c in cells if c!=ref]
    cell=np.zeros((len(people),8,3,len(active)))
    for t,task in enumerate(spec['tasks']):
        for j in range(2):
            key=(task['sharing'][j],task['prices'][j])
            if key!=ref: cell[:,t,j,active.index(key)]=1
    xx=np.concatenate([x[:,:,:,:2],x[:,:,:,5:7],cell],axis=3)
    names=['ASC_A','ASC_B','V2_x_ASC_A','V2_x_ASC_B']+[f'{g}_fee_{p}' for g,p in active]
    xc=np.concatenate([x, x[:,:,:,2:3]**2],axis=3)
    # Categorical price + sharing parameterization and its null vector.
    pp=np.zeros((len(people),8,3,6)); levels=[59,69,79,99,109,119]
    for t,task in enumerate(spec['tasks']):
        for j in range(2):
            for k,lev in enumerate(levels): pp[:,t,j,k]=task['prices'][j]==lev
    cat=np.concatenate([x[:,:,:,:2],x[:,:,:,3:5],x[:,:,:,5:7],pp],axis=3)
    cn=['ASC_A','ASC_B','G1','G2','V2_ASC_A','V2_ASC_B']+[f'fee_{v}' for v in levels]
    dcat=(cat[:,:,:-1]-cat[:,:,-1:]).reshape(-1,len(cn)); _,ss,vh=np.linalg.svd(dcat,full_matrices=False); null=vh[-1]; null/=max(abs(null))
    assert np.max(abs(dcat@null))<1e-10
    assert np.linalg.matrix_rank(dcat)==11 and np.linalg.matrix_rank((xx[:,:,:-1]-xx[:,:,-1:]).reshape(-1,11))==11
    coef=np.linalg.lstsq(dcat,(xx[:,:,:-1]-xx[:,:,-1:]).reshape(-1,11),rcond=None)[0]
    assert np.max(abs(dcat@coef-(xx[:,:,:-1]-xx[:,:,-1:]).reshape(-1,11)))<1e-10
    return {'linear':x,'quadratic':xc,'cells':xx},names,active,{'categorical_columns':cn,'categorical_rank':11,'categorical_columns_count':12,'normalized_null_vector':null,'observed_cell_parameterization_rank':11,'linear_price_rank':7,'quadratic_price_rank':int(np.linalg.matrix_rank((xc[:,:,:-1]-xc[:,:,-1:]).reshape(-1,8))),'same_observed_utility_space_verified':True}


def fit_ab(x,y):
    n,T,J,K=x.shape
    # Common service intercept cancels; retain two version-specific A/B biases.
    columns=np.array([0,2]+list(range(4,K)))
    diff=(x[:,:,0,:]-x[:,:,1,:])[:,:,columns]; mask=y!=2; X=diff[mask]; Y=(y[mask]==0).astype(float)
    def f(b):
        u=X@b; p=expit(u); return float(np.sum(np.logaddexp(0,u)-Y*u)),X.T@(p-Y)
    opt=minimize(f,np.zeros(X.shape[1]),jac=True,method='BFGS',options={'gtol':1e-8,'maxiter':1500}); b=opt.x
    for _ in range(5):
        val,g=f(b)
        if max(abs(g))<1e-7: break
        p=expit(X@b); h=X.T@((p*(1-p))[:,None]*X); b-=np.linalg.solve(h,g)
    val,g=f(b); assert max(abs(g))<1e-5
    p=expit(X@b); h=X.T@((p*(1-p))[:,None]*X); rr=X*(Y-p)[:,None]; cluster=np.zeros((n,len(b))); np.add.at(cluster,np.where(mask)[0],rr)
    contributing=int(mask.any(1).sum()); hi=np.linalg.inv(h); cov=contributing/(contributing-1)*(len(Y)-1)/(len(Y)-len(b))*hi@cluster.T@cluster@hi
    return b,cov,columns,{'records_retained':n,'records_contributing_A_B_choices':contributing,'service_choice_rows':len(Y),'parameters':len(b),'rank':int(np.linalg.matrix_rank(X)),'gradient_max':float(max(abs(g))),'likelihood_target':'A vs B conditional on selecting either service; all-outside records contribute zero A/B likelihood','scope':'allows arbitrary common service propensity under the shared-slope conditional logit assumption; not arbitrary random taste or source validity'}


def cell_contrast(b,cov,names,positive,negative=('G1',89)):
    c=np.zeros(len(b))
    for key,sgn in [(positive,1),(negative,-1)]:
        if key==('G1',89): continue
        c[names.index(f'{key[0]}_fee_{key[1]}')]+=sgn
    effect=float(c@b); se=float(np.sqrt(c@cov@c))
    return {'positive_rule':positive[0],'positive_fee':positive[1],'reference_rule':negative[0],'reference_fee':negative[1],'utility_difference':effect,'cluster_se':se,'ci_low':effect-1.95996398454*se,'ci_high':effect+1.95996398454*se,'p_exploratory_model_conditional':float(2*norm.sf(abs(effect/se)))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',type=Path,required=True); ap.add_argument('--instrument',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args(); out=args.out; out.mkdir(parents=True,exist_ok=True)
    spec=json.loads(args.instrument.read_text()); d,*_=old.read_exports(args.input_dir,spec); models,names,active,audit=build(spec,d); y=d[[f'q{i}' for i in range(1,9)]].to_numpy(int)-1
    edges=[]
    for t,task in enumerate(spec['tasks']):
        for j in range(2): edges.append({'task':t+1,'alternative':'AB'[j],'price':task['prices'][j],'sharing':task['sharing'][j]})
    pd.DataFrame(edges).to_csv(out/'exact_fee_sharing_support.csv',index=False)
    audit['price_sets_by_rule']={g:sorted({row['price'] for row in edges if row['sharing']==g}) for g in ['G0','G1','G2']}
    audit['G1_G2_common_prices']=sorted(set(audit['price_sets_by_rule']['G1'])&set(audit['price_sets_by_rule']['G2']))
    audit['G0_other_common_prices']=sorted(set(audit['price_sets_by_rule']['G0'])&(set(audit['price_sets_by_rule']['G1'])|set(audit['price_sets_by_rule']['G2'])))
    save(out/'price_support_identification.json',audit)
    rows=[]; coefs=[]; fits={}
    for name,x in models.items():
        b,cov,check=fit_full(x,y); fits[name]={'beta':b,'covariance':cov,'checks':check}
        if name=='cells':
            for pos in [('G2',89),('G2',119),('G2',59)]: rows.append({'model':'unrestricted_observed_cells',**cell_contrast(b,cov,names,pos)})
            for j,nm in enumerate(names): coefs.append({'model':'unrestricted_observed_cells','term':nm,'estimate':b[j],'cluster_se':np.sqrt(cov[j,j])})
    b,cov,columns,check=fit_ab(models['cells'],y); abnames=[names[j] for j in columns]; fits['conditional_AB']={'beta':b,'covariance':cov,'checks':check,'names':abnames}
    for pos in [('G2',89),('G2',119),('G2',59)]: rows.append({'model':'conditional_A_B_observed_cells',**cell_contrast(b,cov,abnames,pos)})
    pd.DataFrame(rows).to_csv(out/'common_price_and_supported_fee_contrasts.csv',index=False); pd.DataFrame(coefs).to_csv(out/'observed_cell_coefficients.csv',index=False); save(out/'support_model_fits.json',fits)
    # Exact rank after removing either common-price bridge task.
    jr=[]
    for omit in range(8):
        keep=np.arange(8)!=omit; xx=models['cells'][:,keep]; rank=int(np.linalg.matrix_rank((xx[:,:,:-1]-xx[:,:,-1:]).reshape(-1,11)))
        c=np.zeros(11); c[names.index('G2_fee_89')]=1; D=(xx[:,:,:-1]-xx[:,:,-1:]).reshape(-1,11)
        rank_aug=int(np.linalg.matrix_rank(np.vstack([D,c])))
        jr.append({'omitted_task':omit+1,'cell_model_rank':rank,'G2_vs_G1_at_89_estimable':rank_aug==rank})
    pd.DataFrame(jr).to_csv(out/'common_price_bridge_sensitivity.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1): main()
