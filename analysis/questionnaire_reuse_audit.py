"""Reuse feasibility for the existing health-service instrument, not a new survey.
All 360 records enter the diagnostic fits; provenance stays unchanged. Monetary
trade-offs are conditional model outputs, not verified population WTP estimates.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm
from threadpoolctl import threadpool_limits
import dh36_observational_analysis as old


def fit(x,y):
    n,T,J,K=x.shape; xx=x.reshape(n*T,J,K); yy=y.ravel()
    rank=int(np.linalg.matrix_rank((xx[:,:-1]-xx[:,-1:]).reshape(-1,K)))
    if rank!=K: raise ValueError('unidentified model')
    opt=minimize(old.objective,np.zeros(K),args=(xx,yy),jac=True,method='BFGS',options={'gtol':1e-7,'maxiter':1500})
    b=opt.x
    for _ in range(8):
        val,g=old.objective(b,xx,yy)
        if max(abs(g))<1e-7: break
        p=np.exp(xx@b-logsumexp(xx@b,axis=1,keepdims=True)); ex=np.einsum('rj,rjk->rk',p,xx); h=np.einsum('rj,rjk,rjl->kl',p,xx,xx)-ex.T@ex
        step=np.linalg.solve(h,g)
        for frac in [1.,.5,.25,.125]:
            trial=b-frac*step
            if old.objective(trial,xx,yy)[0]<=val+1e-9: b=trial; break
    val,g=old.objective(b,xx,yy); assert max(abs(g))<1e-5
    p=np.exp(xx@b-logsumexp(xx@b,axis=1,keepdims=True)); ex=np.einsum('rj,rjk->rk',p,xx); h=np.einsum('rj,rjk,rjl->kl',p,xx,xx)-ex.T@ex
    ss=(xx[np.arange(len(yy)),yy]-ex).reshape(n,T,K).sum(1); hi=np.linalg.inv(h); cov=n/(n-1)*(n*T-1)/(n*T-K)*hi@ss.T@ss@hi
    return b,cov,{'records':n,'tasks_per_record':T,'parameters':K,'rank':rank,'gradient_max':float(max(abs(g))),'hessian_min_eigenvalue':float(np.linalg.eigvalsh(h).min())}


def fieller(beta,cov,c,d):
    A=float(c@beta); B=float(d@beta); va=float(c@cov@c); vb=float(d@cov@d); cab=float(c@cov@d); z=norm.ppf(.975)
    aa=B*B-z*z*vb; bb=-2*A*B+2*z*z*cab; cc=A*A-z*z*va; disc=bb*bb-4*aa*cc
    bounded=aa>0 and disc>=0
    roots=sorted([(-bb-np.sqrt(disc))/(2*aa),(-bb+np.sqrt(disc))/(2*aa)]) if bounded else [None,None]
    return {'monthly_CNY_model_tradeoff':A/B if abs(B)>1e-14 else None,'fieller_bounded':bool(bounded),'fieller_low':roots[0],'fieller_high':roots[1],'scope':'conditional all-record model diagnostic; not verified human-population WTP'}


def contrasts(name,b,cov):
    rows=[]; K=len(b)
    for label,weights in [('G1_vs_no_external_sharing',[-1,0]),('G2_vs_no_external_sharing',[0,1]),('G2_vs_G1',[-1,1])]:
        c=np.zeros(K)
        # First row is a signed G1-vs-G0 utility value, not a reversed contrast.
        c[3:5]=[1,0] if label.startswith('G1_') else weights
        effect=float(c@b); se=float(np.sqrt(c@cov@c)); c_money=c*10; den=np.zeros(K); den[2]=-1
        rows.append({'model':name,'contrast':label,'utility_difference':effect,'cluster_se':se,'p_exploratory_conditional_model':float(2*norm.sf(abs(effect/se))),**fieller(b,cov,c_money,den)})
    return rows


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',type=Path,required=True); ap.add_argument('--instrument',type=Path,required=True); ap.add_argument('--saved-panel-cov',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args(); out=args.out; out.mkdir(parents=True,exist_ok=True)
    spec=json.loads(args.instrument.read_text()); people,provenance,*_=old.read_exports(args.input_dir,spec); d=people[['participant_id','version2']+[f'q{i}' for i in range(1,9)]].copy()
    y=d[[f'q{i}' for i in range(1,9)]].to_numpy(int)-1; base=old.menu_arrays(spec); flat,names=old.design(d,base,'mnl_cohort_intercepts'); x=flat.reshape(len(d),8,3,7)
    t=np.linspace(-1,1,8); trend=np.zeros((len(d),8,3,2)); trend[:,:,0,0]=t; trend[:,:,1,1]=t
    models={'version_bundle_intercepts':x,'common_participation_order_trend':np.concatenate([x,trend.sum(3,keepdims=True)],axis=3),'alternative_specific_order_trends':np.concatenate([x,trend],axis=3)}
    rows=[]; details={}; coefficients=[]
    for name,xx in models.items():
        b,cov,check=fit(xx,y); rows.extend(contrasts(name,b,cov)); details[name]=check
        for j,val in enumerate(b): coefficients.append({'model':name,'term':names[j] if j<7 else 'order_trend_'+str(j-6),'estimate':val,'cluster_se':np.sqrt(cov[j,j])})
    # Reuse saved panel covariance instead of refitting it or searching models.
    saved=json.loads(args.saved_panel_cov.read_text())['random_intercept_version_ASC']; rows.extend(contrasts('saved_panel_random_intercept',np.array(saved['beta']),np.array(saved['covariance'])))
    jack=[]
    for omitted in range(8):
        keep=np.arange(8)!=omitted; b,cov,check=fit(x[:,keep],y[:,keep]); rr=contrasts('omit_task_'+str(omitted+1),b,cov)
        for r in rr: jack.append({'omitted_task':omitted+1,**r})
    # Alternative-by-task constants absorb all three price/sharing regressors.
    task=np.zeros((len(d),8,3,14))
    for j in range(7): task[:,j+1,0,2*j]=1; task[:,j+1,1,2*j+1]=1
    saturated=np.concatenate([x,task],axis=3); dd=(saturated[:,:,:-1]-saturated[:,:,-1:]).reshape(-1,21); r=int(np.linalg.matrix_rank(dd)); assert r==18
    pd.DataFrame(rows).to_csv(out/'conditional_sharing_tradeoffs.csv',index=False); pd.DataFrame(coefficients).to_csv(out/'fee_sharing_and_order_models.csv',index=False); pd.DataFrame(jack).to_csv(out/'leave_one_task_sensitivity.csv',index=False)
    result={'population':360,'source_based_models_not_split':True,'price_and_sharing_model_rank':7,'with_common_order_trend_rank':8,'with_alternative_order_trends_rank':9,'with_saturated_alternative_task_effects_columns':21,'with_saturated_alternative_task_effects_rank':r,'lost_directions_due_to_saturated_task_effects':3,'interpretation':'price/sharing are estimable under specified preference/order models; arbitrary alternative-by-task effects absorb them','models':details,'raw_hashes_unchanged':all(hashlib.sha256((args.input_dir/f"{p['version']}_raw.xlsx").read_bytes()).hexdigest()==p['sha256'] for p in provenance),'not_a_preregistered_confirmation':True,'actual_human_provenance_not_certified':True}
    assert result['raw_hashes_unchanged']; (out/'reuse_identification_and_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n'); print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1): main()
