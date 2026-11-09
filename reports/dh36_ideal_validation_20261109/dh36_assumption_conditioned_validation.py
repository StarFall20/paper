"""Full-record DH36 validation under an explicit ideal-assignment scenario.

All 360 records enter every full-sample model. Original exports remain untouched.
Ideal independent participants and 180:180 random assignment are assumptions,
not claims that provenance or allocation has been verified. No source/channel
variables enter these models. The existing observational audit is preserved.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, os, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, xlogy
from scipy.stats import chi2, norm
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits
import dh36_observational_analysis as old
from dh36_panel_model import fit_panel, panel_eval, draws

SEED=20261109

def js(path,obj):
    def convert(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        raise TypeError(str(type(x)))
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=convert,allow_nan=False)+'\n')

def interval(k,n):
    z=norm.ppf(.975); p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den; half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(center-half),float(center+half)]

def perm_stats(y,z,assignments):
    """Balanced participant assignments; all eight responses move together.

For equal group sizes, the studentized Wald is a monotone transform of a
fixed pooled-covariance score. This identity saves covariance inversion in
all permutations; it is checked against direct covariance calculations.
"""
    n=len(y); m=int(z.sum()); assert n==2*m
    s=np.eye(3)[y].reshape(n,-1); keep=np.array([3*t+j for t in range(8) for j in range(2)])
    a=s[:,keep]; total=s.sum(axis=0); inv=np.linalg.pinv(np.cov(a,rowvar=False)*(2/m))
    zmat=np.atleast_2d(assignments)
    c1=zmat@s; c0=total-c1; delta=(c1-c0)/m
    score=np.einsum('bi,ij,bj->b',delta[:,keep],inv,delta[:,keep])
    w=score*2*(m-1)/np.maximum(n-1-score,1e-12)
    expected=total/2
    lr=2*(xlogy(c1,np.divide(c1,expected,out=np.ones_like(c1),where=expected>0))+xlogy(c0,np.divide(c0,expected,out=np.ones_like(c0),where=expected>0))).sum(axis=1)
    return np.column_stack([w,score,lr])

def random_assignments(rng,b,n):
    m=n//2; idx=np.argpartition(rng.random((b,n)),m-1,axis=1)[:,:m]
    a=np.zeros((b,n)); a[np.arange(b)[:,None],idx]=1
    return a

def permutation_suite(y,z,B,seed):
    obs=perm_stats(y,z,z)[0]; rng=np.random.default_rng(seed); values=[]
    for start in range(0,B,500):
        values.append(perm_stats(y,z,random_assignments(rng,min(500,B-start),len(y))))
    values=np.concatenate(values); names=['studentized_cluster_Wald','pooled_fibre_score_generalized_CMH','saturated_task_by_version_composite_LR']
    rows=[]
    for j,nm in enumerate(names):
        k=int(np.sum(values[:,j]>=obs[j]-1e-11))
        rows.append({'test':nm,'statistic':float(obs[j]),'n':len(y),'permutations':B,'exceedances':k,'permutation_p_plus_one':(k+1)/(B+1),'mc_interval_low':interval(k,B)[0],'mc_interval_high':interval(k,B)[1],'asymptotic_chi16_p':float(chi2.sf(obs[j],16)) if j<2 else None,'null':'weak marginal equality; permutation asymptotically robust' if j==0 else 'joint response-vector exchangeability for finite-sample exactness'})
    return pd.DataFrame(rows),values

def direct_wald(y,z):
    s=np.eye(3)[y][:,:,:2].reshape(len(y),16); a,b=s[z==0],s[z==1]
    delta=b.mean(0)-a.mean(0); cov=np.cov(a,rowvar=False)/len(a)+np.cov(b,rowvar=False)/len(b)
    return float(delta@np.linalg.pinv(cov)@delta)

def exact_toy():
    rng=np.random.default_rng(SEED+33); y=rng.integers(0,3,(8,8)); z=np.r_[np.zeros(4),np.ones(4)]
    aa=np.zeros((70,8))
    for i,ix in enumerate(itertools.combinations(range(8),4)): aa[i,list(ix)]=1
    # N=8 gives singular 16-d covariance. Exact enumeration uses LR, whose
    # randomization validity does not require an invertible covariance.
    vals=perm_stats(y,z,aa)[:,2]; pv=np.array([np.mean(vals>=v-1e-10) for v in vals])
    return {'assignments':70,'test':'saturated composite LR','rejection_fraction_at_0_05':float(np.mean(pv<=.05)),'max_validity_excess':float(max(np.mean(pv<=a)-a for a in [.01,.05,.1,.2,.5])),'scope':'one enumerated toy example; not a proof for all designs'}

def candidate_audit(spec):
    menus=[]; rows=[]
    for v in ['V1','V2']:
        for t,task in enumerate(spec['tasks']):
            raw=[]; phi=[]
            for j,alt in enumerate(['A','B']):
                d,s,i,e=[int(x[1:]) for x in spec['profiles'][v][alt]]
                raw.append([d,s,i,e,task['prices'][j],task['sharing'][j]])
                phi.append([d+s,i+e,task['prices'][j],task['sharing'][j]])
            raw.append([0,0,0,0,0,'outside']); phi.append([0,0,0,'outside'])
            menus.append((v,t,raw,phi))
    for t in range(8):
        x0=menus[t]; x1=menus[t+8]
        assert x0[3]==x1[3] and x0[2]!=x1[2]
        rows.append({'task':t+1,'whole_menu_candidate_equal':True,'raw_menu_distinct':True,'profiles_in_fibre':2,'probability_contrasts':2})
    c=np.zeros((8,16))
    for t in range(8): c[t,t]=1/np.sqrt(2); c[t,t+8]=-1/np.sqrt(2)
    expo=c@np.eye(16)/16@c.T
    rank=old.rank_audit(spec)
    return pd.DataFrame(rows),{'finite_design_support_points':16,'candidate_fibres':8,'rank_menu_contrast_operator':int(np.linalg.matrix_rank(c)),'rank_outcome_probability_contrasts':16,'normalized_exposure_eigenvalues':np.linalg.eigvalsh(expo).tolist(),'normalization':'uniform mass 1/16 per menu; unit Euclidean contrast rows','support_covered':'only these 16 menus; not the full attribute universe','raw_attribute_model_rank':rank['raw_model_rank'],'raw_attribute_columns':rank['raw_model_columns'],'unidentified_dimensions':rank['unidentified_dimensions']}

def feature_menus(spec):
    out=np.zeros((2,8,22))
    for v in range(2):
        for t,task in enumerate(spec['tasks']):
            vals=list(np.eye(8)[t])+[(p-89)/10 for p in task['prices']]
            for alt in ['A','B']: vals += [int(s[1:]) for s in spec['profiles'][f'V{v+1}'][alt]]
            for g in task['sharing']: vals += [int(g=='G1'),int(g=='G2')]
            out[v,t]=vals
    return out

def ml_holdout(y,z,spec,B,private,out):
    rng=np.random.default_rng(SEED+2); train=np.zeros(len(y),bool)
    for v in (0,1):
        ix=np.where(z==v)[0]; rng.shuffle(ix); train[ix[:120]]=True
    test=~train; features=feature_menus(spec)
    x=features[z[train]].reshape(-1,22); yy=y[train].ravel()
    clf=RandomForestClassifier(n_estimators=500,max_depth=6,min_samples_leaf=32,max_features=None,random_state=SEED,n_jobs=1)
    clf.fit(x,yy); q1=clf.predict_proba(features.reshape(-1,22)).reshape(2,8,3)
    counts=np.eye(3)[y[train]].sum(0); q0=(counts+.5)/(len(y[train])+1.5)
    yyh=np.eye(3)[y[test]]; n=int(test.sum()); zh=z[test]
    # Frozen nuisance and direction learned on training respondents only.
    per=np.array([np.einsum('ntj,tj->n',yyh-q0,(q1[v]-q0)/q0) for v in (0,1)]).T
    observed=float(per[np.arange(n),zh].mean()); exceed=0
    for start in range(0,B,1000):
        aa=random_assignments(rng,min(1000,B-start),n)
        stats=(per[:,0].sum()+aa@(per[:,1]-per[:,0]))/n
        exceed+=int((stats>=observed-1e-12).sum())
    qchosen=q1[zh]; loss1=-np.log(np.take_along_axis(qchosen,y[test,:,None],axis=2)[:,:,0]).mean(1)
    loss0=-np.log(np.take_along_axis(np.broadcast_to(q0,(n,8,3)),y[test,:,None],axis=2)[:,:,0]).mean(1)
    diff=loss0-loss1; se=diff.std(ddof=1)/np.sqrt(n)
    pd.DataFrame({'row_index':np.arange(len(y)),'train':train}).to_csv(private/'ideal_ML_split_private.csv',index=False)
    held_tests,_=permutation_suite(y[test],zh,B,SEED+3); held_tests.to_csv(out/'holdout_equal_sample_tests.csv',index=False)
    pd.DataFrame([{'version':f'V{v+1}','task':t+1,'alternative':j,'q0':q0[t,j],'q1':q1[v,t,j]} for v in range(2) for t in range(8) for j in range(3)]).to_csv(out/'ML_frozen_probabilities.csv',index=False)
    return {'method':'honest single split random forest residual direction; one-sided heldout randomization score','training_records':int(train.sum()),'holdout_records':n,'training_per_arm':120,'holdout_per_arm':60,'permutations':B,'score':observed,'p_plus_one':(exceed+1)/(B+1),'mc_interval':interval(exceed,B),'mean_heldout_logscore_gain':float(diff.mean()),'descriptive_ci95_gain':[float(diff.mean()-1.96*se),float(diff.mean()+1.96*se)],'features':'task identifier, menu raw attributes, price, sharing; no demographics, provenance or post-choice answers','hyperparameters':clf.get_params(),'split_rule_fixed_before_ML_holdout_evaluation':True,'claim':'Conditional exactness under heldout joint exchangeability; no exact weak-null claim.'}

def independent_mnl_check(d,base,name,fit):
    x,names=old.design(d,base,name); y=d[[f'q{i}' for i in range(1,9)]].to_numpy().reshape(-1)-1
    # Independent grouped-count objective with no use of analytic score code.
    menus,inv=np.unique(x.reshape(len(y),-1),axis=0,return_inverse=True)
    menus=menus.reshape(-1,3,len(names)); counts=np.zeros((len(menus),3)); np.add.at(counts,(inv,y),1)
    def f(b):
        u=menus@b; return -float(np.sum(counts*(u-logsumexp(u,axis=1,keepdims=True))))
    opt=minimize(f,np.zeros(len(names)),method='BFGS',options={'gtol':1e-5,'maxiter':1000})
    eps=1e-5; eye=np.eye(len(names)); fd=np.array([(f(fit['beta']+eps*e)-f(fit['beta']-eps*e))/(2*eps) for e in eye])
    u=x@fit['beta']; p=np.exp(u-logsumexp(u,axis=1,keepdims=True)); ex=np.einsum('rj,rjp->rp',p,x)
    h=np.einsum('rj,rjp,rjq->pq',p,x,x)-ex.T@ex
    return {'model':name,'independent_grouped_nll_difference':abs(f(opt.x)+fit['composite_log_likelihood']),'independent_coefficient_max_difference':float(max(abs(opt.x-fit['beta']))),'gradient_at_solution_max':float(max(abs(fd))),'hessian_min_eigenvalue':float(np.linalg.eigvalsh(h).min()),'probability_sum_error':float(max(abs(p.sum(1)-1))),'observations':len(y),'parameters':len(names)}

def information_check():
    # Three fibres, heterogeneous kappa and conditional h variances.
    p=np.array([.2,.35,.45]); k=np.array([.17,.24,.11]); h=np.array([[-1.,2.],[.3,.9],[-.4,1.2]])
    w=np.repeat(p/2,2); kx=np.repeat(k,2); hv=h.ravel(); Z=np.repeat(np.eye(3),2,axis=0)
    nuisance=Z.T@((w*kx)[:,None]*Z); cross=Z.T@(w*kx*hv)
    schur=float(np.sum(w*kx*hv**2)-cross@np.linalg.solve(nuisance,cross))
    within=float(np.sum(p*k*h.var(axis=1)))
    hnull=Z@np.array([.2,-.7,.4]); cross0=Z.T@(w*kx*hnull)
    null=float(np.sum(w*kx*hnull**2)-cross0@np.linalg.solve(nuisance,cross0))
    return {'schur_information':schur,'E_kappa_Var_h_given_phi':within,'absolute_difference':abs(schur-within),'phi_only_direction_information':null,'scope':'Bernoulli local score, kappa a function of candidate summary. This is standard nuisance projection, not a novelty proof. Panel NCP additionally requires cluster covariance.'}

def copula_y(rng,p,n,rho):
    u=norm.cdf(np.sqrt(rho)*rng.normal(size=(n,1))+np.sqrt(1-rho)*rng.normal(size=(n,8)))
    return (u[:,:,None]>p.cumsum(1)[None,:,:2]).sum(2)

def calibration_power(y,out,reps,perm_B):
    p=(np.eye(3)[y].sum(0)+.5)/(len(y)+1.5); rows=[]; rng=np.random.default_rng(SEED+4)
    scenarios=[('joint_null_independent',0.,0.),('joint_null_correlated',.5,.5),('weak_null_unequal_dependence',.05,.8)]
    for scenario,r0,r1 in scenarios:
        rejects=np.zeros(3,int); pasym=0
        for rep in range(reps):
            yy=np.vstack([copula_y(rng,p,180,r0),copula_y(rng,p,180,r1)]); z=np.r_[np.zeros(180),np.ones(180)]
            obs=perm_stats(yy,z,z)[0]; vals=perm_stats(yy,z,random_assignments(rng,perm_B,360))
            pv=(1+(vals>=obs-1e-10).sum(0))/(perm_B+1); rejects+=pv<=.05; pasym+=chi2.sf(obs[0],16)<=.05
        for j,nm in enumerate(['studentized_Wald_permutation','pooled_score_permutation','saturated_composite_LR_permutation']):
            ci=interval(rejects[j],reps); rows.append({'scenario':scenario,'test':nm,'n':360,'MC_repetitions':reps,'permutations_per_rep':perm_B,'rejection_rate':rejects[j]/reps,'mc_low':ci[0],'mc_high':ci[1],'interpretation':'size for joint null' if r0==r1 else ('size for weak marginal null' if j==0 else 'joint null false; not a weak-null size estimate')})
        ci=interval(pasym,reps); rows.append({'scenario':scenario,'test':'studentized_Wald_chi16','n':360,'MC_repetitions':reps,'permutations_per_rep':0,'rejection_rate':pasym/reps,'mc_low':ci[0],'mc_high':ci[1],'interpretation':'size for weak marginal null'})
        print('calibration',scenario,flush=True)
    pd.DataFrame(rows).to_csv(out/'null_calibration.csv',index=False)
    rows=[]
    for direction in ['all_tasks_A_vs_B','one_task_A_vs_B','all_tasks_service_vs_optout']:
        for n in [360,600,800,1200]:
            for delta in [0.,.03,.05,.08,.10]:
                p0=p.copy(); p1=p.copy(); change=np.zeros_like(p)
                if direction=='all_tasks_A_vs_B': change[:,0]=1; change[:,1]=-1
                elif direction=='one_task_A_vs_B': change[1,0]=1; change[1,1]=-1
                else: change[:,0]=.5; change[:,1]=.5; change[:,2]=-1
                p0-=delta/2*change; p1+=delta/2*change; assert min(p0.min(),p1.min())>0
                k=0; z=np.r_[np.zeros(n//2),np.ones(n//2)]
                for rep in range(reps):
                    yy=np.vstack([copula_y(rng,p0,n//2,.5),copula_y(rng,p1,n//2,.5)])
                    k+=chi2.sf(perm_stats(yy,z,z)[0,0],16)<=.05
                ci=interval(k,reps); rows.append({'direction':direction,'n':n,'max_probability_difference':delta,'rho_latent_copula':.5,'repetitions':reps,'power_or_size':k/reps,'mc_low':ci[0],'mc_high':ci[1],'test':'cluster_Wald_chi16; ideal assignment; hypothetical effects'})
        print('power',direction,flush=True)
    pd.DataFrame(rows).to_csv(out/'prospective_power_grid.csv',index=False)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',type=Path,required=True); ap.add_argument('--instrument',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--permutations',type=int,default=49999); ap.add_argument('--mc-repetitions',type=int,default=500); ap.add_argument('--calibration-permutations',type=int,default=199); args=ap.parse_args()
    out=args.out; out.mkdir(parents=True,exist_ok=True); spec=json.loads(args.instrument.read_text()); started=time.time()
    raw_hash={v:hashlib.sha256((args.input_dir/f'{v}_raw.xlsx').read_bytes()).hexdigest() for v in ['V1','V2']}
    dd,*_=old.read_exports(args.input_dir,spec)
    # Explicitly restrict model data: retain all records; no source/channel covariates.
    d=dd[['participant_id','version','version2']+[f'q{t}' for t in range(1,9)]].copy()
    assert len(d)==360 and d.participant_id.nunique()==360
    y=d[[f'q{t}' for t in range(1,9)]].to_numpy(int)-1; z=d.version2.to_numpy(int); assert z.sum()==180
    base=old.menu_arrays(spec)
    assumptions={'analysis_population':'all 360 exported records, no exclusions','independent_response_records':'assumed; not verified identity or human provenance','assignment':'ideal complete respondent-level 180:180 random assignment, assumed for conditional inference','interference':'none assumed','profile_display':'unchanged questionnaire and whole-menu candidate preservation','source_metadata':'raw provenance retained unchanged; not converted to verified human labels','factual_provenance_reference':'../dh36_model_20261108/DH36_模型与数据核查报告.md','causal_claims':'conditional only; source and actual allocation cannot be validated by fitting a model','analysis_timing':'analysis developed after inspecting outcomes; exploratory, no preregistration claimed'}
    js(out/'assumptions.json',assumptions)
    ca,cert=candidate_audit(spec); ca.to_csv(out/'whole_menu_candidate_audit.csv',index=False); js(out/'support_certificate.json',cert)
    tests,perm=permutation_suite(y,z,args.permutations,SEED); tests.to_csv(out/'full_sample_randomization_tests.csv',index=False)
    # Independent algebra and coding checks, not just optimizer status.
    rng=np.random.default_rng(SEED+77); errs=[]
    for a in random_assignments(rng,100,360): errs.append(abs(direct_wald(y,a)-perm_stats(y,z,a)[0,0]))
    swap=perm_stats(2-y,1-z,1-z)[0]; orig=perm_stats(y,z,z)[0]
    verification={'records':len(y),'tasks_per_record':8,'choice_rows':y.size,'alternative_rows':y.size*3,'dropped_records':0,'raw_sha256':raw_hash,'balanced_Wald_identity_max_error_100_splits':max(errs),'group_and_category_relabeling_max_stat_error':float(max(abs(orig-swap))),'exact_enumeration':exact_toy(),'information_projection':information_check()}
    print('full tests',tests[['test','statistic','permutation_p_plus_one']].to_dict('records'),flush=True)
    ml=ml_holdout(y,z,spec,args.permutations,args.input_dir,out); js(out/'ML_honest_holdout.json',ml)
    coeff=[]; fits={}; mchecks=[]
    for name in ['mnl_common','mnl_cohort_intercepts','mnl_cohort_all']:
        f=old.fit_mnl(d,base,name,validation=True); fits[name]=f
        for row in f['parameters']: coeff.append({'model':name,**row})
        mchecks.append(independent_mnl_check(d,base,name,f))
    pd.DataFrame(coeff).rename(columns={'p_value_association_only':'p_conditional_model'}).to_csv(out/'MNL_coefficients.csv',index=False)
    js(out/'MNL_validation.json',{'independent_checks':mchecks,'models':{n:{k:f[k] for k in ['rank','gradient_max','solver_success','composite_log_likelihood','test']} for n,f in fits.items()}})
    cv,folds=old.predictive_validation(d,base,SEED); cv.drop(columns='label').to_csv(out/'respondent_5fold_marginal_prediction.csv',index=False)
    pd.DataFrame({'participant_id':d.participant_id,'fold':folds}).to_csv(args.input_dir/'ideal_cv_folds_private.csv',index=False)
    panelrows=[]; panel_fits={}; panel_checks={}
    for interaction in [False,True]:
        name='panel_mixed_version_ASC' if interaction else 'panel_mixed_common'
        fit=fit_panel(y,z,base,power=8,interaction=interaction)
        final=fit_panel(y,z,base,power=10,interaction=interaction,start=fit['beta'],multistart=False)
        check=fit_panel(y,z,base,power=10,seed=SEED+1,interaction=interaction,start=final['beta'],multistart=False)
        panel_fits[name]=final
        panel_checks[name]={'draw256':fit['validation'],'draw1024':final['validation'],'independent_scramble1024':check['validation'],'loglik_draw256':fit['loglik'],'loglik_draw1024':final['loglik'],'loglik_independent_scramble1024':check['loglik'],'coefficient_scramble_max_diff':float(max(abs(final['beta']-check['beta'])))}
        for row in final['parameters']: panelrows.append({'model':name,**row})
        print(name,'LL',final['loglik'],'gradient',final['validation']['projected_gradient_max'],flush=True)
    pd.DataFrame(panelrows).to_csv(out/'panel_mixed_logit_coefficients.csv',index=False); js(out/'panel_mixed_logit_validation.json',panel_checks)
    # Identical respondent folds, joint 8-answer log score. Marginal MNL is
    # evaluated as a product of its task probabilities for this comparison.
    joint={name:np.zeros(len(y)) for name in ['MNL_product','panel_mixed_common','panel_mixed_version_ASC']}
    for k in range(5):
        tr=folds!=k; te=~tr
        f=old.fit_mnl(d.loc[tr],base,'mnl_common'); x,_=old.design(d.loc[te],base,'mnl_common'); u=x@f['beta']; lp=(u-logsumexp(u,axis=1,keepdims=True)).reshape(te.sum(),8,3)
        joint['MNL_product'][te]=np.take_along_axis(lp,y[te,:,None],axis=2)[:,:,0].sum(1)
        for interaction in [False,True]:
            name='panel_mixed_version_ASC' if interaction else 'panel_mixed_common'
            f=fit_panel(y[tr],z[tr],base,power=8,interaction=interaction,start=panel_fits[name]['beta'],multistart=False)
            if not f['validation']['solver_success'] or f['validation']['projected_gradient_max']>1e-3: raise RuntimeError('CV panel failed convergence')
            joint[name][te]=panel_eval(f['beta'],y[te],z[te],base,draws(10),interaction,True)[3]
        print('panel CV fold',k,flush=True)
    jr=[]
    for nm,vals in joint.items():
        diff=(vals-joint['MNL_product'])/8; se=diff.std(ddof=1)/np.sqrt(len(y))
        jr.append({'model':nm,'joint_negative_logscore_per_record':float(-vals.mean()),'joint_negative_logscore_divided_by_8':float(-vals.mean()/8),'gain_vs_MNL_per_task':float(diff.mean()),'descriptive_ci_low':float(diff.mean()-1.96*se),'descriptive_ci_high':float(diff.mean()+1.96*se),'folds':5,'prediction_target':'joint full response vector, new respondent'})
    pd.DataFrame(jr).to_csv(out/'respondent_5fold_joint_prediction.csv',index=False)
    calibration_power(y,out,args.mc_repetitions,args.calibration_permutations)
    assert all(hashlib.sha256((args.input_dir/f'{v}_raw.xlsx').read_bytes()).hexdigest()==raw_hash[v] for v in raw_hash)
    assert verification['balanced_Wald_identity_max_error_100_splits']<1e-8
    assert verification['group_and_category_relabeling_max_stat_error']<1e-8
    assert all(c['independent_coefficient_max_difference']<2e-4 and c['hessian_min_eigenvalue']>0 for c in mchecks)
    assert all(c['draw1024']['projected_gradient_max']<1e-3 and c['draw1024']['gradient_check_error']<1e-4 for c in panel_checks.values())
    verification['elapsed_seconds']=time.time()-started; verification['completed_numerical_checks']=True
    js(out/'technical_verification.json',verification)
    js(out/'environment.json',{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'seed':SEED,'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}})
    print('COMPLETE',out,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1): main()
